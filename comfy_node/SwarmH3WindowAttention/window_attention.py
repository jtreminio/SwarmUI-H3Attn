"""T2VA/FL2VA-only local-window attention prototype for MiniMax H3.

H3 packs one sequence as [text | keyframe conditioning | audio | video].
The configured dense backend handles conditioning query rows against the full
packed sequence. FlexAttention handles only video query rows, which attend the
global prefix plus a centered temporal window. Selected transformer layers
remain fully dense so information can cross the complete clip.

The prototype uses PyTorch FlexAttention's compiled BlockMask path. It never
constructs an SxS eager mask: any layout, compilation, or kernel failure falls
back to the attention implementation that was already installed on the model.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import logging
from typing import Any, Callable

import torch

import comfy.patcher_extension
from comfy_api.latest import io

try:
    from torch.nn.attention.flex_attention import BlockMask, flex_attention

    _compiled_flex_attention = torch.compile(flex_attention, fullgraph=True, dynamic=False)
    _FLEX_IMPORT_ERROR: Exception | None = None
except Exception as exc:  # Keep the node loadable so execution reports a useful error.
    BlockMask = None
    _compiled_flex_attention = None
    _FLEX_IMPORT_ERROR = exc


_RUNTIME_KEY = "h3_window_attention_runtime"
_OVERRIDE_KEY = "optimized_attention_override"
_MASK_BLOCK_SIZE = 128
_MAX_MASK_CACHE = 8
_MASK_CACHE: OrderedDict[tuple[Any, ...], Any] = OrderedDict()
_LOGGED: set[tuple[Any, ...]] = set()


@dataclass(frozen=True)
class WindowConfig:
    """Immutable options captured by one patched model clone."""

    window_seconds: float
    window_frames: int
    dense_layers: frozenset[int]
    transformer_layers: int
    verbose: bool


@dataclass
class WindowRuntime:
    """Per-forward H3 layout and transformer-call counter."""

    video_start: int
    video_stop: int
    latent_frames: int
    frame_rows: int
    layer: int = 0
    disabled: bool = False


def _log_once(key: tuple[Any, ...], message: str, level: int = logging.INFO) -> None:
    """Log one diagnostic once per process instead of once per transformer layer."""
    if key not in _LOGGED:
        _LOGGED.add(key)
        logging.log(level, f"[h3_window_attention] {message}")


def _parse_dense_layers(spec: str, count: int) -> frozenset[int]:
    """Parse a comma-separated layer list; negative indices count from the end."""
    layers: set[int] = set()
    for raw in spec.split(","):
        value = raw.strip()
        if not value:
            continue
        try:
            layer = int(value)
        except ValueError as exc:
            raise ValueError(
                f"dense_layers entry {value!r} is not an integer; use values like "
                "'0,9,19,29,39,49'"
            ) from exc
        if layer < 0:
            layer += count
        if layer < 0 or layer >= count:
            raise ValueError(f"dense layer {value} resolves outside 0..{count - 1}")
        layers.add(layer)
    return frozenset(layers)


def _window_frames(seconds: float) -> int:
    """Convert H3 seconds to an odd latent-frame window (5 seconds -> 37)."""
    frames = max(1, int(round(float(seconds) * 7.0)) + 2)
    if frames % 2 == 0:
        frames += 1
    return frames


def _extract_transformer_options(args: tuple[Any, ...], kwargs: dict[str, Any]) -> dict[str, Any] | None:
    """Find the transformer-options dict in H3's diffusion-model wrapper call."""
    options = kwargs.get("transformer_options")
    if isinstance(options, dict):
        return options
    if len(args) > 3 and isinstance(args[3], dict):
        return args[3]
    return None


def _runtime_from_payload(payload: dict[str, Any] | None, config: WindowConfig) -> WindowRuntime | None:
    """Resolve the T2VA/FL2VA video span, returning None for unsupported layouts."""
    if not isinstance(payload, dict):
        _log_once(("payload",), "dense fallback: MiniMax payload is unavailable")
        return None
    if payload.get("refs"):
        _log_once(("refs",), "dense fallback: reference-video mode is outside the prototype scope")
        return None
    layout = payload.get("layout")
    segments = getattr(layout, "segments", None)
    signature = getattr(layout, "signature", None)
    if not segments or not signature or len(signature) < 4:
        _log_once(("layout",), "dense fallback: packed H3 layout is unavailable")
        return None
    video_segments = [(int(start), int(stop)) for start, stop, kind in segments if kind == "video"]
    if len(video_segments) != 1:
        _log_once(("video_segment", len(video_segments)),
                  "dense fallback: expected one target-video segment")
        return None
    video_start, video_stop = video_segments[0]
    latent_frames = int(signature[1])
    video_rows = video_stop - video_start
    if latent_frames <= 0 or video_rows <= 0 or video_rows % latent_frames != 0:
        _log_once(("video_shape", latent_frames, video_rows),
                  "dense fallback: target-video rows do not divide into latent frames")
        return None
    if video_stop != int(getattr(layout, "seq_len", video_stop)):
        _log_once(("video_not_last", video_stop, getattr(layout, "seq_len", None)),
                  "dense fallback: target video is not the final packed segment")
        return None
    if latent_frames <= config.window_frames:
        return None
    return WindowRuntime(
        video_start=video_start,
        video_stop=video_stop,
        latent_frames=latent_frames,
        frame_rows=video_rows // latent_frames,
    )


def _make_forward_wrapper(config: WindowConfig) -> Callable[..., Any]:
    """Publish one forward's packed layout while the H3 transformer executes."""
    def wrapper(executor, *args, **kwargs):
        options = _extract_transformer_options(args, kwargs)
        if options is None:
            return executor(*args, **kwargs)
        runtime = _runtime_from_payload(kwargs.get("minimax_payload"), config)
        sentinel = object()
        previous = options.get(_RUNTIME_KEY, sentinel)
        if runtime is not None:
            options[_RUNTIME_KEY] = runtime
            if config.verbose:
                _log_once(
                    ("active", runtime.video_start, runtime.video_stop,
                     runtime.latent_frames, runtime.frame_rows, config.window_frames,
                     tuple(sorted(config.dense_layers))),
                    f"active: {runtime.latent_frames} latent frames, "
                    f"{runtime.frame_rows} rows/frame, {config.window_frames}-frame window, "
                    f"dense layers {sorted(config.dense_layers)}",
                )
        try:
            return executor(*args, **kwargs)
        finally:
            if previous is sentinel:
                options.pop(_RUNTIME_KEY, None)
            else:
                options[_RUNTIME_KEY] = previous

    return wrapper


def _video_block_rows(
    runtime: WindowRuntime,
    window_frames: int,
) -> tuple[list[list[int]], list[list[int]]]:
    """Return partial and full KV-block rows for video queries.

    FlexAttention applies ``mask_mod`` only to partial blocks. A block is full
    when every real query/key pair in it is allowed by the temporal policy.
    Final short blocks deliberately stay partial so padding is never declared
    valid.
    """
    video_rows = runtime.video_stop - runtime.video_start
    query_block_count = (video_rows + _MASK_BLOCK_SIZE - 1) // _MASK_BLOCK_SIZE
    prefix_block_count = (runtime.video_start + _MASK_BLOCK_SIZE - 1) // _MASK_BLOCK_SIZE
    radius = window_frames // 2
    partial_rows: list[list[int]] = []
    full_rows: list[list[int]] = []

    for query_block in range(query_block_count):
        query_start = query_block * _MASK_BLOCK_SIZE
        query_stop = min(video_rows, query_start + _MASK_BLOCK_SIZE) - 1
        first_query_frame = query_start // runtime.frame_rows
        last_query_frame = query_stop // runtime.frame_rows
        first_key_frame = max(0, first_query_frame - radius)
        last_key_frame = min(runtime.latent_frames - 1, last_query_frame + radius)
        first_local_token = runtime.video_start + first_key_frame * runtime.frame_rows
        last_local_token = runtime.video_start + (last_key_frame + 1) * runtime.frame_rows - 1
        first_local_block = first_local_token // _MASK_BLOCK_SIZE
        last_local_block = last_local_token // _MASK_BLOCK_SIZE
        key_blocks = list(range(prefix_block_count))
        key_blocks.extend(
            range(max(prefix_block_count, first_local_block), last_local_block + 1)
        )

        query_block_is_complete = query_stop - query_start + 1 == _MASK_BLOCK_SIZE
        partial: list[int] = []
        full: list[int] = []
        for key_block in key_blocks:
            key_start = key_block * _MASK_BLOCK_SIZE
            key_stop = min(runtime.video_stop, key_start + _MASK_BLOCK_SIZE) - 1
            key_block_is_complete = key_stop - key_start + 1 == _MASK_BLOCK_SIZE

            fully_valid = False
            if query_block_is_complete and key_block_is_complete:
                if key_stop < runtime.video_start:
                    fully_valid = True
                elif key_start >= runtime.video_start:
                    first_block_key_frame = (
                        key_start - runtime.video_start
                    ) // runtime.frame_rows
                    last_block_key_frame = (
                        key_stop - runtime.video_start
                    ) // runtime.frame_rows
                    fully_valid = (
                        first_block_key_frame >= last_query_frame - radius
                        and last_block_key_frame <= first_query_frame + radius
                    )

            (full if fully_valid else partial).append(key_block)

        partial_rows.append(partial)
        full_rows.append(full)

    return partial_rows, full_rows


def _block_tensors(
    rows: list[list[int]],
    device: torch.device,
    width: int | None = None,
) -> tuple[Any, Any]:
    """Convert ragged block rows to FlexAttention's count/index tensors."""
    counts = [len(row) for row in rows]
    required_width = max(1, max(counts, default=0))
    if width is None:
        width = required_width
    elif width < required_width:
        raise ValueError(f"block metadata width {width} is smaller than {required_width}")
    padded = [row + [0] * (width - len(row)) for row in rows]
    row_count = len(rows)
    num_blocks = torch.tensor(
        counts, dtype=torch.int32, device=device
    ).reshape(1, 1, row_count)
    indices = torch.tensor(
        padded, dtype=torch.int32, device=device
    ).reshape(1, 1, row_count, width)
    return num_blocks, indices


def _video_mask_for(runtime: WindowRuntime, config: WindowConfig, device: torch.device):
    """Build and cache compact block metadata for local video query rows."""
    key = (
        device.type,
        device.index,
        runtime.video_start,
        runtime.video_stop,
        runtime.latent_frames,
        runtime.frame_rows,
        config.window_frames,
    )
    cached = _MASK_CACHE.get(key)
    if cached is not None:
        _MASK_CACHE.move_to_end(key)
        return cached

    video_start = runtime.video_start
    video_stop = runtime.video_stop
    video_rows = video_stop - video_start
    frame_rows = runtime.frame_rows
    radius = config.window_frames // 2

    def mask_mod(_batch, _head, query_index, key_index):
        key_global = key_index < video_start
        key_video = (key_index >= video_start) & (key_index < video_stop)
        query_frame = query_index // frame_rows
        key_frame = (key_index - video_start) // frame_rows
        nearby_video = key_video & ((query_frame - key_frame).abs() <= radius)
        return key_global | nearby_video

    if BlockMask is None:
        raise RuntimeError(f"FlexAttention is unavailable: {_FLEX_IMPORT_ERROR}")

    partial_rows, full_rows = _video_block_rows(runtime, config.window_frames)
    # PyTorch's generated FlexAttention kernel currently addresses both index
    # tensors with kv_indices' row stride. Give partial and full metadata the
    # same padded width so every query row starts at the address the kernel uses.
    metadata_width = max(
        1,
        max((len(row) for row in partial_rows), default=0),
        max((len(row) for row in full_rows), default=0),
    )
    kv_num_blocks, kv_indices = _block_tensors(partial_rows, device, metadata_width)
    full_kv_num_blocks, full_kv_indices = _block_tensors(
        full_rows, device, metadata_width
    )
    block_mask = BlockMask.from_kv_blocks(
        kv_num_blocks,
        kv_indices,
        full_kv_num_blocks=full_kv_num_blocks,
        full_kv_indices=full_kv_indices,
        BLOCK_SIZE=_MASK_BLOCK_SIZE,
        mask_mod=mask_mod,
        seq_lengths=(video_rows, video_stop),
        # H3 inference does not use attention backward. PyTorch's automatic
        # transpose currently assumes square block grids and can index past the
        # shorter video-query grid when KV still covers the global prefix.
        compute_q_blocks=False,
    )
    _MASK_CACHE[key] = block_mask
    _MASK_CACHE.move_to_end(key)
    while len(_MASK_CACHE) > _MAX_MASK_CACHE:
        _MASK_CACHE.popitem(last=False)
    return block_mask


def _make_attention_override(config: WindowConfig, previous: Callable[..., Any] | None):
    """Use dense global queries plus local FlexAttention for eligible H3 layers."""
    def override(func, q, k, v, heads, mask=None, attn_precision=None,
                 skip_reshape=False, skip_output_reshape=False, **kwargs):
        def dense():
            if previous is None:
                return func(
                    q, k, v, heads,
                    mask=mask,
                    attn_precision=attn_precision,
                    skip_reshape=skip_reshape,
                    skip_output_reshape=skip_output_reshape,
                    **kwargs,
                )
            return previous(
                func, q, k, v, heads,
                mask=mask,
                attn_precision=attn_precision,
                skip_reshape=skip_reshape,
                skip_output_reshape=skip_output_reshape,
                **kwargs,
            )

        def configured_dense(query, key, value):
            # ``func`` is ComfyUI's configured backend (SageAttention, PyTorch,
            # xFormers, etc.) beneath the model-specific override chain. Global
            # rows must stay fully dense, so do not pass them through an earlier
            # sparse model override.
            return func(
                query, key, value, heads,
                mask=None,
                attn_precision=attn_precision,
                skip_reshape=True,
                skip_output_reshape=True,
                **kwargs,
            )

        options = kwargs.get("transformer_options")
        runtime = options.get(_RUNTIME_KEY) if isinstance(options, dict) else None
        if runtime is None or runtime.disabled or mask is not None or not skip_reshape:
            return dense()
        if q.ndim != 4 or k.shape != q.shape or v.shape != q.shape:
            return dense()
        if q.shape[0] != 1 or q.shape[1] != heads or q.shape[2] != runtime.video_stop:
            return dense()
        if q.device.type != "cuda" or q.dtype not in (torch.float16, torch.bfloat16):
            return dense()
        if q.shape[-1] != 128:
            return dense()

        layer = runtime.layer
        runtime.layer += 1
        if layer >= config.transformer_layers or layer in config.dense_layers:
            return dense()

        try:
            block_mask = _video_mask_for(runtime, config, q.device)
            if _compiled_flex_attention is None:
                raise RuntimeError(f"FlexAttention is unavailable: {_FLEX_IMPORT_ERROR}")
            # H3's QKV projection is split without copying, leaving the token stride at
            # 3 * hidden_size (21,504 elements). Triton's 32-bit offset arithmetic wraps
            # on long clips once token_index * stride crosses 2^31, corrupting late video
            # frames and every global row that reads those keys. Compact BHSD inputs keep
            # all offsets in range for H3's supported clip lengths.
            global_q = q[:, :, :runtime.video_start, :].contiguous()
            flex_q = q[:, :, runtime.video_start:runtime.video_stop, :].contiguous()
            flex_k = k.contiguous()
            flex_v = v.contiguous()
            if config.verbose:
                _log_once(
                    ("input_layout", tuple(q.stride()), tuple(flex_q.stride())),
                    f"compacted video Q and full KV for safe Triton indexing: token stride "
                    f"{q.stride(-2)} -> {flex_q.stride(-2)}",
                )
            global_output = configured_dense(global_q, flex_k, flex_v)
            video_output = _compiled_flex_attention(
                flex_q,
                flex_k,
                flex_v,
                block_mask=block_mask,
                scale=kwargs.get("scale"),
                kernel_options={
                    "BACKEND": "TRITON",
                    "ROWS_GUARANTEED_SAFE": True,
                    "BLOCKS_ARE_CONTIGUOUS": False,
                },
            )
            output = torch.cat((global_output, video_output), dim=2)
        except Exception as exc:
            runtime.disabled = True
            _log_once(
                ("kernel_error", type(exc).__name__, str(exc)),
                f"FlexAttention failed ({type(exc).__name__}: {exc}); using dense attention",
                logging.ERROR,
            )
            return dense()

        if skip_output_reshape:
            return output
        batch, _, tokens, head_dim = output.shape
        return output.transpose(1, 2).reshape(batch, tokens, heads * head_dim)

    return override


class H3WindowAttentionPatch(io.ComfyNode):
    """Apply the T2VA/FL2VA-only MiniMax H3 local-window attention prototype."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="H3WindowAttentionPatch",
            display_name="MiniMax H3 Window Attention (Prototype)",
            category="SwarmUI/MiniMax H3",
            is_experimental=True,
            description=(
                "Keep prompt, audio, and first/last keyframes global while video tokens "
                "attend a centered temporal window. T2VA/FL2VA only; reference-video "
                "layouts and unsupported kernels fall back to dense attention."
            ),
            inputs=[
                io.Model.Input("model"),
                io.Float.Input(
                    "window_seconds", default=5.0, min=1.0, max=20.0, step=0.5,
                    tooltip="Total centered video window. Five seconds is about 18 latent "
                            "frames on each side of the current frame.",
                ),
                io.String.Input(
                    "dense_layers", default="0,9,19,29,39,49",
                    tooltip="Comma-separated transformer layers that keep full global attention.",
                ),
                io.Boolean.Input(
                    "verbose", default=False,
                    tooltip="Log the resolved H3 layout and window configuration once per shape.",
                ),
            ],
            outputs=[io.Model.Output()],
        )

    @classmethod
    def execute(cls, model, window_seconds: float, dense_layers: str, verbose: bool) -> io.NodeOutput:
        patched = model.clone()
        diffusion_model = patched.get_model_object("diffusion_model")
        blocks = getattr(diffusion_model, "blocks", None)
        if type(diffusion_model).__name__ != "MiniMaxH3Model" or blocks is None:
            raise ValueError(
                "MiniMax H3 Window Attention requires ComfyUI's native MiniMaxH3Model"
            )
        if _FLEX_IMPORT_ERROR is not None:
            raise RuntimeError(f"PyTorch FlexAttention is unavailable: {_FLEX_IMPORT_ERROR}")

        count = len(blocks)
        config = WindowConfig(
            window_seconds=float(window_seconds),
            window_frames=_window_frames(float(window_seconds)),
            dense_layers=_parse_dense_layers(str(dense_layers), count),
            transformer_layers=count,
            verbose=bool(verbose),
        )
        options = patched.model_options.setdefault("transformer_options", {})
        previous = options.get(_OVERRIDE_KEY)
        options[_OVERRIDE_KEY] = _make_attention_override(config, previous)
        patched.add_wrapper_with_key(
            comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL,
            "h3_window_attention_layout",
            _make_forward_wrapper(config),
        )
        return io.NodeOutput(patched)
