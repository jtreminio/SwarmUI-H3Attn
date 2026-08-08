# SwarmUI-H3Attn

Adds an extension-owned MiniMax H3 window-attention prototype and wires two optional third-party accelerator nodes into SwarmUI's generated workflows:

- **H3 Window Attention** — T2VA/FL2VA-only compiled block-sparse attention. Prompt, generated audio, and first/last keyframes stay global; video tokens use a centered temporal window, with configurable dense transformer layers for long-range communication.

- [ComfyUI-SolAttn_triton](https://github.com/kijai/ComfyUI-SolAttn_triton) by kijai — sparse self-attention (Sol-Attn, arXiv 2607.24027) on a Triton kernel.
- [ComfyUI-Spectrum-MiniMax-H3](https://github.com/xmarre/ComfyUI-Spectrum-MiniMax-H3) by xmarre — Chebyshev-ridge spectral forecasting of post-transformer features, letting some solver steps skip the transformer entirely.

Both target the same problem: H3 packs ~37k tokens for a 5s clip at 832x1216 and ~142k for 20s, and its attention is dense `O(N²)`, so 3.8x the tokens costs 14x the attention time.

## What it does

It runs as the last workflow-generation step (priority 1000, after the last core step at 200), finds **every** `MiniMaxH3SigmaShift` node in the workflow, and splices a chain in behind it:

```
MiniMaxH3SigmaShift -> SolAttnPatch -> [SolAttnBlockProbe] -> H3WindowAttentionPatch -> SpectrumApplyMiniMaxH3 -> (consumer)
```

Every link is optional. The local window patch is inserted after Sol-Attn when both are enabled, so it owns eligible H3 attention calls and chains unsupported calls to the prior backend. The probe only appears when Sol-Attn is on, since it exists to measure Sol-Attn's own override.

## UI

One master group, **H3 Attention**, with a toggle. Inside it are independently toggled accelerator groups:

- **H3 Window Attention** — total centered window in seconds, dense transformer layers, and verbose layout diagnostics. It falls back to dense attention for clips no longer than the window and for reference-video layouts.
- **H3 Sol-Attn** — tau, start/end percent, min tokens, INT8 QK + PV, sink conditioning, Morton reorder + curve, dense blocks, tau profile, block probe, TMA, verbose.
- **H3 Spectrum** — blend weight, degree, ridge lambda, window size, flex window, warmup steps, bootstrap first forecast, tail actual steps, max history, history storage, debug.

A toggled group sends none of its parameters when its toggle (or the master's) is off, which is exactly how the extension decides whether to insert each node — no separate "enable" checkbox.

Parameters are feature-flagged (`h3_window_attention`, `sol_attn_triton`, `spectrum_minimax_h3`), so they only appear once the matching custom nodes are available. Window Attention ships with this extension; the third-party groups show an install button until installed.

**Optimal @ 20 Steps** sets every Sol-Attn and Spectrum value to a preset tuned for 20 sampling steps (it does not touch your Steps parameter). Three values are left alone on purpose: Spectrum's History Storage, and Sol-Attn's Dense Blocks and Tau Profile — how sensitive each block is to sparsification is per-model, and only a Block Probe run tells you.

**Dense Blocks / Tau Profile / Block Probe.** Block Probe computes every attention call both sparse and dense, logs each block's relative error worst-first when sampling ends, and outputs the dense result — so that generation is a dense reference and costs roughly dense + sparse. Paste the worst blocks into Dense Blocks (`0-2,-1` style, negatives count from the end) and turn the probe back off. Tau Profile is the graded version of the same knob: `0-30=2.0; 39-42=0.9` sparsifies the insensitive blocks harder and eases off the fragile ones, instead of one tau serving both.

**Bootstrap First Forecast.** Spectrum's one-point bootstrap holds step 0's transformer feature as step 1's prediction, so forecasting starts immediately rather than after a warmup long enough to fit a polynomial. Step 1 still runs its own timestep conditioning, final layer, projections and sigma-dependent audio processing — only the transformer body is skipped. It is defined only for Degree 1 and Warmup Steps ≤ 1; set anything else and the node keeps your degree and warmup and silently drops just the bootstrap, with a console line saying so. Upstream calls it experimental.

## Turbo LoRA converter

`tools/convert_h3_turbo_lora.py` rewrites [larryvrh/MiniMax-H3-Turbo-Lora](https://huggingface.co/larryvrh/MiniMax-H3-Turbo-Lora) (4-step audio-video, "no comfyui support yet") into a LoRA ComfyUI loads with its stock `LoraLoaderModelOnly` — no custom node. Two fixes:

- Its keys are ComfyUI's own `MiniMaxH3Model` submodule paths but carry no prefix, so ComfyUI's generic key map (built from the prefixed unet state dict) never matches them.
- Its adaLN deltas are 2688-wide. The pruned ("curve-form") Comfy-Org checkpoints store `adaln_proj.linear.weight` as `[*, 8]`, replacing the time embedding with `adaln_t_table` and a shared affine map `silu(t_emb(t)) ≈ basis @ curve(t) + offset`. `--curve` puts the LoRA through that same map: `lora_A @ basis` for the weight, plus a `.diff_b` bias patch for `lora_B @ (lora_A @ offset)`.

`tools/minimax_h3_adaln_curve.npz` holds that basis/offset, recovered by least squares from the released weights (blocks.0, 8192 rows) and cross-checked to ~5e-4 — the F16 storage floor — on held-out rows, other blocks and `final_layer`.

```bash
python3 tools/convert_h3_turbo_lora.py --curve minimax_h3_turbo_4step.safetensors mmh3_turbo_4step_pruned.safetensors
```

Drop `--curve` for a non-pruned base (`*_bf16`, `*_int8_convrot`). The outputs are not interchangeable — the adaLN shapes differ. Use it at 4 steps, CFG 1.

## Notes

- Window Attention requires PyTorch FlexAttention. Its first generation at a new shape builds compact block metadata and compiles a specialized Triton attention kernel. Interior allowed blocks are marked fully valid, so only temporal edges and packed-layout boundaries evaluate the token mask. Partial and full index rows intentionally share one padded width because the current generated kernel uses a shared row stride for both. Any unsupported layout or kernel failure is logged and falls back to the existing dense attention backend.
- On windowed layers, the configured ComfyUI backend (SageAttention when enabled) computes prompt, audio, and keyframe query rows against the full sequence; FlexAttention computes only target-video query rows.
- H3's QKV tensors use a very large inherited token stride. The patch compacts video Q and full K/V before FlexAttention so Triton's address calculation remains valid beyond the midpoint of a 20-second clip. This adds two full-sequence copies and one video-only copy per windowed layer.
- Window Attention is intentionally limited to T2VA/FL2VA. Reference-video layouts remain dense until they have a timestamp-aware policy.
- Sol-Attn needs Triton and is bf16 + head_dim 128 only; anything else falls back to the normal attention backend. The first run is slow while it autotunes.
- Spectrum is an approximate accelerator. It changes the denoising trajectory, so output differs from a native run even at identical seed. Its README documents trajectory deviations and degraded fine detail during fast motion.
- Spectrum requires ComfyUI at or after commit `e377e263` (Aug 3 2026) for the `latent_shapes` sampler API.
- Bootstrap First Forecast needs Spectrum v0.1.7+ and Tau Profile needs Sol-Attn at or after `0e334dc` (Aug 6 2026). On older nodes ComfyUI ignores the unknown input rather than erroring, so the rest of the workflow still runs — you just do not get the feature.
