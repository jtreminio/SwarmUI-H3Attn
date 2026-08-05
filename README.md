# SwarmUI-H3Attn

Wires two third-party MiniMax H3 accelerator nodes into SwarmUI's generated workflows:

- [ComfyUI-SolAttn_triton](https://github.com/kijai/ComfyUI-SolAttn_triton) by kijai — sparse self-attention (Sol-Attn, arXiv 2607.24027) on a Triton kernel.
- [ComfyUI-Spectrum-MiniMax-H3](https://github.com/xmarre/ComfyUI-Spectrum-MiniMax-H3) by xmarre — Chebyshev-ridge spectral forecasting of post-transformer features, letting some solver steps skip the transformer entirely.

Both target the same problem: H3 packs ~37k tokens for a 5s clip at 832x1216 and ~142k for 20s, and its attention is dense `O(N²)`, so 3.8x the tokens costs 14x the attention time.

## What it does

It runs as the last workflow-generation step (priority 1000, after the last core step at 200), finds **every** `MiniMaxH3SigmaShift` node in the workflow, and splices a chain in behind it:

```
MiniMaxH3SigmaShift -> SolAttnPatch -> SpectrumApplyMiniMaxH3 -> (whatever used to consume the sigma shift)
```

Either link is optional; whichever sub-groups are enabled get inserted, in that order.

## UI

One master group, **H3 Attention**, with a toggle. Inside it, one toggled sub-group per upstream extension:

- **H3 Sol-Attn** — tau, start/end percent, min tokens, INT8 QK, sink conditioning, Morton reorder + curve, TMA, verbose.
- **H3 Spectrum** — blend weight, degree, ridge lambda, window size, flex window, warmup steps, tail actual steps, max history, history storage, debug.

A toggled group sends none of its parameters when its toggle (or the master's) is off, which is exactly how the extension decides whether to insert each node — no separate "enable" checkbox.

Parameters are feature-flagged (`sol_attn_triton`, `spectrum_minimax_h3`), so they only appear once the matching custom nodes are installed. Until then each sub-group shows an install button.

## Notes

- Sol-Attn needs Triton and is bf16 + head_dim 128 only; anything else falls back to the normal attention backend. The first run is slow while it autotunes.
- Spectrum is an approximate accelerator. It changes the denoising trajectory, so output differs from a native run even at identical seed. Its README documents trajectory deviations and degraded fine detail during fast motion.
- Spectrum requires ComfyUI at or after commit `e377e263` (Aug 3 2026) for the `latent_shapes` sampler API.
