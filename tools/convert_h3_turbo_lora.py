#!/usr/bin/env python3
"""Convert larryvrh/MiniMax-H3-Turbo-Lora into a LoRA ComfyUI can load.

Two things are wrong with the upstream file:

1. Its keys have no prefix. They are already ComfyUI's own MiniMaxH3Model submodule
   paths, but ComfyUI's generic LoRA key map is built from the prefixed unet state
   dict, so only 'diffusion_model.<path>' matches.

2. The adaLN projections don't fit the pruned ("curve-form") checkpoints. Comfy-Org's
   pruned releases replace the 2688-wide time embedding with an 8-wide curve, storing
   a shared affine map -- silu(t_emb(t)) ~= basis @ curve(t) + offset -- so their
   adaln_proj.linear.weight is [*, 8] while the LoRA's is [*, 2688]. Applying the same
   map to the LoRA lands its adaLN delta in the same space:

       dW_curve = lora_B @ (lora_A @ basis)
       db       = lora_B @ (lora_A @ offset)     # emitted as a .diff_b patch

   basis/offset were recovered by least squares from the released weights themselves
   (blocks.0, 8192 rows) and cross-check to ~5e-4 on held-out rows, other blocks and
   final_layer -- i.e. to the F16 storage floor. See minimax_h3_adaln_curve.npz.

Pass --curve for a pruned base model, omit it for a full one (*_bf16, *_int8_convrot
without "pruned"). The two are not interchangeable: adaLN shapes differ.

    python3 convert_h3_turbo_lora.py --curve minimax_h3_turbo_4step.safetensors out.safetensors
"""
import argparse
import json
import os
import struct

import numpy as np

PREFIX = "diffusion_model."
ADALN_SUFFIX = "adaln_proj.linear"
DEFAULT_CURVE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "minimax_h3_adaln_curve.npz")


def read_safetensors(path):
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        hdr = json.loads(f.read(n))
        blob = f.read()
    meta = hdr.pop("__metadata__", None)
    out = {}
    for k, m in hdr.items():
        a, b = m["data_offsets"]
        raw = blob[a:b]
        if m["dtype"] == "BF16":
            arr = (np.frombuffer(raw, "<u2").astype(np.uint32) << 16).view(np.float32)
        elif m["dtype"] == "F16":
            arr = np.frombuffer(raw, "<f2").astype(np.float32)
        elif m["dtype"] == "F32":
            arr = np.frombuffer(raw, "<f4")
        else:
            raise SystemExit(f"unhandled dtype {m['dtype']} on {k}")
        out[k] = arr.reshape(m["shape"])
    return out, meta


def to_bf16(arr):
    u = arr.astype(np.float32).view(np.uint32).astype(np.uint64)
    u = (u + 0x7FFF + ((u >> 16) & 1)) >> 16  # round to nearest even, then truncate
    return u.astype("<u2").tobytes()


def write_safetensors(path, tensors, meta):
    hdr, blobs, off = {}, [], 0
    for k, arr in tensors.items():
        raw = to_bf16(arr)
        hdr[k] = {"dtype": "BF16", "shape": list(arr.shape), "data_offsets": [off, off + len(raw)]}
        blobs.append(raw)
        off += len(raw)
    if meta:
        hdr["__metadata__"] = meta
    head = json.dumps(hdr, separators=(",", ":")).encode()
    head += b" " * (-len(head) % 8)  # keep the data block 8-byte aligned
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(head)))
        f.write(head)
        for raw in blobs:
            f.write(raw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--curve", action="store_true", help="target a pruned/curve-form base model")
    ap.add_argument("--curve-data", default=DEFAULT_CURVE)
    args = ap.parse_args()

    tensors, meta = read_safetensors(args.src)
    basis = offset = None
    if args.curve:
        data = np.load(args.curve_data)
        basis, offset = data["basis"].astype(np.float64), data["offset"].astype(np.float64)

    out, projected = {}, 0
    for k, arr in tensors.items():
        if not k.endswith((".lora_A.weight", ".lora_B.weight")):
            raise SystemExit(f"unexpected key (not a plain lora_A/B pair): {k}")
        name = k.rsplit(".lora_", 1)[0]
        if basis is not None and name.endswith(ADALN_SUFFIX) and k.endswith(".lora_A.weight"):
            if arr.shape[1] != basis.shape[0]:
                raise SystemExit(f"{k}: expected [rank, {basis.shape[0]}], got {list(arr.shape)}")
            a64 = arr.astype(np.float64)
            out[f"{PREFIX}{name}.diff_b"] = (tensors[f"{name}.lora_B.weight"].astype(np.float64) @ (a64 @ offset)).astype(np.float32)
            arr = (a64 @ basis).astype(np.float32)
            projected += 1
        out[PREFIX + k] = arr

    meta = dict(meta or {})
    meta["adaln_form"] = "curve" if args.curve else "full"
    write_safetensors(args.dst, out, meta)
    print(f"{len(out)} tensors written ({projected} adaLN projected onto the curve basis) -> {args.dst}")


if __name__ == "__main__":
    main()
