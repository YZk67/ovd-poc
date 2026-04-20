#!/usr/bin/env python3
"""Inspect TPA prototype_queries in a trained LaMI-DETR checkpoint.

Loads a .pth, pulls out every `class_embed.{i}.tpa.prototype_queries`
(and same under `transformer.decoder.class_embed.{i}.tpa.prototype_queries`),
reports per-TPA:
  * norms of the K queries (are they collapsed to 0?)
  * pairwise cosine similarity between the K queries (are they parallel?)

If cos-sim among queries is ~1, queries are identical → softmax picks same
prompt for every slot → all K prototypes are the same prompt's value →
orth_off_mse=1. That's the "TPA collapsed" signature.

Usage:
  python tools/diagnose_tpa_prototypes.py --ckpt /path/to/model.pth
"""

import argparse
import torch
import torch.nn.functional as F


def inspect_prototype_queries(name: str, w: torch.Tensor) -> None:
    # w: [K, hidden_dim]
    K = w.shape[0]
    norms = w.norm(dim=-1)
    Pn = F.normalize(w, p=2, dim=-1)
    G = Pn @ Pn.t()  # [K, K] cosine similarities
    off = G - torch.eye(K, device=G.device)
    off_abs_mean = off.abs().sum().item() / (K * K - K)
    off_mean = off.sum().item() / (K * K - K)
    print(f"\n{name}")
    print(f"  shape={tuple(w.shape)}  norms={[f'{x:.3f}' for x in norms.tolist()]}")
    print(f"  pairwise |cos|_mean(off-diag) = {off_abs_mean:.4f}")
    print(f"  pairwise  cos _mean(off-diag) = {off_mean:.4f}")
    print(f"  cosine matrix:")
    for row in G.tolist():
        print(f"    [{', '.join(f'{v:+.3f}' for v in row)}]")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    args = p.parse_args()

    sd = torch.load(args.ckpt, map_location="cpu")
    if "model" in sd:
        sd = sd["model"]

    keys = sorted(k for k in sd if k.endswith("tpa.prototype_queries"))
    if not keys:
        print("No TPA prototype_queries found in checkpoint.")
        return

    print(f"Found {len(keys)} TPA prototype_queries tensors")
    for k in keys:
        inspect_prototype_queries(k, sd[k])


if __name__ == "__main__":
    main()
