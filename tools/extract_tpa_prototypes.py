"""
One-shot extraction of TPA prototypes from a trained model.

Loads the model + checkpoint, runs TPA forward once on all 1203 LVIS prompts,
saves the resulting (1203, K=5, 768) prototype tensor to disk. Does NOT need
images; CPU is fine.

Run on the training server:
    python tools/extract_tpa_prototypes.py \
        --config lami_dino/configs/dino_convnext_large_4scale_12ep_lvis.py \
        --ckpt   /root/autodl-tmp/model_final.pth \
        --out    /root/autodl-tmp/tpa_prototypes.npy

Then scp the .npy to local for plotting.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config",
                   default="lami_dino/configs/dino_convnext_large_4scale_12ep_lvis.py")
    p.add_argument("--ckpt", default="/root/autodl-tmp/model_final.pth")
    p.add_argument("--out", default="/root/autodl-tmp/tpa_prototypes.npy")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(repo_root))

    from detectron2.checkpoint import DetectionCheckpointer
    from detectron2.config import LazyConfig, instantiate

    print(f"[load] config: {args.config}")
    cfg = LazyConfig.load(args.config)
    print(f"[load] instantiating model")
    model = instantiate(cfg.model)
    model.eval()
    model = model.to(args.device)
    print(f"[load] checkpoint: {args.ckpt}")
    DetectionCheckpointer(model).load(args.ckpt)

    last = model.transformer.decoder.class_embed[-1]
    if not getattr(last, "use_tpa", False):
        sys.exit("[!] model does not have TPA enabled")

    print(f"[run] TPA forward on full LVIS prompt embeddings...")
    text_feats = last._maybe_move_text_feats(training=False)  # [1203, 8, 768]
    print(f"      input shape: {tuple(text_feats.shape)}")

    with torch.no_grad():
        prototypes, _ = last.tpa(text_feats, with_loss=False)  # [1203, K, 768]
    prototypes = prototypes.detach().cpu().numpy()
    print(f"      output shape: {prototypes.shape}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, prototypes)
    print(f"[save] {out}  ({prototypes.nbytes / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
