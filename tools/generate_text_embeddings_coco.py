"""Generate CLIP text embeddings for OV-COCO 65 classes.

Uses the same ConvNext-L CLIP backbone as the detector to produce text embeddings
that are compatible with the visual embedding space used during training.

Output: dataset/metadata/ovdcoco_text_desc_convnextl.npy  — shape [65, 768], L2-normed

Usage (run from project root on remote server):
    python tools/generate_text_embeddings_coco.py

Requirements:
    open_clip_torch  (pip install open-clip-torch)
"""

import argparse
import json
import os

import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--class-file",
        default="tools/dataset/metadata/ovcoco_all_classes.json",
        help="JSON file with list of 65 OV-COCO class names",
    )
    parser.add_argument(
        "--output",
        default="dataset/metadata/ovdcoco_text_desc_convnextl.npy",
        help="Output NPY path",
    )
    parser.add_argument(
        "--model",
        default="convnext_large_d",
        help="OpenCLIP model name (must match detector backbone)",
    )
    parser.add_argument(
        "--pretrained",
        default="laion2b_s26b_b102k_augreg",
        help="OpenCLIP pretrained weights tag",
    )
    args = parser.parse_args()

    try:
        import open_clip
    except ImportError:
        raise ImportError("open_clip_torch not found. Run: pip install open-clip-torch")

    all_classes = json.load(open(args.class_file))
    print(f"Generating text embeddings for {len(all_classes)} classes: {all_classes[:5]} ...")

    model, _, _ = open_clip.create_model_and_transforms(
        args.model, pretrained=args.pretrained
    )
    tokenizer = open_clip.get_tokenizer(args.model)
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    print(f"Loaded {args.model} ({args.pretrained}) on {device}")

    # Prompt ensemble — same templates used by CLIP paper and OV-DQUO
    templates = [
        "a photo of a {}",
        "a photo of the {}",
        "a {} in the scene",
        "there is a {} in the scene",
        "a photo of a small {}",
        "a photo of a large {}",
    ]

    with torch.no_grad():
        text_feats = []
        for cls in all_classes:
            feats = []
            for tmpl in templates:
                text = tokenizer(tmpl.format(cls)).to(device)
                feat = model.encode_text(text)  # [1, feat_dim]
                feat = feat / feat.norm(dim=-1, keepdim=True)
                feats.append(feat)
            cls_feat = torch.stack(feats, dim=0).mean(dim=0)  # [1, feat_dim]
            cls_feat = cls_feat / cls_feat.norm(dim=-1, keepdim=True)
            text_feats.append(cls_feat)
        text_feats = torch.cat(text_feats, dim=0)  # [num_classes, feat_dim]

    print(f"Text embeddings shape: {text_feats.shape}")
    print(f"Norm check (should be ~1.0): {text_feats.norm(dim=-1).mean().item():.4f}")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    np.save(args.output, text_feats.cpu().float().numpy())
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
