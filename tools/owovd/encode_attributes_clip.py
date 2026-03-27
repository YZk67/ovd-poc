#!/usr/bin/env python3
"""Encode visual attributes into CLIP embeddings.

Takes the attribute JSON from generate_attributes.py and encodes each
attribute into a CLIP ConvNeXt-L text embedding.

Output: .pth file with {'att_text': [...], 'att_embedding': Tensor[N, 768]}
Compatible with OW-OVD's att_embeddings format.

Usage:
    python tools/owovd/encode_attributes_clip.py \
        --attr-json tools/owovd/coco_attributes.json \
        --output dataset/metadata/coco_att_embeddings.pth
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F


def main():
    parser = argparse.ArgumentParser(description="Encode attributes with CLIP")
    parser.add_argument("--attr-json", required=True, help="Attribute JSON from generate_attributes.py")
    parser.add_argument("--output", default="dataset/metadata/coco_att_embeddings.pth", help="Output .pth path")
    parser.add_argument("--model", default="convnext_large_d", help="OpenCLIP model name")
    parser.add_argument("--pretrained", default="laion2b_s26b_b102k_augreg", help="OpenCLIP pretrained tag")
    args = parser.parse_args()

    try:
        import open_clip
    except ImportError:
        raise ImportError("open_clip_torch not found. Run: pip install open-clip-torch")

    # Load attributes
    data = json.load(open(args.attr_json))
    all_attrs = data["all_attributes"]
    print(f"Encoding {len(all_attrs)} attributes...")

    # Load CLIP model
    model, _, _ = open_clip.create_model_and_transforms(args.model, pretrained=args.pretrained)
    tokenizer = open_clip.get_tokenizer(args.model)
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    print(f"Loaded {args.model} ({args.pretrained}) on {device}")

    # Encode each attribute with prompt templates
    templates = [
        "a photo of something that {}",
        "an object that {}",
        "something that {}",
    ]

    with torch.no_grad():
        embeddings = []
        for attr in all_attrs:
            feats = []
            for tmpl in templates:
                text = tokenizer(tmpl.format(attr)).to(device)
                feat = model.encode_text(text)
                feat = F.normalize(feat, p=2, dim=-1)
                feats.append(feat)
            attr_feat = torch.stack(feats, dim=0).mean(dim=0)
            attr_feat = F.normalize(attr_feat, p=2, dim=-1)
            embeddings.append(attr_feat.squeeze(0))

        embeddings = torch.stack(embeddings, dim=0)  # [N, 768]

    print(f"Attribute embeddings shape: {embeddings.shape}")
    print(f"Norm check: {embeddings.norm(dim=-1).mean().item():.4f}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"att_text": all_attrs, "att_embedding": embeddings.cpu()}, out_path)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
