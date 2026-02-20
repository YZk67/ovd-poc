#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Paper-style clustering:
  (1) Visual descriptions per class (from GPT-3.5 or other LLM)  -> desc_json
  (2) T5 encoder embeddings for descriptions
  (3) KMeans++ clustering

Output:
  ovdcoco_cluster_K.npy  (shape [num_classes], int32)

You must provide:
  --cat_info_json  (ovd_ins_train2017_all_cat_info.json; defines class order)
  --desc_json      JSON mapping class name (or index) -> description text

desc_json formats supported:
  A) {"person": "...", "bicycle": "...", ...}
  B) {"0": "...", "1": "...", ...}  # index aligned with cat_info order
"""

import argparse
import json
import os
from typing import Dict, List

import numpy as np
from sklearn.cluster import KMeans


def parse_classes(cat_info_json: str) -> List[str]:
    with open(cat_info_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Case 1: list of {"id": int, "name": str, ...}
    if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
        if "id" in data[0] and "name" in data[0]:
            data_sorted = sorted(data, key=lambda x: int(x["id"]))
            classes = [d["name"] for d in data_sorted]
            # sanity check (optional)
            ids = [int(d["id"]) for d in data_sorted]
            if ids != list(range(len(ids))):
                print("[WARN] category ids are not consecutive 0..N-1:", ids[:10], "...", ids[-10:])
            return classes

    # Case 2: COCO-style dict with "categories"
    if isinstance(data, dict) and "categories" in data:
        cats = data["categories"]
        if isinstance(cats, list) and len(cats) > 0 and isinstance(cats[0], dict) and "name" in cats[0]:
            if "id" in cats[0]:
                cats = sorted(cats, key=lambda x: int(x["id"]))
            return [c["name"] for c in cats]

    # Case 3: detectron2 metadata-ish
    if isinstance(data, dict) and "thing_classes" in data and isinstance(data["thing_classes"], list):
        return data["thing_classes"]

    # Case 4: simple {"classes":[...]}
    if isinstance(data, dict) and "classes" in data and isinstance(data["classes"], list):
        return data["classes"]

    raise ValueError("Cannot infer class list from cat_info_json. Please inspect the json schema.")


def load_desc(desc_json: str) -> Dict[str, str]:
    with open(desc_json, "r", encoding="utf-8") as f:
        d = json.load(f)
    out: Dict[str, str] = {}
    for k, v in d.items():
        if isinstance(v, str) and v.strip():
            out[str(k)] = v.strip()
    return out


def t5_embed_texts(
    texts: List[str],
    model_name: str,
    device: str,
    max_length: int = 128,
    batch_size: int = 32,
) -> np.ndarray:
    """
    T5 encoder embeddings (encoder-only):
      - tokenize (with explicit max_length)
      - run T5EncoderModel
      - mean-pool over tokens (masked)
      - L2 normalize
    """
    import torch
    from transformers import AutoTokenizer, T5EncoderModel

    tok = AutoTokenizer.from_pretrained(model_name)
    model = T5EncoderModel.from_pretrained(model_name).to(device).eval()

    feats = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            inp = tok(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(device)

            out = model(**inp)
            h = out.last_hidden_state  # [B,T,H]

            attn = inp["attention_mask"].unsqueeze(-1).type_as(h)  # [B,T,1]
            h = h * attn
            denom = attn.sum(dim=1).clamp(min=1)
            sent = h.sum(dim=1) / denom  # [B,H]

            sent = sent / (sent.norm(dim=-1, keepdim=True) + 1e-12)
            feats.append(sent.detach().cpu().float().numpy())

    return np.concatenate(feats, axis=0).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cat_info_json", type=str, required=True)
    ap.add_argument("--desc_json", type=str, required=True)
    ap.add_argument("--out_npy", type=str, required=True)
    ap.add_argument("--k", type=int, default=64)  # <= num_classes
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--t5_model", type=str, default="t5-base")
    ap.add_argument("--max_length", type=int, default=128)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--one_based", action="store_true")
    args = ap.parse_args()

    classes = parse_classes(args.cat_info_json)
    n = len(classes)
    if args.k > n:
        raise ValueError(f"k must be <= #classes. k={args.k}, n={n}")

    desc_map = load_desc(args.desc_json)

    # Resolve description per class in cat_info order:
    # prefer index-aligned desc ("0","1",...) then fallback to class name key.
    descs = []
    missing = []
    for i, name in enumerate(classes):
        d = desc_map.get(str(i)) or desc_map.get(name)
        if not d:
            missing.append((i, name))
            d = f"{name}"  # fallback
        descs.append(d)

    if missing:
        print(f"[WARN] Missing {len(missing)} descriptions. Example:", missing[:5])

    emb = t5_embed_texts(
        descs,
        model_name=args.t5_model,
        device=args.device,
        max_length=args.max_length,
        batch_size=args.batch_size,
    )

    km = KMeans(
        n_clusters=args.k,
        init="k-means++",
        n_init=50,
        max_iter=300,
        random_state=args.seed,
    )
    labels = km.fit_predict(emb).astype(np.int32)
    if args.one_based:
        labels += 1

    os.makedirs(os.path.dirname(args.out_npy) or ".", exist_ok=True)
    np.save(args.out_npy, labels)

    uniq, cnt = np.unique(labels, return_counts=True)
    print(f"[OK] saved {args.out_npy}")
    print(f"n_classes={n}, k={args.k}, unique_clusters={len(uniq)}")
    print(f"min={int(cnt.min())}, max={int(cnt.max())}, mean={cnt.mean():.2f}")


if __name__ == "__main__":
    main()
