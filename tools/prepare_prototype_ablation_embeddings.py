#!/usr/bin/env python3
"""
Prepare text embedding files for prototype ablation runs.

This script does not encode text. It transforms existing CLIP/OpenCLIP embedding
banks into the shapes expected by the ablation configs:

  - class-name only:          [C, D]
  - CLIP template average:    [C, D]
  - LLM prompt average:       [C, D]
  - optional random K prompts [C, K, D]

Example:
    python tools/prepare_prototype_ablation_embeddings.py \
        --llm-prompt-embedding dataset/metadata/lvis_claude_prompts_convnextl.npy \
        --class-name-embedding dataset/metadata/lvis_class_name_convnextl.npy \
        --clip-template-embedding dataset/metadata/lvis_clip_templates_convnextl.npy \
        --output-dir dataset/metadata/ablation \
        --prefix lvis
"""

import argparse
from pathlib import Path
from typing import Optional

import numpy as np


def l2_normalize(x: np.ndarray, axis: int = -1, eps: float = 1e-12) -> np.ndarray:
    norm = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / np.maximum(norm, eps)


def load_embedding(path: Optional[Path], label: str) -> Optional[np.ndarray]:
    if path is None:
        print(f"[skip] {label}: no input path")
        return None
    if not path.exists():
        print(f"[skip] {label}: {path} does not exist")
        return None
    arr = np.load(path)
    if arr.ndim not in (2, 3):
        raise ValueError(f"{label} must have shape [C,D] or [C,N,D], got {arr.shape}")
    print(f"[load] {label}: {path} shape={arr.shape} dtype={arr.dtype}")
    return arr.astype(np.float32, copy=False)


def to_single_prototype(arr: np.ndarray, method: str) -> np.ndarray:
    if arr.ndim == 2:
        return l2_normalize(arr.astype(np.float32, copy=False))
    if method == "mean":
        out = arr.mean(axis=1)
    elif method == "first":
        out = arr[:, 0]
    else:
        raise ValueError(f"Unknown single-prototype method: {method}")
    return l2_normalize(out.astype(np.float32, copy=False))


def random_k(arr: np.ndarray, k: int, seed: int) -> np.ndarray:
    if arr.ndim != 3:
        raise ValueError(f"random K requires [C,N,D] embeddings, got {arr.shape}")
    num_classes, num_prompts, _ = arr.shape
    if k > num_prompts:
        raise ValueError(f"k={k} is larger than the available prompts N={num_prompts}")
    rng = np.random.default_rng(seed)
    out = np.empty((num_classes, k, arr.shape[-1]), dtype=np.float32)
    for class_idx in range(num_classes):
        inds = rng.choice(num_prompts, size=k, replace=False)
        out[class_idx] = arr[class_idx, inds]
    return l2_normalize(out)


def save_embedding(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, arr.astype(np.float32, copy=False))
    loaded = np.load(path)
    print(f"[save] {path} shape={loaded.shape} dtype={loaded.dtype}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--class-name-embedding", type=Path, default=None)
    parser.add_argument("--clip-template-embedding", type=Path, default=None)
    parser.add_argument(
        "--llm-prompt-embedding",
        type=Path,
        default=Path("dataset/metadata/lvis_claude_prompts_convnextl.npy"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("dataset/metadata/ablation"))
    parser.add_argument("--prefix", type=str, default="lvis")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--include-random-k",
        action="store_true",
        help="Also write a static random-K prompt bank. Disabled for the first quick round.",
    )
    args = parser.parse_args()

    class_name = load_embedding(args.class_name_embedding, "class-name")
    clip_templates = load_embedding(args.clip_template_embedding, "CLIP templates")
    llm_prompts = load_embedding(args.llm_prompt_embedding, "LLM prompts")

    if class_name is not None:
        save_embedding(
            args.output_dir / f"{args.prefix}_class_name_only_convnextl.npy",
            to_single_prototype(class_name, method="first"),
        )

    if clip_templates is not None:
        save_embedding(
            args.output_dir / f"{args.prefix}_clip_templates_avg_convnextl.npy",
            to_single_prototype(clip_templates, method="mean"),
        )

    if llm_prompts is not None:
        save_embedding(
            args.output_dir / f"{args.prefix}_llm_prompts_avg_convnextl.npy",
            to_single_prototype(llm_prompts, method="mean"),
        )
        if args.include_random_k:
            save_embedding(
                args.output_dir / f"{args.prefix}_llm_random{args.k}_seed{args.seed}_convnextl.npy",
                random_k(llm_prompts, k=args.k, seed=args.seed),
            )


if __name__ == "__main__":
    main()
