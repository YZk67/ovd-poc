#!/usr/bin/env python
"""Fail-fast asset checks before an expensive InstructDet paper run."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Optional

import numpy as np


def _readable_file(path: Path, label: str, errors: list[str]) -> bool:
    try:
        is_file = path.is_file()
        size = path.stat().st_size if is_file else 0
    except OSError as exc:
        errors.append(f"{label}: cannot access {path}: {exc}")
        return False
    if not is_file:
        errors.append(f"{label}: missing or unreadable: {path}")
        return False
    if size <= 0:
        errors.append(f"{label}: empty file: {path}")
        return False
    return True


def _check_array(
    path: Path,
    label: str,
    errors: list[str],
    *,
    classes: int,
    ndim: tuple[int, ...],
    embedding_dim: int = 768,
    minimum_prompts: Optional[int] = None,
) -> None:
    if not _readable_file(path, label, errors):
        return
    try:
        array = np.load(path, mmap_mode="r")
    except Exception as exc:
        errors.append(f"{label}: cannot load {path}: {exc}")
        return
    if array.ndim not in ndim:
        errors.append(f"{label}: expected ndim in {ndim}, got shape {array.shape}")
        return
    if array.shape[0] != classes or array.shape[-1] != embedding_dim:
        errors.append(
            f"{label}: expected {classes} classes and {embedding_dim}D, got {array.shape}"
        )
    if minimum_prompts is not None and (
        array.ndim != 3 or array.shape[1] < minimum_prompts
    ):
        errors.append(
            f"{label}: expected at least {minimum_prompts} prompts per class, got {array.shape}"
        )
    if not np.isfinite(array).all():
        errors.append(f"{label}: contains NaN or Inf values")
    print(f"[OK] {label}: shape={array.shape}, dtype={array.dtype}, path={path}")


def _check_json_list(
    path: Path,
    label: str,
    errors: list[str],
    *,
    expected_length: int,
) -> None:
    if not _readable_file(path, label, errors):
        return
    try:
        value = json.loads(path.read_text())
    except Exception as exc:
        errors.append(f"{label}: cannot parse {path}: {exc}")
        return
    if not isinstance(value, list) or len(value) != expected_length:
        errors.append(
            f"{label}: expected a list of length {expected_length}, "
            f"got {type(value).__name__} length={len(value) if isinstance(value, list) else 'n/a'}"
        )
        return
    print(f"[OK] {label}: {len(value)} entries, path={path}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prompt-bank",
        default="dataset/metadata/lvis_claude_prompts_convnextl.npy",
    )
    parser.add_argument(
        "--vlm-bank",
        default="dataset/metadata/lvis_visual_desc_confuse_lvis_convnextl.npy",
    )
    parser.add_argument(
        "--all-classes",
        default="dataset/lvis/lvis_v1_all_classes.json",
    )
    parser.add_argument(
        "--seen-classes",
        default="dataset/lvis/lvis_v1_seen_classes.json",
    )
    parser.add_argument(
        "--clip-backbone",
        default="pretrained_models/clip_convnext_large_trans.pth",
    )
    parser.add_argument(
        "--clip-head",
        default="pretrained_models/clip_convnext_large_head.pth",
    )
    parser.add_argument("--hash", action="store_true", help="print SHA-256 provenance hashes")
    args = parser.parse_args()

    errors: list[str] = []
    prompt_bank = Path(args.prompt_bank)
    vlm_bank = Path(args.vlm_bank)
    all_classes = Path(args.all_classes)
    seen_classes = Path(args.seen_classes)
    clip_backbone = Path(args.clip_backbone)
    clip_head = Path(args.clip_head)

    _check_array(
        prompt_bank,
        "TPA prompt bank",
        errors,
        classes=1203,
        ndim=(3,),
        minimum_prompts=5,
    )
    _check_array(
        vlm_bank,
        "VLM score-ensemble bank",
        errors,
        classes=1203,
        ndim=(2, 3),
    )
    _check_json_list(all_classes, "LVIS all-class order", errors, expected_length=1203)
    _check_json_list(seen_classes, "LVIS seen-class order", errors, expected_length=866)
    _readable_file(clip_backbone, "CLIP ConvNeXt-L backbone", errors)
    _readable_file(clip_head, "CLIP ConvNeXt-L head", errors)

    if args.hash:
        for label, path in (
            ("prompt_bank", prompt_bank),
            ("vlm_bank", vlm_bank),
            ("clip_backbone", clip_backbone),
            ("clip_head", clip_head),
        ):
            try:
                if path.is_file():
                    print(f"[SHA256] {label}={_sha256(path)}")
            except OSError:
                # The access error was already reported by the checks above.
                pass

    if errors:
        for error in errors:
            print(f"[ERROR] {error}")
        raise SystemExit(f"preflight failed with {len(errors)} error(s)")
    print("Preflight passed: required LVIS/CLIP assets are readable and shape-compatible.")


if __name__ == "__main__":
    main()
