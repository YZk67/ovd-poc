#!/usr/bin/env python3
"""
Train a lightweight D3 crop-description verifier.

This is intentionally kept outside the detector. It answers one question first:
given a detector box crop and a target phrase, can a small verifier distinguish
matching region-description pairs from hard negatives?

Inputs are JSONL files produced by tools/build_d3_verifier_pairs.py.
The script can either encode OpenCLIP crop/text features into a cache or train
directly from precomputed cache files.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _jsonable_args(args: argparse.Namespace) -> Dict[str, Any]:
    values = {}
    for key, value in vars(args).items():
        values[key] = str(value) if isinstance(value, Path) else value
    return values


def _negative_kind(row: Mapping[str, Any]) -> str:
    value = row.get("negative_type")
    return str(value) if value is not None else "positive"


def _sample_count(base_count: int, ratio: float, available: int) -> int:
    if ratio < 0:
        return available
    return min(available, int(round(base_count * ratio)))


def _sample_total_for_positive_count(
    positive_count: int,
    *,
    same_phrase_neg_per_pos: float,
    wrong_phrase_neg_per_pos: float,
    same_available: int,
    wrong_available: int,
) -> int:
    same_count = _sample_count(positive_count, same_phrase_neg_per_pos, same_available)
    wrong_count = _sample_count(positive_count, wrong_phrase_neg_per_pos, wrong_available)
    return positive_count + same_count + wrong_count


def _fit_positive_count_to_budget(
    positive_limit: int,
    *,
    max_samples: Optional[int],
    same_phrase_neg_per_pos: float,
    wrong_phrase_neg_per_pos: float,
    same_available: int,
    wrong_available: int,
) -> int:
    if max_samples is None or max_samples <= 0:
        return positive_limit
    if positive_limit <= 0:
        return 0

    lo = 0
    hi = min(positive_limit, max_samples)
    best = 0
    while lo <= hi:
        mid = (lo + hi) // 2
        total = _sample_total_for_positive_count(
            mid,
            same_phrase_neg_per_pos=same_phrase_neg_per_pos,
            wrong_phrase_neg_per_pos=wrong_phrase_neg_per_pos,
            same_available=same_available,
            wrong_available=wrong_available,
        )
        if total <= max_samples:
            best = mid
            lo = mid + 1
        else:
            hi = mid - 1

    return best if best > 0 else min(positive_limit, max_samples)


def _sample_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    same_phrase_neg_per_pos: float,
    wrong_phrase_neg_per_pos: float,
    max_positives: Optional[int],
    max_samples: Optional[int],
    seed: int,
) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    positives = [dict(row) for row in rows if int(row.get("label", 0)) == 1]
    same_phrase = [
        dict(row)
        for row in rows
        if int(row.get("label", 0)) == 0 and _negative_kind(row) == "same_phrase_bad_box"
    ]
    wrong_phrase = [
        dict(row)
        for row in rows
        if int(row.get("label", 0)) == 0 and _negative_kind(row).startswith("wrong_phrase_same_region")
    ]

    rng.shuffle(positives)
    rng.shuffle(same_phrase)
    rng.shuffle(wrong_phrase)

    positive_limit = len(positives)
    if max_positives is not None and max_positives > 0:
        positive_limit = min(positive_limit, max_positives)
    positive_limit = _fit_positive_count_to_budget(
        positive_limit,
        max_samples=max_samples,
        same_phrase_neg_per_pos=same_phrase_neg_per_pos,
        wrong_phrase_neg_per_pos=wrong_phrase_neg_per_pos,
        same_available=len(same_phrase),
        wrong_available=len(wrong_phrase),
    )

    positives = positives[:positive_limit]
    same_count = _sample_count(len(positives), same_phrase_neg_per_pos, len(same_phrase))
    wrong_count = _sample_count(len(positives), wrong_phrase_neg_per_pos, len(wrong_phrase))
    selected = positives + same_phrase[:same_count] + wrong_phrase[:wrong_count]

    if max_samples is not None and max_samples > 0 and len(selected) > max_samples:
        pos = [row for row in selected if int(row.get("label", 0)) == 1]
        neg = [row for row in selected if int(row.get("label", 0)) == 0]
        if len(pos) > max_samples:
            selected = rng.sample(pos, max_samples)
        else:
            selected = pos + rng.sample(neg, max_samples - len(pos))

    rng.shuffle(selected)
    return selected


def _summarize_rows(name: str, rows: Sequence[Mapping[str, Any]]) -> Dict[str, int]:
    counts = Counter(_negative_kind(row) for row in rows)
    summary = {
        "total": len(rows),
        "positive": sum(1 for row in rows if int(row.get("label", 0)) == 1),
        "negative": sum(1 for row in rows if int(row.get("label", 0)) == 0),
    }
    summary.update({kind: int(count) for kind, count in sorted(counts.items())})
    print(f"{name} rows: {json.dumps(summary, sort_keys=True)}")
    return summary


def _resolve_image_path(image_root: Path, file_name: str) -> Path:
    file_path = Path(file_name)
    if file_path.is_absolute() and file_path.exists():
        return file_path

    candidates = [
        image_root / file_name,
        image_root / file_path.name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _expanded_xyxy(
    bbox_xywh: Sequence[float],
    *,
    width: int,
    height: int,
    margin: float,
) -> Optional[Tuple[int, int, int, int]]:
    x, y, w, h = [float(v) for v in bbox_xywh]
    if w <= 1 or h <= 1:
        return None

    pad_x = w * margin
    pad_y = h * margin
    x0 = max(0.0, x - pad_x)
    y0 = max(0.0, y - pad_y)
    x1 = min(float(width), x + w + pad_x)
    y1 = min(float(height), y + h + pad_y)
    if x1 <= x0 + 1 or y1 <= y0 + 1:
        return None
    return int(math.floor(x0)), int(math.floor(y0)), int(math.ceil(x1)), int(math.ceil(y1))


def _load_openclip(model_name: str, pretrained: str, device: str):
    try:
        import open_clip
    except ImportError as exc:
        raise ImportError("open_clip is required. Run this script in the lami env.") from exc

    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name,
        pretrained=pretrained,
        device=device,
    )
    model.eval()
    return model, preprocess, open_clip.tokenize


@torch.no_grad()
def _encode_texts(
    model,
    tokenizer,
    texts: Sequence[str],
    *,
    batch_size: int,
    device: str,
) -> torch.Tensor:
    features = []
    for start in tqdm(range(0, len(texts), batch_size), desc="encoding texts"):
        tokens = tokenizer(list(texts[start : start + batch_size])).to(device)
        feats = model.encode_text(tokens)
        feats = F.normalize(feats.float(), p=2, dim=-1)
        features.append(feats.cpu())
    return torch.cat(features, dim=0)


@torch.no_grad()
def _encode_crops(
    model,
    tensors: Sequence[torch.Tensor],
    *,
    device: str,
) -> torch.Tensor:
    batch = torch.stack(list(tensors), dim=0).to(device)
    feats = model.encode_image(batch)
    feats = F.normalize(feats.float(), p=2, dim=-1)
    return feats.cpu()


def _prompt_for_row(row: Mapping[str, Any], template: str) -> str:
    return template.format(phrase=str(row["phrase"]))


def _save_cache(path: Path, cache: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(cache), path)
    print(f"saved cache: {path}")


def _load_cache(path: Path) -> Dict[str, Any]:
    try:
        cache = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        cache = torch.load(path, map_location="cpu")
    print(f"loaded cache: {path}")
    return cache


def _encode_rows_to_cache(
    rows: Sequence[Mapping[str, Any]],
    *,
    image_root: Path,
    crop_margin: float,
    prompt_template: str,
    model,
    preprocess,
    tokenizer,
    image_batch_size: int,
    text_batch_size: int,
    device: str,
) -> Dict[str, Any]:
    prompts = sorted({_prompt_for_row(row, prompt_template) for row in rows})
    text_features = _encode_texts(
        model,
        tokenizer,
        prompts,
        batch_size=text_batch_size,
        device=device,
    )
    text_by_prompt = {prompt: text_features[idx] for idx, prompt in enumerate(prompts)}

    rows_by_file: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        rows_by_file[str(row["file_name"])].append(row)

    crop_feats: List[torch.Tensor] = []
    text_feats: List[torch.Tensor] = []
    labels: List[float] = []
    detector_scores: List[float] = []
    negative_types: List[str] = []
    target_category_ids: List[int] = []
    image_ids: List[int] = []

    pending_crops: List[torch.Tensor] = []
    pending_rows: List[Mapping[str, Any]] = []
    missing_images = 0
    invalid_crops = 0

    def flush() -> None:
        if not pending_crops:
            return
        feats = _encode_crops(model, pending_crops, device=device)
        for feat, row in zip(feats, pending_rows):
            prompt = _prompt_for_row(row, prompt_template)
            crop_feats.append(feat)
            text_feats.append(text_by_prompt[prompt])
            labels.append(float(row["label"]))
            detector_scores.append(float(row.get("detector_score", 0.0)))
            negative_types.append(_negative_kind(row))
            target_category_ids.append(int(row.get("target_category_id", -1)))
            image_ids.append(int(row.get("image_id", -1)))
        pending_crops.clear()
        pending_rows.clear()

    for file_name, file_rows in tqdm(rows_by_file.items(), desc="encoding crop images"):
        image_path = _resolve_image_path(image_root, file_name)
        if not image_path.exists():
            missing_images += len(file_rows)
            continue

        image = Image.open(image_path).convert("RGB")
        width, height = image.size
        for row in file_rows:
            xyxy = _expanded_xyxy(row["bbox"], width=width, height=height, margin=crop_margin)
            if xyxy is None:
                invalid_crops += 1
                continue
            pending_crops.append(preprocess(image.crop(xyxy)))
            pending_rows.append(row)
            if len(pending_crops) >= image_batch_size:
                flush()
        image.close()

    flush()
    if not crop_feats:
        raise RuntimeError("No valid crop features were encoded.")

    cache = {
        "crop_feats": torch.stack(crop_feats, dim=0).float(),
        "text_feats": torch.stack(text_feats, dim=0).float(),
        "labels": torch.tensor(labels, dtype=torch.float32),
        "detector_scores": torch.tensor(detector_scores, dtype=torch.float32),
        "negative_types": negative_types,
        "target_category_ids": torch.tensor(target_category_ids, dtype=torch.long),
        "image_ids": torch.tensor(image_ids, dtype=torch.long),
        "meta": {
            "num_input_rows": len(rows),
            "num_encoded_rows": len(labels),
            "missing_images": missing_images,
            "invalid_crops": invalid_crops,
            "crop_margin": crop_margin,
            "prompt_template": prompt_template,
        },
    }
    print(json.dumps(cache["meta"], indent=2))
    return cache


def _cache_path(cache_dir: Optional[Path], split: str) -> Optional[Path]:
    if cache_dir is None:
        return None
    return cache_dir / f"{split}_features.pt"


def _load_or_build_cache(
    *,
    split: str,
    jsonl_path: Optional[Path],
    explicit_cache_path: Optional[Path],
    cache_dir: Optional[Path],
    args: argparse.Namespace,
    model=None,
    preprocess=None,
    tokenizer=None,
) -> Dict[str, Any]:
    cache_path = explicit_cache_path or _cache_path(cache_dir, split)
    if cache_path is not None and cache_path.exists() and not args.overwrite_cache:
        return _load_cache(cache_path)

    if jsonl_path is None:
        raise ValueError(f"{split} cache does not exist; pass --{split}-jsonl to build it.")
    if model is None or preprocess is None or tokenizer is None:
        raise ValueError("OpenCLIP model/preprocess/tokenizer are required to build feature caches.")

    rows = _read_jsonl(jsonl_path)
    rows = _sample_rows(
        rows,
        same_phrase_neg_per_pos=args.same_phrase_neg_per_pos,
        wrong_phrase_neg_per_pos=args.wrong_phrase_neg_per_pos,
        max_positives=args.max_positives,
        max_samples=args.max_train_samples if split == "train" else args.max_val_samples,
        seed=args.seed if split == "train" else args.seed + 1,
    )
    _summarize_rows(f"selected {split}", rows)

    cache = _encode_rows_to_cache(
        rows,
        image_root=args.image_root,
        crop_margin=args.crop_margin,
        prompt_template=args.prompt_template,
        model=model,
        preprocess=preprocess,
        tokenizer=tokenizer,
        image_batch_size=args.image_batch_size,
        text_batch_size=args.text_batch_size,
        device=args.device,
    )
    if cache_path is not None:
        _save_cache(cache_path, cache)
    return cache


class PairFeatureDataset(Dataset):
    def __init__(self, cache: Mapping[str, Any]) -> None:
        self.crop_feats = cache["crop_feats"].float()
        self.text_feats = cache["text_feats"].float()
        self.labels = cache["labels"].float()
        self.detector_scores = cache["detector_scores"].float()
        if self.crop_feats.shape != self.text_feats.shape:
            raise ValueError(
                f"crop/text feature shape mismatch: {self.crop_feats.shape} vs {self.text_feats.shape}"
            )

    def __len__(self) -> int:
        return int(self.labels.numel())

    @property
    def input_dim(self) -> int:
        return int(self.crop_feats.shape[1] * 4 + 1)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        crop = self.crop_feats[index]
        text = self.text_feats[index]
        score = self.detector_scores[index].view(1)
        features = torch.cat([crop, text, crop * text, torch.abs(crop - text), score], dim=0)
        return features, self.labels[index]


def _validate_cache_labels(name: str, cache: Mapping[str, Any]) -> None:
    labels = cache["labels"].float()
    positives = int(labels.sum().item())
    negatives = int(labels.numel()) - positives
    if positives == 0 or negatives == 0:
        raise ValueError(
            f"{name} cache has {positives} positives and {negatives} negatives. "
            "The verifier needs both classes; remove stale caches or rerun with --overwrite-cache."
        )


class CropDescriptionVerifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        second_hidden = max(64, hidden_dim // 2)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, second_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(second_hidden, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features).squeeze(-1)


def _average_precision(scores: np.ndarray, labels: np.ndarray) -> float:
    labels = labels.astype(np.int64)
    positives = int(labels.sum())
    if positives == 0:
        return float("nan")
    order = np.argsort(-scores, kind="mergesort")
    sorted_labels = labels[order]
    tp = np.cumsum(sorted_labels)
    precision = tp / (np.arange(len(sorted_labels)) + 1)
    return float((precision * sorted_labels).sum() / positives)


def _roc_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    labels = labels.astype(np.int64)
    n_pos = int(labels.sum())
    n_neg = int(len(labels) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1, dtype=np.float64)
    pos_rank_sum = ranks[labels == 1].sum()
    auc = (pos_rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def _binary_metrics(logits: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    labels = labels.astype(np.int64)
    predictions = (logits >= 0.0).astype(np.int64)
    return {
        "n": int(len(labels)),
        "positive": int(labels.sum()),
        "negative": int(len(labels) - labels.sum()),
        "positive_rate": float(labels.mean()) if len(labels) else float("nan"),
        "auc": _roc_auc(logits, labels),
        "ap": _average_precision(logits, labels),
        "accuracy_at_0_5": float((predictions == labels).mean()) if len(labels) else float("nan"),
    }


def _subset_metrics(
    logits: np.ndarray,
    labels: np.ndarray,
    negative_types: Sequence[str],
    *,
    prefix: str,
) -> Dict[str, float]:
    if prefix == "overall":
        mask = np.ones(len(labels), dtype=bool)
    elif prefix == "same_phrase_bad_box":
        mask = np.asarray(
            [(label == 1) or (kind == "same_phrase_bad_box") for label, kind in zip(labels, negative_types)],
            dtype=bool,
        )
    elif prefix == "wrong_phrase_same_region":
        mask = np.asarray(
            [
                (label == 1) or str(kind).startswith("wrong_phrase_same_region")
                for label, kind in zip(labels, negative_types)
            ],
            dtype=bool,
        )
    else:
        raise ValueError(prefix)
    return _binary_metrics(logits[mask], labels[mask])


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataset: PairFeatureDataset,
    cache: Mapping[str, Any],
    *,
    batch_size: int,
    device: str,
    num_workers: int,
) -> Dict[str, Any]:
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    logits_chunks = []
    label_chunks = []
    total_loss = 0.0
    total_count = 0

    for features, labels in loader:
        features = features.to(device)
        labels = labels.to(device)
        logits = model(features)
        loss = F.binary_cross_entropy_with_logits(logits, labels, reduction="sum")
        total_loss += float(loss.item())
        total_count += int(labels.numel())
        logits_chunks.append(logits.cpu())
        label_chunks.append(labels.cpu())

    logits_np = torch.cat(logits_chunks).numpy()
    labels_np = torch.cat(label_chunks).numpy()
    negative_types = list(cache["negative_types"])
    metrics = {
        "loss": total_loss / max(1, total_count),
        "overall": _subset_metrics(logits_np, labels_np, negative_types, prefix="overall"),
        "same_phrase_bad_box": _subset_metrics(
            logits_np,
            labels_np,
            negative_types,
            prefix="same_phrase_bad_box",
        ),
        "wrong_phrase_same_region": _subset_metrics(
            logits_np,
            labels_np,
            negative_types,
            prefix="wrong_phrase_same_region",
        ),
    }
    return metrics


def train(args: argparse.Namespace, train_cache: Mapping[str, Any], val_cache: Mapping[str, Any]) -> Dict[str, Any]:
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    train_dataset = PairFeatureDataset(train_cache)
    val_dataset = PairFeatureDataset(val_cache)
    model = CropDescriptionVerifier(
        input_dim=train_dataset.input_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    ).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=False,
    )

    history: List[Dict[str, Any]] = []
    best_score = -float("inf")
    best_epoch = -1
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        train_count = 0
        for features, labels in tqdm(loader, desc=f"epoch {epoch}/{args.epochs}"):
            features = features.to(args.device)
            labels = labels.to(args.device)
            logits = model(features)
            loss = F.binary_cross_entropy_with_logits(logits, labels)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            train_loss += float(loss.item()) * int(labels.numel())
            train_count += int(labels.numel())

        val_metrics = evaluate(
            model,
            val_dataset,
            val_cache,
            batch_size=args.batch_size,
            device=args.device,
            num_workers=args.num_workers,
        )
        epoch_record = {
            "epoch": epoch,
            "train_loss": train_loss / max(1, train_count),
            "val": val_metrics,
        }
        history.append(epoch_record)

        wrong_ap = float(val_metrics["wrong_phrase_same_region"]["ap"])
        overall_ap = float(val_metrics["overall"]["ap"])
        score = wrong_ap if not math.isnan(wrong_ap) else overall_ap
        if score > best_score:
            best_score = score
            best_epoch = epoch
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "input_dim": train_dataset.input_dim,
                    "hidden_dim": args.hidden_dim,
                    "dropout": args.dropout,
                    "epoch": epoch,
                    "score": score,
                    "args": _jsonable_args(args),
                    "val": val_metrics,
                },
                args.output_dir / "verifier_best.pt",
            )

        print(
            json.dumps(
                {
                    "epoch": epoch,
                    "train_loss": epoch_record["train_loss"],
                    "overall_ap": val_metrics["overall"]["ap"],
                    "overall_auc": val_metrics["overall"]["auc"],
                    "wrong_phrase_ap": val_metrics["wrong_phrase_same_region"]["ap"],
                    "wrong_phrase_auc": val_metrics["wrong_phrase_same_region"]["auc"],
                    "same_phrase_ap": val_metrics["same_phrase_bad_box"]["ap"],
                    "same_phrase_auc": val_metrics["same_phrase_bad_box"]["auc"],
                },
                sort_keys=True,
            )
        )

        with (args.output_dir / "metrics.json").open("w") as f:
            json.dump(
                {
                    "best_epoch": best_epoch,
                    "best_score": best_score,
                    "history": history,
                    "args": _jsonable_args(args),
                    "train_cache_meta": train_cache.get("meta", {}),
                    "val_cache_meta": val_cache.get("meta", {}),
                },
                f,
                indent=2,
            )

    return {
        "best_epoch": best_epoch,
        "best_score": best_score,
        "history": history,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-jsonl", type=Path, default=None)
    parser.add_argument("--val-jsonl", type=Path, default=None)
    parser.add_argument("--train-cache", type=Path, default=None)
    parser.add_argument("--val-cache", type=Path, default=None)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("output/d3_crop_verifier_w075"))
    parser.add_argument("--image-root", type=Path, default=Path("dataset/d3/images"))
    parser.add_argument("--prompt-template", default="the described target is {phrase}")
    parser.add_argument("--model", default="convnext_large_d_320")
    parser.add_argument("--pretrained", default="laion2b_s29b_b131k_ft_soup")
    parser.add_argument("--crop-margin", type=float, default=0.1)
    parser.add_argument("--same-phrase-neg-per-pos", type=float, default=1.0)
    parser.add_argument("--wrong-phrase-neg-per-pos", type=float, default=2.0)
    parser.add_argument("--max-positives", type=int, default=None)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--overwrite-cache", action="store_true")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--image-batch-size", type=int, default=64)
    parser.add_argument("--text-batch-size", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_cache_path = args.train_cache or _cache_path(args.cache_dir, "train")
    val_cache_path = args.val_cache or _cache_path(args.cache_dir, "val")
    needs_encoding = False
    for path in (train_cache_path, val_cache_path):
        if path is None or args.overwrite_cache or not path.exists():
            needs_encoding = True

    model = preprocess = tokenizer = None
    if needs_encoding:
        model, preprocess, tokenizer = _load_openclip(args.model, args.pretrained, args.device)

    train_cache = _load_or_build_cache(
        split="train",
        jsonl_path=args.train_jsonl,
        explicit_cache_path=args.train_cache,
        cache_dir=args.cache_dir,
        args=args,
        model=model,
        preprocess=preprocess,
        tokenizer=tokenizer,
    )
    val_cache = _load_or_build_cache(
        split="val",
        jsonl_path=args.val_jsonl,
        explicit_cache_path=args.val_cache,
        cache_dir=args.cache_dir,
        args=args,
        model=model,
        preprocess=preprocess,
        tokenizer=tokenizer,
    )

    _validate_cache_labels("train", train_cache)
    _validate_cache_labels("val", val_cache)
    print(f"train encoded rows: {len(train_cache['labels'])}")
    print(f"val encoded rows: {len(val_cache['labels'])}")
    result = train(args, train_cache, val_cache)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
