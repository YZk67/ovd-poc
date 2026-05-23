#!/usr/bin/env python3
"""
Train a D3 top-k candidate reranker from detector ROI features.

This is the first trained version of the top300 box-phrase reranker. It consumes
JSONL rows from tools/export_d3_topk_candidate_pairs.py, maps each row back to
the saved detector ROI feature by (file_name, query_index), and trains the same
small verifier architecture that DINO can load through
model.region_verifier_checkpoint.
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
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
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


def _saved_prediction_path(saved_output_dir: Path, file_name: str) -> Path:
    path = Path(file_name)
    return saved_output_dir / path.with_suffix(".pth").name


def _torch_load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _load_roi_features(path: Path) -> torch.Tensor:
    data = _torch_load(path)
    if "roi_features_ori" not in data:
        raise KeyError(f"{path} is missing roi_features_ori.")
    roi_features = data["roi_features_ori"]
    if roi_features.ndim == 3:
        roi_features = roi_features[0]
    if roi_features.ndim != 2:
        raise ValueError(f"{path} roi_features_ori has unsupported shape {tuple(roi_features.shape)}.")
    return F.normalize(roi_features.float(), p=2, dim=-1).cpu()


def _load_text_bank(path: Path, *, text_index: Optional[int]) -> torch.Tensor:
    bank = np.load(path)
    if bank.ndim == 3:
        if text_index is None:
            if bank.shape[1] != 1:
                raise ValueError(
                    f"{path} has shape {bank.shape}; pass --text-index for a multi-prompt bank."
                )
            bank = bank[:, 0, :]
        else:
            bank = bank[:, text_index, :]
    if bank.ndim != 2:
        raise ValueError(f"Expected 2D text bank after selection, got shape {bank.shape}.")
    tensor = torch.from_numpy(bank).float()
    return F.normalize(tensor, p=2, dim=-1).cpu()


def _sample_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    max_samples: Optional[int],
    seed: int,
) -> List[Dict[str, Any]]:
    rows = [dict(row) for row in rows]
    if max_samples is None or max_samples <= 0 or len(rows) <= max_samples:
        return rows

    rng = random.Random(seed)
    positives = [row for row in rows if int(row.get("label", 0)) == 1]
    negatives = [row for row in rows if int(row.get("label", 0)) == 0]
    if len(positives) >= max_samples:
        selected = rng.sample(positives, max_samples)
    else:
        selected = positives + rng.sample(negatives, max_samples - len(positives))
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


def _cache_path(cache_dir: Optional[Path], split: str) -> Optional[Path]:
    if cache_dir is None:
        return None
    return cache_dir / f"{split}_features.pt"


def _save_cache(path: Path, cache: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(cache), path)
    print(f"saved cache: {path}")


def _load_cache(path: Path) -> Dict[str, Any]:
    cache = _torch_load(path)
    print(f"loaded cache: {path}")
    return cache


def _encode_rows_to_cache(
    rows: Sequence[Mapping[str, Any]],
    *,
    saved_output_dir: Path,
    store_fp16: bool,
    split: str,
) -> Dict[str, Any]:
    rows_by_file: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        rows_by_file[str(row["file_name"])].append(row)

    region_feats: List[torch.Tensor] = []
    labels: List[float] = []
    detector_scores: List[float] = []
    negative_types: List[str] = []
    target_category_ids: List[int] = []
    image_ids: List[int] = []
    query_indices: List[int] = []
    query_ranks: List[int] = []
    phrase_ranks: List[int] = []
    target_ious: List[float] = []
    best_any_ious: List[float] = []

    missing_outputs = 0
    invalid_query_indices = 0
    invalid_categories = 0

    for file_name, file_rows in tqdm(rows_by_file.items(), desc=f"encoding {split} candidate rows"):
        saved_path = _saved_prediction_path(saved_output_dir, file_name)
        if not saved_path.exists():
            missing_outputs += len(file_rows)
            continue
        roi_features = _load_roi_features(saved_path)

        for row in file_rows:
            query_index = int(row.get("query_index", -1))
            if query_index < 0 or query_index >= roi_features.shape[0]:
                invalid_query_indices += 1
                continue
            target_category_id = int(row.get("target_category_id", -1))
            if target_category_id <= 0:
                invalid_categories += 1
                continue

            region_feats.append(roi_features[query_index])
            labels.append(float(row["label"]))
            detector_scores.append(float(row.get("detector_score", 0.0)))
            negative_types.append(_negative_kind(row))
            target_category_ids.append(target_category_id)
            image_ids.append(int(row.get("image_id", -1)))
            query_indices.append(query_index)
            query_ranks.append(int(row.get("query_rank", -1)))
            phrase_ranks.append(int(row.get("phrase_rank", -1)))
            target_ious.append(float(row.get("target_iou", 0.0)))
            best_any_ious.append(float(row.get("best_any_iou", 0.0)))

    if not region_feats:
        raise RuntimeError("No candidate ROI features were encoded.")

    region_tensor = torch.stack(region_feats, dim=0).float()
    if store_fp16:
        region_tensor = region_tensor.half()

    cache = {
        "region_feats": region_tensor,
        "labels": torch.tensor(labels, dtype=torch.float32),
        "detector_scores": torch.tensor(detector_scores, dtype=torch.float32),
        "negative_types": negative_types,
        "target_category_ids": torch.tensor(target_category_ids, dtype=torch.long),
        "image_ids": torch.tensor(image_ids, dtype=torch.long),
        "query_indices": torch.tensor(query_indices, dtype=torch.long),
        "query_ranks": torch.tensor(query_ranks, dtype=torch.long),
        "phrase_ranks": torch.tensor(phrase_ranks, dtype=torch.long),
        "target_ious": torch.tensor(target_ious, dtype=torch.float32),
        "best_any_ious": torch.tensor(best_any_ious, dtype=torch.float32),
        "meta": {
            "feature_source": "detector_topk_candidate_roi_features",
            "num_input_rows": len(rows),
            "num_encoded_rows": len(labels),
            "missing_outputs": missing_outputs,
            "invalid_query_indices": invalid_query_indices,
            "invalid_categories": invalid_categories,
            "store_fp16": store_fp16,
        },
    }
    print(json.dumps(cache["meta"], indent=2, sort_keys=True))
    return cache


def _load_or_build_cache(
    *,
    split: str,
    jsonl_path: Optional[Path],
    explicit_cache_path: Optional[Path],
    cache_dir: Optional[Path],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    cache_path = explicit_cache_path or _cache_path(cache_dir, split)
    if cache_path is not None and cache_path.exists() and not args.overwrite_cache:
        return _load_cache(cache_path)

    if jsonl_path is None:
        raise ValueError(f"{split} cache does not exist; pass --{split}-jsonl to build it.")

    rows = _read_jsonl(jsonl_path)
    rows = _sample_rows(
        rows,
        max_samples=args.max_train_samples if split == "train" else args.max_val_samples,
        seed=args.seed if split == "train" else args.seed + 1,
    )
    _summarize_rows(f"selected {split}", rows)
    cache = _encode_rows_to_cache(
        rows,
        saved_output_dir=args.saved_output_dir,
        store_fp16=args.cache_fp16,
        split=split,
    )
    if cache_path is not None:
        _save_cache(cache_path, cache)
    return cache


def _build_pair_features(
    region_feats: torch.Tensor,
    text_feats: torch.Tensor,
    detector_scores: torch.Tensor,
    feature_mode: str,
) -> torch.Tensor:
    score = detector_scores.to(dtype=region_feats.dtype).view(-1, 1)
    if feature_mode == "full":
        return torch.cat(
            [
                region_feats,
                text_feats,
                region_feats * text_feats,
                torch.abs(region_feats - text_feats),
                score,
            ],
            dim=-1,
        )
    if feature_mode == "no_text":
        return torch.cat([region_feats, score], dim=-1)
    if feature_mode == "no_detector_score":
        return torch.cat(
            [
                region_feats,
                text_feats,
                region_feats * text_feats,
                torch.abs(region_feats - text_feats),
            ],
            dim=-1,
        )
    raise ValueError(f"Unsupported feature mode: {feature_mode}")


class CandidatePairDataset(Dataset):
    def __init__(
        self,
        cache: Mapping[str, Any],
        *,
        text_bank: torch.Tensor,
        feature_mode: str,
        rank_group: str,
    ) -> None:
        self.region_feats = cache["region_feats"]
        self.labels = cache["labels"].float()
        self.detector_scores = cache["detector_scores"].float()
        self.target_category_ids = cache["target_category_ids"].long()
        self.image_ids = cache["image_ids"].long()
        self.negative_types = list(cache["negative_types"])
        self.text_bank = text_bank.float()
        self.feature_mode = feature_mode
        self.rank_group = rank_group

        if int(self.target_category_ids.min().item()) <= 0:
            raise ValueError("target_category_ids must be 1-based positive ids.")
        if int(self.target_category_ids.max().item()) > self.text_bank.shape[0]:
            raise ValueError(
                f"target_category_id max={int(self.target_category_ids.max().item())} "
                f"but text bank has {self.text_bank.shape[0]} rows."
            )
        if self.region_feats.shape[0] != self.labels.numel():
            raise ValueError("region_feats/labels row count mismatch.")
        if self.image_ids.numel() != self.labels.numel():
            raise ValueError("image_ids/labels row count mismatch.")
        if len(self.negative_types) != self.labels.numel():
            raise ValueError("negative_types/labels row count mismatch.")
        if self.region_feats.shape[-1] != self.text_bank.shape[-1]:
            raise ValueError(
                f"ROI/text feature dim mismatch: {self.region_feats.shape[-1]} vs {self.text_bank.shape[-1]}."
            )
        if rank_group not in {"image", "image_category"}:
            raise ValueError(f"Unsupported rank_group={rank_group!r}.")
        if rank_group == "image":
            self.group_ids = self.image_ids
        else:
            stride = int(self.target_category_ids.max().item()) + 1
            self.group_ids = self.image_ids * stride + self.target_category_ids

    def __len__(self) -> int:
        return int(self.labels.numel())

    @property
    def input_dim(self) -> int:
        return int(
            _build_pair_features(
                self.region_feats[:1].float(),
                self.text_bank[self.target_category_ids[:1] - 1],
                self.detector_scores[:1],
                self.feature_mode,
            ).shape[1]
        )

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]:
        region = self.region_feats[index].float()
        text = self.text_bank[int(self.target_category_ids[index].item()) - 1]
        score = self.detector_scores[index : index + 1]
        features = _build_pair_features(
            region.view(1, -1),
            text.view(1, -1),
            score,
            self.feature_mode,
        ).squeeze(0)
        return features, self.labels[index], self.group_ids[index], self.negative_types[index]


class CandidateReranker(nn.Module):
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
    sorted_scores = scores[order]
    sorted_labels = labels[order]

    ap = 0.0
    tp = 0
    fp = 0
    previous_recall = 0.0
    start = 0
    while start < len(sorted_labels):
        end = start + 1
        while end < len(sorted_labels) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        group = sorted_labels[start:end]
        tp += int(group.sum())
        fp += int(len(group) - group.sum())
        recall = tp / positives
        precision = tp / max(1, tp + fp)
        ap += (recall - previous_recall) * precision
        previous_recall = recall
        start = end
    return float(ap)


def _roc_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    labels = labels.astype(np.int64)
    n_pos = int(labels.sum())
    n_neg = int(len(labels) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(len(scores), dtype=np.float64)
    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        average_rank = (start + 1 + end) / 2.0
        ranks[order[start:end]] = average_rank
        start = end
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
    subset: str,
) -> Dict[str, float]:
    if subset == "overall":
        mask = np.ones(len(labels), dtype=bool)
    elif subset == "wrong_phrase_good_box":
        mask = np.asarray(
            [(label == 1) or (kind == "wrong_phrase_good_box") for label, kind in zip(labels, negative_types)],
            dtype=bool,
        )
    elif subset == "same_phrase_bad_box":
        mask = np.asarray(
            [(label == 1) or (kind == "same_phrase_bad_box") for label, kind in zip(labels, negative_types)],
            dtype=bool,
        )
    elif subset == "background_bad_box":
        mask = np.asarray(
            [(label == 1) or (kind == "background_bad_box") for label, kind in zip(labels, negative_types)],
            dtype=bool,
        )
    else:
        raise ValueError(subset)
    return _binary_metrics(logits[mask], labels[mask])


def _detector_baseline_metrics(cache: Mapping[str, Any]) -> Dict[str, Any]:
    scores = cache["detector_scores"].float().numpy()
    labels = cache["labels"].float().numpy()
    negative_types = list(cache["negative_types"])
    return {
        "overall": _subset_metrics(scores, labels, negative_types, subset="overall"),
        "wrong_phrase_good_box": _subset_metrics(scores, labels, negative_types, subset="wrong_phrase_good_box"),
        "same_phrase_bad_box": _subset_metrics(scores, labels, negative_types, subset="same_phrase_bad_box"),
        "background_bad_box": _subset_metrics(scores, labels, negative_types, subset="background_bad_box"),
    }


def _parse_rank_neg_types(value: str) -> Optional[set]:
    value = value.strip()
    if value.lower() == "all":
        return None
    return {item.strip() for item in value.split(",") if item.strip()}


def _pairwise_rank_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    group_ids: torch.Tensor,
    negative_types: Sequence[str],
    *,
    margin: float,
    rank_neg_types: Optional[set],
    max_pairs_per_group: int,
) -> Tuple[torch.Tensor, int, int]:
    if rank_neg_types is None:
        neg_type_mask = torch.ones_like(labels, dtype=torch.bool)
    else:
        neg_type_mask = torch.tensor(
            [str(kind) in rank_neg_types for kind in negative_types],
            dtype=torch.bool,
            device=logits.device,
        )

    losses = []
    rank_pair_count = 0
    rank_group_count = 0
    for group_id in torch.unique(group_ids):
        in_group = group_ids == group_id
        pos_logits = logits[in_group & (labels > 0.5)]
        neg_logits = logits[in_group & (labels <= 0.5) & neg_type_mask]
        if pos_logits.numel() == 0 or neg_logits.numel() == 0:
            continue

        pair_losses = F.softplus(margin - (pos_logits[:, None] - neg_logits[None, :])).reshape(-1)
        if max_pairs_per_group > 0 and pair_losses.numel() > max_pairs_per_group:
            keep = torch.randperm(pair_losses.numel(), device=pair_losses.device)[:max_pairs_per_group]
            pair_losses = pair_losses[keep]

        losses.append(pair_losses.mean())
        rank_pair_count += int(pair_losses.numel())
        rank_group_count += 1

    if not losses:
        return logits.sum() * 0.0, 0, 0
    return torch.stack(losses).mean(), rank_pair_count, rank_group_count


def _compute_train_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    group_ids: torch.Tensor,
    negative_types: Sequence[str],
    *,
    args: argparse.Namespace,
    pos_weight: torch.Tensor,
    rank_neg_types: Optional[set],
) -> Tuple[torch.Tensor, Dict[str, float]]:
    bce_loss = F.binary_cross_entropy_with_logits(logits, labels, pos_weight=pos_weight)
    rank_loss, rank_pair_count, rank_group_count = _pairwise_rank_loss(
        logits,
        labels,
        group_ids,
        negative_types,
        margin=args.ranking_margin,
        rank_neg_types=rank_neg_types,
        max_pairs_per_group=args.rank_max_pairs_per_group,
    )

    if args.loss_type == "bce":
        loss = bce_loss
    elif args.loss_type == "pairwise_rank":
        loss = rank_loss
    elif args.loss_type == "bce_pairwise":
        loss = bce_loss + args.rank_loss_weight * rank_loss
    else:
        raise ValueError(f"Unsupported loss_type={args.loss_type!r}.")

    return loss, {
        "bce_loss": float(bce_loss.detach().item()),
        "rank_loss": float(rank_loss.detach().item()),
        "rank_pair_count": float(rank_pair_count),
        "rank_group_count": float(rank_group_count),
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataset: CandidatePairDataset,
    cache: Mapping[str, Any],
    *,
    batch_size: int,
    device: str,
    num_workers: int,
    pos_weight: Optional[torch.Tensor],
) -> Dict[str, Any]:
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    logits_chunks = []
    label_chunks = []
    total_loss = 0.0
    total_count = 0

    for features, labels, _, _ in loader:
        features = features.to(device)
        labels = labels.to(device)
        logits = model(features)
        loss = F.binary_cross_entropy_with_logits(
            logits,
            labels,
            pos_weight=pos_weight,
            reduction="sum",
        )
        total_loss += float(loss.item())
        total_count += int(labels.numel())
        logits_chunks.append(logits.cpu())
        label_chunks.append(labels.cpu())

    logits_np = torch.cat(logits_chunks).numpy()
    labels_np = torch.cat(label_chunks).numpy()
    negative_types = list(cache["negative_types"])
    return {
        "loss": total_loss / max(1, total_count),
        "overall": _subset_metrics(logits_np, labels_np, negative_types, subset="overall"),
        "wrong_phrase_good_box": _subset_metrics(
            logits_np,
            labels_np,
            negative_types,
            subset="wrong_phrase_good_box",
        ),
        "same_phrase_bad_box": _subset_metrics(
            logits_np,
            labels_np,
            negative_types,
            subset="same_phrase_bad_box",
        ),
        "background_bad_box": _subset_metrics(
            logits_np,
            labels_np,
            negative_types,
            subset="background_bad_box",
        ),
    }


def _auto_pos_weight(cache: Mapping[str, Any]) -> float:
    labels = cache["labels"].float()
    pos = float(labels.sum().item())
    neg = float(labels.numel() - labels.sum().item())
    if pos <= 0:
        raise ValueError("Training cache has no positive labels.")
    return neg / pos


def _validate_cache_labels(name: str, cache: Mapping[str, Any]) -> None:
    labels = cache["labels"].float()
    positives = int(labels.sum().item())
    negatives = int(labels.numel()) - positives
    if positives == 0 or negatives == 0:
        raise ValueError(f"{name} cache has {positives} positives and {negatives} negatives.")


def train(args: argparse.Namespace, train_cache: Mapping[str, Any], val_cache: Mapping[str, Any]) -> Dict[str, Any]:
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    text_bank = _load_text_bank(args.text_embedding, text_index=args.text_index)
    train_dataset = CandidatePairDataset(
        train_cache,
        text_bank=text_bank,
        feature_mode=args.feature_mode,
        rank_group=args.rank_group,
    )
    val_dataset = CandidatePairDataset(
        val_cache,
        text_bank=text_bank,
        feature_mode=args.feature_mode,
        rank_group=args.rank_group,
    )

    model = CandidateReranker(
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
        pin_memory=args.device.startswith("cuda"),
    )

    if args.pos_weight == "auto":
        pos_weight_value = _auto_pos_weight(train_cache)
    else:
        pos_weight_value = float(args.pos_weight)
    pos_weight = torch.tensor(pos_weight_value, dtype=torch.float32, device=args.device)
    rank_neg_types = _parse_rank_neg_types(args.rank_neg_types)

    history: List[Dict[str, Any]] = []
    best_score = -float("inf")
    best_epoch = -1
    args.output_dir.mkdir(parents=True, exist_ok=True)

    detector_baseline = _detector_baseline_metrics(val_cache)
    print("val detector baseline:")
    print(json.dumps(detector_baseline, indent=2, sort_keys=True))

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        train_bce_loss = 0.0
        train_rank_loss = 0.0
        train_count = 0
        train_rank_pairs = 0.0
        train_rank_groups = 0.0
        for features, labels, group_ids, negative_types in tqdm(loader, desc=f"epoch {epoch}/{args.epochs}"):
            features = features.to(args.device)
            labels = labels.to(args.device)
            group_ids = group_ids.to(args.device)
            logits = model(features)
            loss, loss_parts = _compute_train_loss(
                logits,
                labels,
                group_ids,
                negative_types,
                args=args,
                pos_weight=pos_weight,
                rank_neg_types=rank_neg_types,
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            train_loss += float(loss.item()) * int(labels.numel())
            train_bce_loss += loss_parts["bce_loss"] * int(labels.numel())
            train_rank_loss += loss_parts["rank_loss"] * int(labels.numel())
            train_rank_pairs += loss_parts["rank_pair_count"]
            train_rank_groups += loss_parts["rank_group_count"]
            train_count += int(labels.numel())

        val_metrics = evaluate(
            model,
            val_dataset,
            val_cache,
            batch_size=args.batch_size,
            device=args.device,
            num_workers=args.num_workers,
            pos_weight=pos_weight,
        )
        epoch_record = {
            "epoch": epoch,
            "train_loss": train_loss / max(1, train_count),
            "train_bce_loss": train_bce_loss / max(1, train_count),
            "train_rank_loss": train_rank_loss / max(1, train_count),
            "train_rank_pairs": train_rank_pairs,
            "train_rank_groups": train_rank_groups,
            "val": val_metrics,
        }
        history.append(epoch_record)

        wrong_ap = float(val_metrics["wrong_phrase_good_box"]["ap"])
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
                    "feature_mode": args.feature_mode,
                    "epoch": epoch,
                    "score": score,
                    "args": _jsonable_args(args),
                    "val": val_metrics,
                    "detector_baseline": detector_baseline,
                },
                args.output_dir / "verifier_best.pt",
            )

        print(
            json.dumps(
                {
                    "epoch": epoch,
                    "train_loss": epoch_record["train_loss"],
                    "train_bce_loss": epoch_record["train_bce_loss"],
                    "train_rank_loss": epoch_record["train_rank_loss"],
                    "train_rank_pairs": train_rank_pairs,
                    "train_rank_groups": train_rank_groups,
                    "overall_ap": val_metrics["overall"]["ap"],
                    "overall_auc": val_metrics["overall"]["auc"],
                    "wrong_phrase_ap": val_metrics["wrong_phrase_good_box"]["ap"],
                    "wrong_phrase_auc": val_metrics["wrong_phrase_good_box"]["auc"],
                    "same_phrase_ap": val_metrics["same_phrase_bad_box"]["ap"],
                    "same_phrase_auc": val_metrics["same_phrase_bad_box"]["auc"],
                    "background_ap": val_metrics["background_bad_box"]["ap"],
                    "background_auc": val_metrics["background_bad_box"]["auc"],
                    "best_epoch": best_epoch,
                    "best_score": best_score,
                },
                sort_keys=True,
            )
        )

        with (args.output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
            json.dump(
                {
                    "best_epoch": best_epoch,
                    "best_score": best_score,
                    "history": history,
                    "detector_baseline": detector_baseline,
                    "args": _jsonable_args(args),
                    "pos_weight": pos_weight_value,
                    "train_cache_meta": train_cache.get("meta", {}),
                    "val_cache_meta": val_cache.get("meta", {}),
                },
                handle,
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
    parser.add_argument(
        "--saved-output-dir",
        type=Path,
        default=Path("output/d3_topk_candidate_dumps_w075/pth"),
        help="Directory of detector dumps with roi_features_ori.",
    )
    parser.add_argument(
        "--text-embedding",
        type=Path,
        default=Path("dataset/metadata/d3_description_anchor_target_w100_bank_convnextl.npy"),
        help="Target phrase text embedding bank. Category id c maps to row c-1.",
    )
    parser.add_argument("--text-index", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("output/d3_candidate_reranker_w075_top300x50"))
    parser.add_argument("--overwrite-cache", action="store_true")
    parser.add_argument("--cache-fp16", action="store_true")
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument(
        "--feature-mode",
        choices=("full", "no_text", "no_detector_score"),
        default="no_detector_score",
        help=(
            "Verifier input features. full uses region/text/interactions/score; "
            "no_text uses region/score only; no_detector_score removes detector score."
        ),
    )
    parser.add_argument(
        "--pos-weight",
        default="auto",
        help="Positive BCE weight. Use 'auto' for neg/pos, or a numeric value.",
    )
    parser.add_argument(
        "--loss-type",
        choices=("bce", "pairwise_rank", "bce_pairwise"),
        default="bce",
        help="Training objective. pairwise_rank optimizes positive > hard negative within a group.",
    )
    parser.add_argument(
        "--rank-group",
        choices=("image", "image_category"),
        default="image",
        help="Group used by pairwise_rank loss.",
    )
    parser.add_argument(
        "--rank-neg-types",
        default="wrong_phrase_good_box,same_phrase_bad_box",
        help="Comma-separated negative_type list for ranking, or 'all'.",
    )
    parser.add_argument("--ranking-margin", type=float, default=0.0)
    parser.add_argument(
        "--rank-loss-weight",
        type=float,
        default=1.0,
        help="Pairwise rank loss weight when --loss-type=bce_pairwise.",
    )
    parser.add_argument(
        "--rank-max-pairs-per-group",
        type=int,
        default=4096,
        help="Subsample rank pairs per group. Use 0 to keep all.",
    )
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

    train_cache = _load_or_build_cache(
        split="train",
        jsonl_path=args.train_jsonl,
        explicit_cache_path=args.train_cache,
        cache_dir=args.cache_dir,
        args=args,
    )
    val_cache = _load_or_build_cache(
        split="val",
        jsonl_path=args.val_jsonl,
        explicit_cache_path=args.val_cache,
        cache_dir=args.cache_dir,
        args=args,
    )
    _validate_cache_labels("train", train_cache)
    _validate_cache_labels("val", val_cache)
    print(f"train encoded rows: {len(train_cache['labels'])}")
    print(f"val encoded rows: {len(val_cache['labels'])}")

    result = train(args, train_cache, val_cache)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
