#!/usr/bin/env python3
"""
Train a token-level D3 crop/phrase verifier.

This verifier is intentionally outside the detector. It consumes candidate rows
from tools/export_d3_topk_candidate_pairs.py, crops the candidate boxes, extracts
frozen CLIP image/text token features, and trains a small cross-attention module
to classify whether a crop matches a D3 phrase.

Use the `lami` environment for this script because it depends on CLIP/open_clip.
Do not install these packages into the detector training environment.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter
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


def _save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)


def _torch_load(path: Path, map_location: str = "cpu") -> Any:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _negative_kind(row: Mapping[str, Any]) -> str:
    value = row.get("negative_type")
    return str(value) if value is not None else "positive"


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


def _sample_count(base_count: int, ratio: float, available: int) -> int:
    if ratio < 0:
        return available
    return min(available, int(round(base_count * ratio)))


def _sample_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    max_samples: Optional[int],
    max_positives: Optional[int],
    wrong_phrase_neg_per_pos: float,
    same_phrase_neg_per_pos: float,
    background_neg_per_pos: float,
    seed: int,
) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    positives = [dict(row) for row in rows if int(row.get("label", 0)) == 1]
    wrong = [dict(row) for row in rows if int(row.get("label", 0)) == 0 and _negative_kind(row) == "wrong_phrase_good_box"]
    same = [dict(row) for row in rows if int(row.get("label", 0)) == 0 and _negative_kind(row) == "same_phrase_bad_box"]
    background = [dict(row) for row in rows if int(row.get("label", 0)) == 0 and _negative_kind(row) == "background_bad_box"]
    other_neg = [
        dict(row)
        for row in rows
        if int(row.get("label", 0)) == 0
        and _negative_kind(row) not in {"wrong_phrase_good_box", "same_phrase_bad_box", "background_bad_box"}
    ]

    for bucket in (positives, wrong, same, background, other_neg):
        rng.shuffle(bucket)

    if max_positives is not None and max_positives > 0:
        positives = positives[:max_positives]

    pos_count = len(positives)
    selected = (
        positives
        + wrong[: _sample_count(pos_count, wrong_phrase_neg_per_pos, len(wrong))]
        + same[: _sample_count(pos_count, same_phrase_neg_per_pos, len(same))]
        + background[: _sample_count(pos_count, background_neg_per_pos, len(background))]
    )

    if not selected and other_neg:
        selected = other_neg[: max_samples or len(other_neg)]

    if max_samples is not None and max_samples > 0 and len(selected) > max_samples:
        selected_pos = [row for row in selected if int(row.get("label", 0)) == 1]
        selected_neg = [row for row in selected if int(row.get("label", 0)) == 0]
        if len(selected_pos) >= max_samples:
            selected = rng.sample(selected_pos, max_samples)
        else:
            selected = selected_pos + rng.sample(selected_neg, max_samples - len(selected_pos))

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


class D3CandidateCropDataset(Dataset):
    def __init__(
        self,
        rows: Sequence[Mapping[str, Any]],
        *,
        image_root: Path,
        prompt_template: str,
    ) -> None:
        self.rows = [dict(row) for row in rows]
        self.image_root = image_root
        self.prompt_template = prompt_template

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        row = self.rows[index]
        return {
            "image_id": int(row.get("image_id", -1)),
            "file_name": str(row["file_name"]),
            "bbox": [float(v) for v in row["bbox"]],
            "phrase": str(row["phrase"]),
            "text": self.prompt_template.format(phrase=str(row["phrase"])),
            "label": float(row.get("label", 0)),
            "detector_score": float(row.get("detector_score", row.get("score", 0.0))),
            "negative_type": _negative_kind(row),
        }


class ClipTokenEncoder(nn.Module):
    def __init__(
        self,
        *,
        backend: str,
        model_name: str,
        pretrained: str,
        openai_clip_model: str,
        device: str,
    ) -> None:
        super().__init__()
        self.backend = backend
        self.model_name = model_name
        self.pretrained = pretrained
        self.openai_clip_model = openai_clip_model
        self.device_name = device
        self.model, self.preprocess, self.tokenizer, self.backend_name = self._load_backend()
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad_(False)

    def _load_backend(self):
        errors = []
        if self.backend in {"auto", "open_clip"}:
            try:
                import open_clip

                model, _, preprocess = open_clip.create_model_and_transforms(
                    self.model_name,
                    pretrained=self.pretrained,
                    device=self.device_name,
                )
                return model, preprocess, open_clip.tokenize, f"open_clip:{self.model_name}:{self.pretrained}"
            except Exception as exc:
                errors.append(f"open_clip unavailable: {exc}")
                if self.backend == "open_clip":
                    raise

        if self.backend in {"auto", "openai_clip"}:
            try:
                import clip

                model, preprocess = clip.load(self.openai_clip_model, device=self.device_name)
                return model, preprocess, clip.tokenize, f"openai_clip:{self.openai_clip_model}"
            except Exception as exc:
                errors.append(f"openai clip unavailable: {exc}")
                if self.backend == "openai_clip":
                    raise

        raise ImportError("No CLIP backend is available:\n" + "\n".join(errors))

    @property
    def dtype(self) -> torch.dtype:
        dtype = getattr(self.model, "dtype", None)
        if dtype is not None:
            return dtype
        return next(self.model.parameters()).dtype

    def image_token_dim(self) -> int:
        dummy = torch.zeros(1, 3, 224, 224, device=self.device_name, dtype=self.dtype)
        with torch.no_grad():
            tokens = self.encode_image_tokens(dummy)
        return int(tokens.shape[-1])

    def text_token_dim(self) -> int:
        tokens = self.tokenizer(["dummy text"]).to(self.device_name)
        with torch.no_grad():
            text_tokens, _ = self.encode_text_tokens(tokens)
        return int(text_tokens.shape[-1])

    @torch.no_grad()
    def encode_image_tokens(self, images: torch.Tensor) -> torch.Tensor:
        images = images.to(self.device_name, dtype=self.dtype)
        if self.backend_name.startswith("open_clip"):
            visual = self.model.visual
            if not hasattr(visual, "output_tokens"):
                raise RuntimeError(
                    f"{self.backend_name} visual tower does not expose output_tokens. "
                    "Use a ViT open_clip model such as --model ViT-L-14."
                )
            old_output_tokens = bool(visual.output_tokens)
            visual.output_tokens = True
            output = visual(images)
            visual.output_tokens = old_output_tokens
            if not isinstance(output, tuple) or len(output) != 2:
                raise RuntimeError(f"{self.backend_name} did not return image tokens.")
            _, tokens = output
            return tokens.float()

        visual = self.model.visual
        x = visual.conv1(images)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)
        cls = visual.class_embedding.to(x.dtype)
        cls = cls + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device)
        x = torch.cat([cls, x], dim=1)
        x = x + visual.positional_embedding.to(x.dtype)
        x = visual.ln_pre(x)
        x = x.permute(1, 0, 2)
        x = visual.transformer(x)
        x = x.permute(1, 0, 2)
        return x.float()

    @torch.no_grad()
    def encode_text_tokens(self, token_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        token_ids = token_ids.to(self.device_name)
        if self.backend_name.startswith("open_clip"):
            if hasattr(self.model, "text") and hasattr(self.model.text, "output_tokens"):
                old_output_tokens = bool(self.model.text.output_tokens)
                self.model.text.output_tokens = True
                output = self.model.text(token_ids)
                self.model.text.output_tokens = old_output_tokens
                if not isinstance(output, tuple) or len(output) != 2:
                    raise RuntimeError(f"{self.backend_name} did not return text tokens.")
                _, tokens = output
                return tokens.float(), token_ids != 0

            cast_dtype = self.model.transformer.get_cast_dtype()
            x = self.model.token_embedding(token_ids).to(cast_dtype)
            x = x + self.model.positional_embedding.to(cast_dtype)
            x = self.model.transformer(x, attn_mask=self.model.attn_mask)
            x = self.model.ln_final(x)
            return x.float(), token_ids != 0

        x = self.model.token_embedding(token_ids).type(self.dtype)
        x = x + self.model.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)
        x = self.model.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.model.ln_final(x).float()
        return x, token_ids != 0


class TokenCrossVerifier(nn.Module):
    def __init__(
        self,
        *,
        image_dim: int,
        text_dim: int,
        hidden_dim: int = 256,
        num_heads: int = 4,
        dropout: float = 0.1,
        use_detector_score: bool = True,
    ) -> None:
        super().__init__()
        self.image_dim = int(image_dim)
        self.text_dim = int(text_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_heads = int(num_heads)
        self.dropout = float(dropout)
        self.use_detector_score = bool(use_detector_score)

        self.image_proj = nn.Linear(image_dim, hidden_dim)
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        self.cross_attn = nn.MultiheadAttention(
            hidden_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout_layer = nn.Dropout(dropout)
        input_dim = hidden_dim * 3 + (1 if use_detector_score else 0)
        self.head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, max(64, hidden_dim // 2)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(max(64, hidden_dim // 2), 1),
        )

    def forward(
        self,
        image_tokens: torch.Tensor,
        text_tokens: torch.Tensor,
        text_mask: torch.Tensor,
        detector_scores: torch.Tensor,
    ) -> torch.Tensor:
        image_hidden = self.image_proj(image_tokens.float())
        text_hidden = self.text_proj(text_tokens.float())
        attn_output, _ = self.cross_attn(
            query=text_hidden,
            key=image_hidden,
            value=image_hidden,
            need_weights=False,
        )
        fused_text = self.norm(text_hidden + self.dropout_layer(attn_output))

        mask = text_mask.to(device=fused_text.device, dtype=fused_text.dtype).unsqueeze(-1)
        denom = mask.sum(dim=1).clamp_min(1.0)
        cross_pool = (fused_text * mask).sum(dim=1) / denom
        text_pool = (text_hidden * mask).sum(dim=1) / denom
        image_pool = image_hidden.mean(dim=1)
        pieces = [cross_pool, text_pool, image_pool]
        if self.use_detector_score:
            pieces.append(detector_scores.float().view(-1, 1))
        features = torch.cat(pieces, dim=-1)
        return self.head(features).squeeze(-1)


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


def _binary_metrics(scores: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    labels = labels.astype(np.int64)
    predictions = (scores >= 0.0).astype(np.int64)
    return {
        "n": int(len(labels)),
        "positive": int(labels.sum()),
        "negative": int(len(labels) - labels.sum()),
        "positive_rate": float(labels.mean()) if len(labels) else float("nan"),
        "auc": _roc_auc(scores, labels),
        "ap": _average_precision(scores, labels),
        "accuracy_at_0_5": float((predictions == labels).mean()) if len(labels) else float("nan"),
    }


def _collate_rows(
    batch: Sequence[Mapping[str, Any]],
    *,
    image_root: Path,
    preprocess,
    tokenizer,
    crop_margin: float,
) -> Dict[str, Any]:
    images = []
    texts = []
    labels = []
    detector_scores = []
    negative_types = []
    kept_rows = []
    invalid_crops = 0
    missing_images = 0
    for row in batch:
        image_path = _resolve_image_path(image_root, str(row["file_name"]))
        if not image_path.exists():
            missing_images += 1
            continue
        image = Image.open(image_path).convert("RGB")
        width, height = image.size
        xyxy = _expanded_xyxy(row["bbox"], width=width, height=height, margin=crop_margin)
        if xyxy is None:
            invalid_crops += 1
            continue
        images.append(preprocess(image.crop(xyxy)))
        texts.append(str(row["text"]))
        labels.append(float(row["label"]))
        detector_scores.append(float(row["detector_score"]))
        negative_types.append(str(row["negative_type"]))
        kept_rows.append(row)

    if images:
        image_tensor = torch.stack(images, dim=0)
        text_tokens = tokenizer(texts)
        label_tensor = torch.tensor(labels, dtype=torch.float32)
        score_tensor = torch.tensor(detector_scores, dtype=torch.float32)
    else:
        image_tensor = torch.empty(0)
        text_tokens = torch.empty(0, dtype=torch.long)
        label_tensor = torch.empty(0, dtype=torch.float32)
        score_tensor = torch.empty(0, dtype=torch.float32)

    return {
        "images": image_tensor,
        "text_tokens": text_tokens,
        "labels": label_tensor,
        "detector_scores": score_tensor,
        "negative_types": negative_types,
        "rows": kept_rows,
        "invalid_crops": invalid_crops,
        "missing_images": missing_images,
    }


def _run_model_batch(
    *,
    clip_encoder: ClipTokenEncoder,
    verifier: TokenCrossVerifier,
    batch: Mapping[str, Any],
    device: str,
) -> Optional[torch.Tensor]:
    if batch["labels"].numel() == 0:
        return None
    with torch.no_grad():
        image_tokens = clip_encoder.encode_image_tokens(batch["images"])
        text_tokens, text_mask = clip_encoder.encode_text_tokens(batch["text_tokens"])
    logits = verifier(
        image_tokens.to(device),
        text_tokens.to(device),
        text_mask.to(device),
        batch["detector_scores"].to(device),
    )
    return logits


def evaluate(
    *,
    verifier: TokenCrossVerifier,
    clip_encoder: ClipTokenEncoder,
    loader: DataLoader,
    device: str,
) -> Dict[str, Any]:
    verifier.eval()
    logits_all = []
    labels_all = []
    detector_scores_all = []
    invalid_crops = 0
    missing_images = 0
    with torch.no_grad():
        for batch in tqdm(loader, desc="validating"):
            invalid_crops += int(batch["invalid_crops"])
            missing_images += int(batch["missing_images"])
            logits = _run_model_batch(
                clip_encoder=clip_encoder,
                verifier=verifier,
                batch=batch,
                device=device,
            )
            if logits is None:
                continue
            logits_all.append(logits.detach().cpu())
            labels_all.append(batch["labels"].detach().cpu())
            detector_scores_all.append(batch["detector_scores"].detach().cpu())

    if not logits_all:
        raise RuntimeError("No validation samples were encoded.")

    logits_np = torch.cat(logits_all).numpy()
    labels_np = torch.cat(labels_all).numpy()
    detector_np = torch.cat(detector_scores_all).numpy()
    return {
        "token_verifier": _binary_metrics(logits_np, labels_np),
        "detector_score": _binary_metrics(detector_np, labels_np),
        "invalid_crops": invalid_crops,
        "missing_images": missing_images,
    }


def _save_checkpoint(
    path: Path,
    *,
    verifier: TokenCrossVerifier,
    clip_encoder: ClipTokenEncoder,
    args: argparse.Namespace,
    epoch: int,
    score: float,
    history: Sequence[Mapping[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_state": verifier.state_dict(),
        "image_dim": verifier.image_dim,
        "text_dim": verifier.text_dim,
        "hidden_dim": verifier.hidden_dim,
        "num_heads": verifier.num_heads,
        "dropout": verifier.dropout,
        "use_detector_score": verifier.use_detector_score,
        "clip_backend": args.clip_backend,
        "clip_model": args.clip_model,
        "clip_pretrained": args.clip_pretrained,
        "openai_clip_model": args.openai_clip_model,
        "prompt_template": args.prompt_template,
        "crop_margin": args.crop_margin,
        "epoch": epoch,
        "score": score,
        "history": list(history),
        "args": _jsonable_args(args),
        "feature_source": "clip_crop_phrase_tokens",
        "backend_name": clip_encoder.backend_name,
    }
    torch.save(checkpoint, path)
    print(f"saved checkpoint: {path}")


def train(args: argparse.Namespace) -> Dict[str, Any]:
    train_rows = _read_jsonl(args.train_jsonl)
    val_rows = _read_jsonl(args.val_jsonl)
    train_rows = _sample_rows(
        train_rows,
        max_samples=args.max_train_samples,
        max_positives=args.max_train_positives,
        wrong_phrase_neg_per_pos=args.wrong_phrase_neg_per_pos,
        same_phrase_neg_per_pos=args.same_phrase_neg_per_pos,
        background_neg_per_pos=args.background_neg_per_pos,
        seed=args.seed,
    )
    val_rows = _sample_rows(
        val_rows,
        max_samples=args.max_val_samples,
        max_positives=args.max_val_positives,
        wrong_phrase_neg_per_pos=args.wrong_phrase_neg_per_pos,
        same_phrase_neg_per_pos=args.same_phrase_neg_per_pos,
        background_neg_per_pos=args.background_neg_per_pos,
        seed=args.seed + 1,
    )
    train_summary = _summarize_rows("selected train", train_rows)
    val_summary = _summarize_rows("selected val", val_rows)

    device = args.device
    clip_encoder = ClipTokenEncoder(
        backend=args.clip_backend,
        model_name=args.clip_model,
        pretrained=args.clip_pretrained,
        openai_clip_model=args.openai_clip_model,
        device=device,
    )
    print(f"loaded token backend: {clip_encoder.backend_name}")

    image_dim = clip_encoder.image_token_dim()
    text_dim = clip_encoder.text_token_dim()
    print(f"token dims: image={image_dim}, text={text_dim}")

    verifier = TokenCrossVerifier(
        image_dim=image_dim,
        text_dim=text_dim,
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        dropout=args.dropout,
        use_detector_score=not args.no_detector_score,
    ).to(device)

    train_dataset = D3CandidateCropDataset(train_rows, image_root=args.image_root, prompt_template=args.prompt_template)
    val_dataset = D3CandidateCropDataset(val_rows, image_root=args.image_root, prompt_template=args.prompt_template)

    collate = lambda batch: _collate_rows(  # noqa: E731
        batch,
        image_root=args.image_root,
        preprocess=clip_encoder.preprocess,
        tokenizer=clip_encoder.tokenizer,
        crop_margin=args.crop_margin,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate,
        pin_memory=True,
    )

    pos_count = max(1, train_summary["positive"])
    neg_count = max(1, train_summary["negative"])
    pos_weight = torch.tensor([min(args.max_pos_weight, neg_count / pos_count)], dtype=torch.float32, device=device)
    optimizer = torch.optim.AdamW(verifier.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    history = []
    best_score = -float("inf")
    best_epoch = 0

    for epoch in range(1, args.epochs + 1):
        verifier.train()
        losses = []
        invalid_crops = 0
        missing_images = 0
        for batch in tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}"):
            invalid_crops += int(batch["invalid_crops"])
            missing_images += int(batch["missing_images"])
            optimizer.zero_grad(set_to_none=True)
            logits = _run_model_batch(
                clip_encoder=clip_encoder,
                verifier=verifier,
                batch=batch,
                device=device,
            )
            if logits is None:
                continue
            labels = batch["labels"].to(device)
            loss = F.binary_cross_entropy_with_logits(logits, labels, pos_weight=pos_weight)
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(verifier.parameters(), args.grad_clip)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        val_metrics = evaluate(verifier=verifier, clip_encoder=clip_encoder, loader=val_loader, device=device)
        score = float(val_metrics["token_verifier"]["ap"])
        record = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)) if losses else float("nan"),
            "train_invalid_crops": invalid_crops,
            "train_missing_images": missing_images,
            "val": val_metrics,
        }
        history.append(record)
        print(json.dumps(record, sort_keys=True))
        if score > best_score:
            best_score = score
            best_epoch = epoch
            _save_checkpoint(
                args.output_dir / "verifier_best.pt",
                verifier=verifier,
                clip_encoder=clip_encoder,
                args=args,
                epoch=epoch,
                score=score,
                history=history,
            )

    _save_checkpoint(
        args.output_dir / "verifier_final.pt",
        verifier=verifier,
        clip_encoder=clip_encoder,
        args=args,
        epoch=args.epochs,
        score=float(history[-1]["val"]["token_verifier"]["ap"]) if history else float("nan"),
        history=history,
    )
    summary = {
        "best_epoch": best_epoch,
        "best_score": best_score,
        "train_rows": train_summary,
        "val_rows": val_summary,
        "history": history,
    }
    _save_json(args.output_dir / "metrics.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-jsonl", type=Path, required=True)
    parser.add_argument("--val-jsonl", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, default=Path("dataset/d3/images"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prompt-template", default="the described target is {phrase}")
    parser.add_argument("--clip-backend", choices=("auto", "open_clip", "openai_clip"), default="open_clip")
    parser.add_argument("--clip-model", default="ViT-L-14", help="OpenCLIP ViT model name.")
    parser.add_argument("--clip-pretrained", default="openai", help="OpenCLIP pretrained tag.")
    parser.add_argument("--openai-clip-model", default="ViT-L/14")
    parser.add_argument("--crop-margin", type=float, default=0.25)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--no-detector-score", action="store_true")
    parser.add_argument("--max-train-samples", type=int, default=20000)
    parser.add_argument("--max-val-samples", type=int, default=5000)
    parser.add_argument("--max-train-positives", type=int, default=0)
    parser.add_argument("--max-val-positives", type=int, default=0)
    parser.add_argument("--wrong-phrase-neg-per-pos", type=float, default=4.0)
    parser.add_argument("--same-phrase-neg-per-pos", type=float, default=2.0)
    parser.add_argument("--background-neg-per-pos", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-pos-weight", type=float, default=20.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    train(parse_args())


if __name__ == "__main__":
    main()
