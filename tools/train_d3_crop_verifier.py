#!/usr/bin/env python3
"""
Train a lightweight D3 image-region-description verifier.

This is intentionally kept outside the detector. It answers one question first:
given a detector box region and a target phrase, can a small verifier
distinguish matching region-description pairs from hard negatives?

Inputs are JSONL files produced by tools/build_d3_verifier_pairs.py.
The script can either encode crop/text features into a cache or train directly
from precomputed cache files.
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
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader, Dataset, Sampler
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


def _draw_boxed_image(
    image: Image.Image,
    xyxy: Tuple[int, int, int, int],
    *,
    line_width: int,
) -> Image.Image:
    boxed = image.copy()
    draw = ImageDraw.Draw(boxed)
    x0, y0, x1, y1 = xyxy
    width = max(2, int(line_width))
    for offset in range(width):
        draw.rectangle((x0 - offset, y0 - offset, x1 + offset, y1 + offset), outline=(255, 0, 0), width=1)
    return boxed


def _make_region_image(
    image: Image.Image,
    bbox_xywh: Sequence[float],
    *,
    image_mode: str,
    crop_margin: float,
    box_line_width: int,
) -> Optional[Image.Image]:
    width, height = image.size
    tight_xyxy = _expanded_xyxy(bbox_xywh, width=width, height=height, margin=0.0)
    if tight_xyxy is None:
        return None
    if image_mode == "boxed":
        return _draw_boxed_image(image, tight_xyxy, line_width=box_line_width)
    if image_mode == "crop":
        crop_xyxy = _expanded_xyxy(bbox_xywh, width=width, height=height, margin=crop_margin)
        return image.crop(crop_xyxy) if crop_xyxy is not None else None
    raise ValueError(f"Unsupported image mode: {image_mode}")


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


class _SiglipAdapter:
    def __init__(self, model) -> None:
        self.model = model

    def eval(self):
        self.model.eval()
        return self

    def encode_text(self, tokens) -> torch.Tensor:
        return self.model.get_text_features(**tokens)

    def encode_image(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.model.get_image_features(pixel_values=pixel_values)


def _load_siglip(model_name: str, _pretrained: str, device: str):
    try:
        from transformers import AutoModel, AutoProcessor
    except ImportError as exc:
        raise ImportError("transformers is required for --encoder-backend siglip.") from exc

    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()
    adapter = _SiglipAdapter(model)

    def preprocess(image: Image.Image) -> torch.Tensor:
        return processor(images=image, return_tensors="pt")["pixel_values"].squeeze(0)

    def tokenize(texts: Sequence[str]):
        return processor(
            text=list(texts),
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

    return adapter, preprocess, tokenize


def _load_encoder(backend: str, model_name: str, pretrained: str, device: str):
    if backend == "open_clip":
        model, preprocess, tokenizer = _load_openclip(model_name, pretrained, device)
    elif backend == "siglip":
        model, preprocess, tokenizer = _load_siglip(model_name, pretrained, device)
    else:
        raise ValueError(f"Unsupported encoder backend: {backend}")
    setattr(model, "backend_name", backend)
    setattr(model, "model_name", model_name)
    setattr(model, "pretrained_name", pretrained)
    return model, preprocess, tokenizer


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
    image_mode: str,
    crop_margin: float,
    box_line_width: int,
    prompt_template: str,
    target_field: str,
    weight_field: Optional[str],
    positive_threshold: float,
    positive_weight: float,
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
    targets: List[float] = []
    weights: List[float] = []
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
            label = float(row["label"])
            target = float(row.get(target_field, row["label"]))
            weight = float(row.get(weight_field, 1.0)) if weight_field else 1.0
            if target >= positive_threshold:
                weight *= positive_weight
            labels.append(label)
            targets.append(target)
            weights.append(weight)
            detector_scores.append(float(row.get("detector_score", 0.0)))
            negative_types.append(_negative_kind(row))
            target_category_ids.append(int(row.get("target_category_id", -1)))
            image_ids.append(int(row.get("image_id", -1)))
        pending_crops.clear()
        pending_rows.clear()

    for file_name, file_rows in tqdm(rows_by_file.items(), desc="encoding region images"):
        image_path = _resolve_image_path(image_root, file_name)
        if not image_path.exists():
            missing_images += len(file_rows)
            continue

        image = Image.open(image_path).convert("RGB")
        for row in file_rows:
            region_image = _make_region_image(
                image,
                row["bbox"],
                image_mode=image_mode,
                crop_margin=crop_margin,
                box_line_width=box_line_width,
            )
            if region_image is None:
                invalid_crops += 1
                continue
            pending_crops.append(preprocess(region_image))
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
        "targets": torch.tensor(targets, dtype=torch.float32),
        "weights": torch.tensor(weights, dtype=torch.float32),
        "detector_scores": torch.tensor(detector_scores, dtype=torch.float32),
        "negative_types": negative_types,
        "target_category_ids": torch.tensor(target_category_ids, dtype=torch.long),
        "image_ids": torch.tensor(image_ids, dtype=torch.long),
        "meta": {
            "num_input_rows": len(rows),
            "num_encoded_rows": len(labels),
            "missing_images": missing_images,
            "invalid_crops": invalid_crops,
            "encoder_backend": str(getattr(model, "backend_name", "")),
            "encoder_model": str(getattr(model, "model_name", "")),
            "encoder_pretrained": str(getattr(model, "pretrained_name", "")),
            "image_mode": image_mode,
            "crop_margin": crop_margin,
            "box_line_width": int(box_line_width),
            "prompt_template": prompt_template,
            "target_field": target_field,
            "weight_field": weight_field,
            "positive_threshold": float(positive_threshold),
            "positive_weight": float(positive_weight),
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
        raise ValueError("Feature encoder model/preprocess/tokenizer are required to build feature caches.")

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
        image_mode=args.image_mode,
        crop_margin=args.crop_margin,
        box_line_width=args.box_line_width,
        prompt_template=args.prompt_template,
        target_field=args.target_field,
        weight_field=args.weight_field,
        positive_threshold=args.positive_threshold,
        positive_weight=args.positive_weight,
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
    def __init__(self, cache: Mapping[str, Any], *, feature_mode: str) -> None:
        self.crop_feats = cache["crop_feats"].float()
        self.text_feats = cache["text_feats"].float()
        self.labels = cache["labels"].float()
        self.targets = cache.get("targets", self.labels).float()
        self.weights = cache.get("weights", torch.ones_like(self.labels)).float()
        self.detector_scores = cache["detector_scores"].float()
        self.image_ids = cache.get("image_ids", torch.arange(self.labels.numel())).long()
        self.feature_mode = feature_mode
        if self.crop_feats.shape != self.text_feats.shape:
            raise ValueError(
                f"crop/text feature shape mismatch: {self.crop_feats.shape} vs {self.text_feats.shape}"
            )

    def __len__(self) -> int:
        return int(self.labels.numel())

    @property
    def input_dim(self) -> int:
        return int(_build_pair_features(
            self.crop_feats[:1],
            self.text_feats[:1],
            self.detector_scores[:1],
            self.feature_mode,
        ).shape[1])

    def __getitem__(
        self,
        index: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        crop = self.crop_feats[index]
        text = self.text_feats[index]
        score = self.detector_scores[index : index + 1]
        features = _build_pair_features(
            crop.view(1, -1),
            text.view(1, -1),
            score,
            self.feature_mode,
        ).squeeze(0)
        return (
            features,
            self.targets[index],
            self.labels[index],
            self.weights[index],
            self.image_ids[index],
            self.detector_scores[index],
        )


class GroupedFeatureBatchSampler(Sampler[List[int]]):
    def __init__(self, image_ids: torch.Tensor, *, batch_size: int, seed: int, shuffle: bool = True) -> None:
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.shuffle = bool(shuffle)
        self.epoch = 0
        groups: Dict[int, List[int]] = defaultdict(list)
        for index, image_id in enumerate(image_ids.tolist()):
            groups[int(image_id)].append(index)
        self.groups = list(groups.values())
        self._length = self._estimate_length()

    def _estimate_length(self) -> int:
        length = 0
        current = 0
        for group in self.groups:
            group_size = len(group)
            if current and current + group_size > self.batch_size:
                length += 1
                current = 0
            current += group_size
        if current:
            length += 1
        return max(1, length)

    def __len__(self) -> int:
        return self._length

    def __iter__(self) -> Iterable[List[int]]:
        rng = random.Random(self.seed + self.epoch)
        self.epoch += 1
        groups = [list(group) for group in self.groups]
        if self.shuffle:
            rng.shuffle(groups)
            for group in groups:
                rng.shuffle(group)

        batch: List[int] = []
        for group in groups:
            if batch and len(batch) + len(group) > self.batch_size:
                yield batch
                batch = []
            batch.extend(group)
        if batch:
            yield batch


def _build_pair_features(
    crop_feats: torch.Tensor,
    text_feats: torch.Tensor,
    detector_scores: torch.Tensor,
    feature_mode: str,
) -> torch.Tensor:
    score = detector_scores.to(dtype=crop_feats.dtype).view(-1, 1)
    if feature_mode == "full":
        return torch.cat(
            [
                crop_feats,
                text_feats,
                crop_feats * text_feats,
                torch.abs(crop_feats - text_feats),
                score,
            ],
            dim=-1,
        )
    if feature_mode == "no_text":
        return torch.cat([crop_feats, score], dim=-1)
    if feature_mode == "no_detector_score":
        return torch.cat(
            [
                crop_feats,
                text_feats,
                crop_feats * text_feats,
                torch.abs(crop_feats - text_feats),
            ],
            dim=-1,
        )
    raise ValueError(f"Unsupported feature mode: {feature_mode}")


def _validate_cache_labels(name: str, cache: Mapping[str, Any]) -> None:
    labels = cache["labels"].float()
    positives = int(labels.sum().item())
    negatives = int(labels.numel()) - positives
    if positives == 0 or negatives == 0:
        raise ValueError(
            f"{name} cache has {positives} positives and {negatives} negatives. "
            "The verifier needs both classes; remove stale caches or rerun with --overwrite-cache."
        )


def _infer_clip_feature_dim(input_dim: int, feature_mode: str) -> Tuple[int, bool]:
    if feature_mode == "full":
        if (input_dim - 1) % 4 != 0:
            raise ValueError(f"Cannot infer CLIP feature dim from full input_dim={input_dim}.")
        return (input_dim - 1) // 4, True
    if feature_mode == "no_detector_score":
        if input_dim % 4 != 0:
            raise ValueError(f"Cannot infer CLIP feature dim from no_detector_score input_dim={input_dim}.")
        return input_dim // 4, False
    raise ValueError(f"gated_bilinear verifier requires text features; got feature_mode={feature_mode}.")


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


class GatedBilinearDescriptionVerifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        dropout: float,
        *,
        feature_mode: str,
        projection_dim: int,
    ) -> None:
        super().__init__()
        clip_dim, has_detector_score = _infer_clip_feature_dim(input_dim, feature_mode)
        self.clip_dim = clip_dim
        self.has_detector_score = has_detector_score
        self.feature_mode = feature_mode

        projection_dim = int(projection_dim)
        if projection_dim <= 0:
            raise ValueError("--projection-dim must be positive for gated_bilinear.")

        self.image_norm = nn.LayerNorm(clip_dim)
        self.text_norm = nn.LayerNorm(clip_dim)
        self.pair_norm = nn.LayerNorm(input_dim)
        self.image_proj = nn.Linear(clip_dim, projection_dim)
        self.text_proj = nn.Linear(clip_dim, projection_dim)
        self.pair_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.interaction_gate = nn.Sequential(
            nn.Linear(hidden_dim, projection_dim),
            nn.Sigmoid(),
        )

        score_dim = 2 if has_detector_score else 0
        head_dim = hidden_dim + projection_dim * 4 + 1 + score_dim
        second_hidden = max(64, hidden_dim // 2)
        self.head = nn.Sequential(
            nn.LayerNorm(head_dim),
            nn.Linear(head_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, second_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(second_hidden, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        image = features[:, : self.clip_dim]
        text = features[:, self.clip_dim : self.clip_dim * 2]
        image_proj = self.image_proj(self.image_norm(image))
        text_proj = self.text_proj(self.text_norm(text))
        pair_proj = self.pair_proj(self.pair_norm(features))
        gate = self.interaction_gate(pair_proj)
        interaction = image_proj * text_proj * gate
        cosine = (image * text).sum(dim=-1, keepdim=True)

        parts = [
            pair_proj,
            image_proj,
            text_proj,
            interaction,
            torch.abs(image_proj - text_proj),
            cosine,
        ]
        if self.has_detector_score:
            score = features[:, -1:].clamp(1e-6, 1.0 - 1e-6)
            parts.extend([score, torch.logit(score)])
        return self.head(torch.cat(parts, dim=-1)).squeeze(-1)


def _build_verifier_model(
    *,
    arch: str,
    input_dim: int,
    hidden_dim: int,
    dropout: float,
    feature_mode: str,
    projection_dim: int,
) -> nn.Module:
    if arch == "mlp":
        return CropDescriptionVerifier(input_dim=input_dim, hidden_dim=hidden_dim, dropout=dropout)
    if arch == "gated_bilinear":
        return GatedBilinearDescriptionVerifier(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            feature_mode=feature_mode,
            projection_dim=projection_dim,
        )
    raise ValueError(f"Unsupported verifier architecture: {arch}")


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


def _is_valid_selection_metric(metrics: Mapping[str, Any]) -> bool:
    return int(metrics.get("positive", 0)) > 0 and int(metrics.get("negative", 0)) > 0


def _select_checkpoint_score(val_metrics: Mapping[str, Any]) -> Tuple[str, float]:
    for name in ("wrong_phrase_same_region", "same_phrase_bad_box", "overall"):
        metrics = val_metrics[name]
        if _is_valid_selection_metric(metrics):
            score = float(metrics["ap"])
            if not math.isnan(score):
                return f"{name}_ap", score
    return "overall_ap", float("-inf")


def _weighted_bce_loss(logits: torch.Tensor, targets: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    weights = weights.to(dtype=loss.dtype)
    return (loss * weights).sum() / weights.sum().clamp_min(1e-6)


def _safe_logit(scores: torch.Tensor) -> torch.Tensor:
    scores = scores.clamp(1e-6, 1.0 - 1e-6)
    return torch.log(scores / (1.0 - scores))


def _teacher_scores(
    detector_scores: torch.Tensor,
    targets: torch.Tensor,
    *,
    mode: str,
    fusion_weight: float,
) -> torch.Tensor:
    detector_scores = detector_scores.to(dtype=targets.dtype).clamp(1e-6, 1.0 - 1e-6)
    targets = targets.clamp(1e-6, 1.0 - 1e-6)
    if mode == "logit_add":
        return torch.sigmoid(_safe_logit(detector_scores) + float(fusion_weight) * _safe_logit(targets))
    if mode == "linear":
        return (1.0 - float(fusion_weight)) * detector_scores + float(fusion_weight) * targets
    if mode == "replace":
        return targets
    raise ValueError(f"Unsupported listwise teacher mode: {mode}")


def _listwise_distillation_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    detector_scores: torch.Tensor,
    image_ids: torch.Tensor,
    *,
    teacher_mode: str,
    teacher_fusion_weight: float,
    topk: int,
    temperature: float,
) -> torch.Tensor:
    teacher = _teacher_scores(
        detector_scores,
        targets,
        mode=teacher_mode,
        fusion_weight=teacher_fusion_weight,
    )
    temperature = max(float(temperature), 1e-6)
    losses = []
    for image_id in torch.unique(image_ids):
        group = torch.nonzero(image_ids == image_id, as_tuple=False).flatten()
        if group.numel() < 2:
            continue
        teacher_group = teacher[group]
        student_log_distribution = F.log_softmax(logits[group] / temperature, dim=0)
        if topk > 0 and group.numel() > topk:
            top = torch.topk(teacher_group, k=int(topk), largest=True).indices
            teacher_distribution = F.softmax(_safe_logit(teacher_group[top]) / temperature, dim=0)
            losses.append(-(teacher_distribution.detach() * student_log_distribution[top]).sum())
        else:
            teacher_distribution = F.softmax(_safe_logit(teacher_group) / temperature, dim=0)
            losses.append(-(teacher_distribution.detach() * student_log_distribution).sum())
    if not losses:
        return logits.new_zeros(())
    return torch.stack(losses).mean()


def _pairwise_ranking_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    positive_threshold: float,
    negative_threshold: float,
    margin: float,
    max_items_per_side: int,
) -> torch.Tensor:
    pos_idx = torch.nonzero(targets >= positive_threshold, as_tuple=False).flatten()
    neg_idx = torch.nonzero(targets <= negative_threshold, as_tuple=False).flatten()
    if pos_idx.numel() == 0 or neg_idx.numel() == 0:
        return logits.new_zeros(())

    if max_items_per_side > 0:
        if pos_idx.numel() > max_items_per_side:
            order = torch.randperm(pos_idx.numel(), device=pos_idx.device)[:max_items_per_side]
            pos_idx = pos_idx[order]
        if neg_idx.numel() > max_items_per_side:
            order = torch.randperm(neg_idx.numel(), device=neg_idx.device)[:max_items_per_side]
            neg_idx = neg_idx[order]

    margin_tensor = logits.new_tensor(float(margin))
    pair_diffs = logits[pos_idx].view(-1, 1) - logits[neg_idx].view(1, -1)
    return F.relu(margin_tensor - pair_diffs).mean()


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

    for features, targets, labels, weights, _image_ids, _detector_scores in loader:
        features = features.to(device)
        targets = targets.to(device)
        labels = labels.to(device)
        weights = weights.to(device)
        logits = model(features)
        loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        total_loss += float((loss * weights).sum().item())
        total_count += float(weights.sum().item())
        logits_chunks.append(logits.cpu())
        label_chunks.append(labels.cpu())

    logits_np = torch.cat(logits_chunks).numpy()
    labels_np = torch.cat(label_chunks).numpy()
    negative_types = list(cache["negative_types"])
    metrics = {
        "loss": total_loss / max(1e-6, total_count),
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

    train_dataset = PairFeatureDataset(train_cache, feature_mode=args.feature_mode)
    val_dataset = PairFeatureDataset(val_cache, feature_mode=args.feature_mode)
    model = _build_verifier_model(
        arch=args.verifier_arch,
        input_dim=train_dataset.input_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        feature_mode=args.feature_mode,
        projection_dim=args.projection_dim,
    ).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    uses_listwise = args.loss_type in {"listwise", "bce_listwise", "bce_rank_listwise"}
    if uses_listwise:
        loader = DataLoader(
            train_dataset,
            batch_sampler=GroupedFeatureBatchSampler(
                train_dataset.image_ids,
                batch_size=args.batch_size,
                seed=args.seed,
                shuffle=True,
            ),
            num_workers=args.num_workers,
        )
    else:
        loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            drop_last=False,
        )

    history: List[Dict[str, Any]] = []
    best_score = -float("inf")
    best_metric = ""
    best_epoch = -1
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_meta = train_cache.get("meta", {})
    encoder_backend = str(train_meta.get("encoder_backend") or args.encoder_backend)
    encoder_model = str(train_meta.get("encoder_model") or args.model)
    encoder_pretrained = str(train_meta.get("encoder_pretrained") or args.pretrained)

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        train_bce_loss = 0.0
        train_rank_loss = 0.0
        train_listwise_loss = 0.0
        train_count = 0
        for features, targets, _labels, weights, image_ids, detector_scores in tqdm(
            loader,
            desc=f"epoch {epoch}/{args.epochs}",
        ):
            features = features.to(args.device)
            targets = targets.to(args.device)
            weights = weights.to(args.device)
            image_ids = image_ids.to(args.device)
            detector_scores = detector_scores.to(args.device)
            logits = model(features)
            bce_loss = _weighted_bce_loss(logits, targets, weights)
            rank_loss = logits.new_zeros(())
            listwise_loss = logits.new_zeros(())
            if args.loss_type in {"bce_rank", "bce_rank_listwise"} and args.ranking_weight > 0:
                rank_loss = _pairwise_ranking_loss(
                    logits,
                    targets,
                    positive_threshold=args.ranking_positive_threshold,
                    negative_threshold=args.ranking_negative_threshold,
                    margin=args.ranking_margin,
                    max_items_per_side=args.ranking_max_items_per_side,
                )
            if uses_listwise and args.listwise_weight > 0:
                listwise_loss = _listwise_distillation_loss(
                    logits,
                    targets,
                    detector_scores,
                    image_ids,
                    teacher_mode=args.listwise_teacher_mode,
                    teacher_fusion_weight=args.listwise_teacher_fusion_weight,
                    topk=args.listwise_topk,
                    temperature=args.listwise_temperature,
                )

            if args.loss_type == "listwise":
                loss = listwise_loss
            elif args.loss_type == "bce":
                loss = bce_loss
            elif args.loss_type == "bce_rank":
                loss = bce_loss + args.ranking_weight * rank_loss
            elif args.loss_type == "bce_listwise":
                loss = bce_loss + args.listwise_weight * listwise_loss
            elif args.loss_type == "bce_rank_listwise":
                loss = bce_loss + args.ranking_weight * rank_loss + args.listwise_weight * listwise_loss
            else:
                raise ValueError(f"Unsupported loss_type={args.loss_type!r}.")

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            batch_weight = float(weights.sum().item())
            train_loss += float(loss.item()) * batch_weight
            train_bce_loss += float(bce_loss.item()) * batch_weight
            train_rank_loss += float(rank_loss.item()) * batch_weight
            train_listwise_loss += float(listwise_loss.item()) * batch_weight
            train_count += batch_weight

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
            "train_loss": train_loss / max(1e-6, train_count),
            "train_bce_loss": train_bce_loss / max(1e-6, train_count),
            "train_rank_loss": train_rank_loss / max(1e-6, train_count),
            "train_listwise_loss": train_listwise_loss / max(1e-6, train_count),
            "val": val_metrics,
        }
        history.append(epoch_record)

        score_metric, score = _select_checkpoint_score(val_metrics)
        if score > best_score:
            best_score = score
            best_metric = score_metric
            best_epoch = epoch
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "input_dim": train_dataset.input_dim,
                    "hidden_dim": args.hidden_dim,
                    "dropout": args.dropout,
                    "verifier_arch": args.verifier_arch,
                    "projection_dim": args.projection_dim,
                    "feature_mode": args.feature_mode,
                    "encoder_backend": encoder_backend,
                    "encoder_model": encoder_model,
                    "encoder_pretrained": encoder_pretrained,
                    "target_field": args.target_field,
                    "weight_field": args.weight_field,
                    "image_mode": args.image_mode,
                    "loss_type": args.loss_type,
                    "score_metric": score_metric,
                    "listwise_weight": args.listwise_weight,
                    "listwise_topk": args.listwise_topk,
                    "listwise_temperature": args.listwise_temperature,
                    "listwise_teacher_mode": args.listwise_teacher_mode,
                    "listwise_teacher_fusion_weight": args.listwise_teacher_fusion_weight,
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
                    "train_bce_loss": epoch_record["train_bce_loss"],
                    "train_rank_loss": epoch_record["train_rank_loss"],
                    "train_listwise_loss": epoch_record["train_listwise_loss"],
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
                    "best_metric": best_metric,
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
        "best_metric": best_metric,
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
    parser.add_argument(
        "--encoder-backend",
        choices=("open_clip", "siglip"),
        default="open_clip",
        help="Feature encoder backend used to build crop/text caches.",
    )
    parser.add_argument(
        "--model",
        default="convnext_large_d_320",
        help="OpenCLIP model name or HuggingFace model id for --encoder-backend siglip.",
    )
    parser.add_argument(
        "--pretrained",
        default="laion2b_s29b_b131k_ft_soup",
        help="OpenCLIP pretrained tag. Ignored by --encoder-backend siglip.",
    )
    parser.add_argument(
        "--image-mode",
        choices=("crop", "boxed"),
        default="crop",
        help="Region image passed to the encoder. boxed uses the full image with a red target box.",
    )
    parser.add_argument("--crop-margin", type=float, default=0.1)
    parser.add_argument("--box-line-width", type=int, default=6)
    parser.add_argument(
        "--target-field",
        default="label",
        help=(
            "JSONL field used as BCE training target. Keep the default for hard labels; "
            "use soft_label for VLM distillation rows."
        ),
    )
    parser.add_argument("--weight-field", default=None, help="Optional JSONL field used as per-row loss weight.")
    parser.add_argument("--positive-threshold", type=float, default=0.5)
    parser.add_argument("--positive-weight", type=float, default=1.0)
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
    parser.add_argument(
        "--verifier-arch",
        choices=("mlp", "gated_bilinear"),
        default="mlp",
        help=(
            "Verifier architecture. mlp preserves the original concat-feature scorer; "
            "gated_bilinear adds low-rank image-text interaction on top of the same cache."
        ),
    )
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument(
        "--projection-dim",
        type=int,
        default=256,
        help="Low-rank interaction dimension used by --verifier-arch gated_bilinear.",
    )
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument(
        "--loss-type",
        choices=("bce", "bce_rank", "listwise", "bce_listwise", "bce_rank_listwise"),
        default="bce",
    )
    parser.add_argument("--ranking-weight", type=float, default=0.0)
    parser.add_argument("--ranking-margin", type=float, default=0.0)
    parser.add_argument("--ranking-positive-threshold", type=float, default=0.5)
    parser.add_argument("--ranking-negative-threshold", type=float, default=0.3)
    parser.add_argument("--ranking-max-items-per-side", type=int, default=128)
    parser.add_argument(
        "--listwise-weight",
        type=float,
        default=1.0,
        help="Weight for per-image top-k teacher-distribution distillation losses.",
    )
    parser.add_argument(
        "--listwise-topk",
        type=int,
        default=20,
        help="Teacher-ranked candidates per image used in listwise distillation. Use 0 for all candidates.",
    )
    parser.add_argument(
        "--listwise-temperature",
        type=float,
        default=0.07,
        help="Softmax temperature for listwise teacher and student distributions.",
    )
    parser.add_argument(
        "--listwise-teacher-mode",
        choices=("logit_add", "linear", "replace"),
        default="logit_add",
        help="How to combine detector score and soft_label into the listwise teacher score.",
    )
    parser.add_argument(
        "--listwise-teacher-fusion-weight",
        type=float,
        default=0.04,
        help="Fusion weight used by the listwise teacher; 0.04 matches the best Qwen cached setting.",
    )
    parser.add_argument(
        "--feature-mode",
        choices=("full", "no_text", "no_detector_score"),
        default="full",
        help=(
            "Verifier input features. full uses crop/text/interactions/score; "
            "no_text removes all phrase features; no_detector_score removes detector score."
        ),
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

    train_cache_path = args.train_cache or _cache_path(args.cache_dir, "train")
    val_cache_path = args.val_cache or _cache_path(args.cache_dir, "val")
    needs_encoding = False
    for path in (train_cache_path, val_cache_path):
        if path is None or args.overwrite_cache or not path.exists():
            needs_encoding = True

    model = preprocess = tokenizer = None
    if needs_encoding:
        model, preprocess, tokenizer = _load_encoder(
            args.encoder_backend,
            args.model,
            args.pretrained,
            args.device,
        )

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
