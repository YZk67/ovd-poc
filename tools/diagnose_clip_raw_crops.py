#!/usr/bin/env python
"""Compare rare-GT raw-image CLIP crops with saved p3 GT-ROI features.

This is a read-only diagnostic, not an inference or training option. It reuses
the final detector checkpoint's frozen ConvNeXt-L visual trunk, frozen CLIP
head, image normalization, and 1203-class text bank. Only selected GT crops
run through the visual trunk; the detector is never run again.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from tools.diagnose_tpa_usage import true_class_clip_stats  # noqa: E402


def select_rare_rows(rows, *, max_misses, max_hit_controls, max_no_box_controls, seed):
    """Keep all misses by default and deterministic, bounded control groups."""
    groups = defaultdict(list)
    for row in rows:
        if row["status"] not in ("current_miss", "current_hit", "no_eligible_box"):
            raise ValueError(f"unknown rare GT status: {row['status']!r}")
        groups[row["status"]].append(row)
    rng = np.random.default_rng(seed)
    selected = []
    limits = {
        "current_miss": max_misses,
        "current_hit": max_hit_controls,
        "no_eligible_box": max_no_box_controls,
    }
    for status, limit in limits.items():
        group = groups[status]
        if status != "current_miss" and limit == 0:
            group = []
        elif limit > 0 and len(group) > limit:
            indices = np.sort(rng.choice(len(group), size=limit, replace=False))
            group = [group[int(index)] for index in indices]
        selected.extend(dict(row) for row in group)
    selected.sort(key=lambda row: (int(row["image_id"]), int(row["gt_index"])))
    return selected


def square_context_crop(image, xyxy, *, scale, size):
    """Square GT crop with optional context and mean-RGB border padding."""
    if scale < 1.0 or size < 1:
        raise ValueError("crop scale must be >=1 and output size positive")
    x0, y0, x1, y1 = (float(value) for value in xyxy)
    if not (x1 > x0 and y1 > y0):
        raise ValueError(f"invalid GT box: {xyxy}")
    side = max(x1 - x0, y1 - y0) * scale
    left = math.floor((x0 + x1 - side) * 0.5)
    top = math.floor((y0 + y1 - side) * 0.5)
    right = math.ceil((x0 + x1 + side) * 0.5)
    bottom = math.ceil((y0 + y1 + side) * 0.5)
    canvas_side = max(right - left, bottom - top)
    if canvas_side < 1:
        raise ValueError("crop rounded to an empty image")
    canvas = Image.new("RGB", (canvas_side, canvas_side), (124, 116, 104))
    src_left = max(left, 0)
    src_top = max(top, 0)
    src_right = min(left + canvas_side, image.width)
    src_bottom = min(top + canvas_side, image.height)
    if src_right > src_left and src_bottom > src_top:
        patch = image.crop((src_left, src_top, src_right, src_bottom))
        canvas.paste(patch, (src_left - left, src_top - top))
    resampling = Image.Resampling.BICUBIC if hasattr(Image, "Resampling") else Image.BICUBIC
    return canvas.resize((size, size), resample=resampling)


def raw_crop_clip_logits(model, crop_batch):
    """Use the same frozen CLIP trunk/head/text bank as detector ROI ensembling."""
    if not model.score_ensemble:
        raise ValueError("raw crop diagnosis requires score_ensemble=True")
    normalized = model.normalizer(crop_batch.float())
    _features, unnormalized_features = model.backbone(normalized)
    visual = unnormalized_features["p3"]
    visual = model.head(model.thead(model.identical(visual)))
    if visual.ndim != 2 or visual.shape[0] != crop_batch.shape[0]:
        raise ValueError(f"unexpected CLIP head output shape: {tuple(visual.shape)}")
    visual = F.normalize(visual.float(), p=2, dim=-1)
    return (
        visual @ model.vlm_content_query_embedding.t()
        * float(model.vlm_temperature)
    )


def summarize_raw_crop_rows(rows, scales):
    """Compare raw crops to the previously measured GT-box p3 ROI per status."""
    report = {}
    for scale in scales:
        key = str(scale)
        report[key] = {}
        for status in ("current_miss", "current_hit", "no_eligible_box"):
            selected = [row for row in rows if row["status"] == status]
            if not selected:
                report[key][status] = {"count": 0}
                continue
            roi_ranks = np.asarray([row["gt_rank"] for row in selected])
            raw_ranks = np.asarray([row["raw_crop"][key]["rank"] for row in selected])
            roi_probs = np.asarray([row["gt_true_probability"] for row in selected])
            raw_probs = np.asarray(
                [row["raw_crop"][key]["true_probability"] for row in selected]
            )
            report[key][status] = {
                "count": len(selected),
                "roi_top1": float(np.mean(roi_ranks == 1)),
                "raw_top1": float(np.mean(raw_ranks == 1)),
                "roi_top5": float(np.mean(roi_ranks <= 5)),
                "raw_top5": float(np.mean(raw_ranks <= 5)),
                "roi_median_rank": float(np.median(roi_ranks)),
                "raw_median_rank": float(np.median(raw_ranks)),
                "roi_median_true_probability": float(np.median(roi_probs)),
                "raw_median_true_probability": float(np.median(raw_probs)),
                "raw_rescues_roi_top5": int(np.sum((roi_ranks > 5) & (raw_ranks <= 5))),
                "raw_worsens_roi_top5": int(np.sum((roi_ranks <= 5) & (raw_ranks > 5))),
                "median_raw_minus_roi_log_probability": float(
                    np.median(
                        np.log(np.maximum(raw_probs, 1e-12))
                        - np.log(np.maximum(roi_probs, 1e-12))
                    )
                ),
            }
    return report


def print_summary(summary):
    print("\n=== Raw GT crop vs p3 GT-box ROI: same CLIP head/text bank ===")
    print(
        "The raw-crop path also changes resolution and context. This comparison "
        "does not by itself isolate an ROIAlign bug or estimate detector AP."
    )
    print(
        f"{'scale':>6} {'status':>16} {'N':>5} {'ROI top5%':>10} "
        f"{'crop top5%':>11} {'ROI p-med':>12} {'crop p-med':>12} "
        f"{'rescued':>8} {'worsened':>9}"
    )
    for scale, statuses in summary.items():
        for status in ("current_miss", "current_hit", "no_eligible_box"):
            row = statuses[status]
            if not row["count"]:
                continue
            print(
                f"{scale:>6} {status:>16} {row['count']:5d} "
                f"{100 * row['roi_top5']:10.2f} {100 * row['raw_top5']:11.2f} "
                f"{row['roi_median_true_probability']:12.6f} "
                f"{row['raw_median_true_probability']:12.6f} "
                f"{row['raw_rescues_roi_top5']:8d} "
                f"{row['raw_worsens_roi_top5']:9d}"
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-json", required=True)
    parser.add_argument("--config-file", default="lami_dino/configs/dino_convnext_large_4scale_12ep_lvis.py")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--crop-size", type=int, default=320)
    parser.add_argument("--crop-scales", type=float, nargs="+", default=[1.0, 1.25])
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-misses", type=int, default=0, help="0 keeps all current misses")
    parser.add_argument("--max-hit-controls", type=int, default=100, help="0 omits hit controls")
    parser.add_argument("--max-no-box-controls", type=int, default=50, help="0 omits no-box controls")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "opts", nargs=argparse.REMAINDER,
        help="LazyConfig overrides, preceded by -- after --crop-scales",
    )
    args = parser.parse_args()
    if args.opts and args.opts[0] == "--":
        args.opts = args.opts[1:]
    if args.crop_size < 32 or args.batch_size < 1:
        raise ValueError("--crop-size must be >=32 and --batch-size positive")
    if len(set(args.crop_scales)) != len(args.crop_scales):
        raise ValueError("--crop-scales must be unique")
    if any(scale < 1.0 for scale in args.crop_scales):
        raise ValueError("--crop-scales must be >=1")
    if min(args.max_misses, args.max_hit_controls, args.max_no_box_controls) < 0:
        raise ValueError("sample limits must be nonnegative")

    source = json.loads(Path(args.source_json).read_text(encoding="utf-8"))
    if Path(source["checkpoint"]).resolve() != Path(args.checkpoint).resolve():
        raise ValueError("source diagnostic and raw crop checkpoint differ")
    if Path(source["config"]).resolve() != Path(args.config_file).resolve():
        raise ValueError("source diagnostic and raw crop config differ")
    source_rows = source.get("gt_roi_rare_instances")
    if not source_rows:
        raise ValueError("source JSON needs --compare-gt-roi per-GT results")
    selected = select_rare_rows(
        source_rows,
        max_misses=args.max_misses,
        max_hit_controls=args.max_hit_controls,
        max_no_box_controls=args.max_no_box_controls,
        seed=args.seed,
    )
    if not selected:
        raise ValueError("no rare GT rows selected")
    print(
        f"[select] {len(selected)} rare GT rows, "
        f"status counts={dict((status, sum(row['status'] == status for row in selected)) for status in ('current_miss', 'current_hit', 'no_eligible_box'))}",
        flush=True,
    )

    from detectron2.checkpoint import DetectionCheckpointer
    from detectron2.config import LazyConfig, instantiate
    from detectron2.data import get_detection_dataset_dicts

    cfg = LazyConfig.load(args.config_file)
    cfg = LazyConfig.apply_overrides(cfg, args.opts)
    if str(cfg.dataloader.test.mapper.img_format).upper() != "RGB":
        raise ValueError("raw GT crops assume the validation mapper reads RGB images")
    model = instantiate(cfg.model)
    if not torch.cuda.is_available():
        raise RuntimeError("raw CLIP crop diagnosis requires a GPU")
    device = torch.device("cuda")
    model.to(device).eval()
    checkpoint_state = DetectionCheckpointer(model).load(args.checkpoint)
    if not getattr(model, "score_ensemble", False):
        raise ValueError("raw crop diagnosis needs the CLIP score-ensemble branch")
    for name in ("alpha", "beta", "novel_scale", "tpa_eval_mode_scale", "vlm_temperature"):
        expected = source["fusion_protocol"].get(name)
        if expected is not None and not math.isclose(float(getattr(model, name)), float(expected)):
            raise ValueError(f"{name} differs from the source diagnostic")
    print(
        f"[load] checkpoint={args.checkpoint} iteration={checkpoint_state.get('iteration')} "
        f"crop_size={args.crop_size} scales={args.crop_scales}",
        flush=True,
    )

    dataset_name = cfg.dataloader.test.dataset.names
    if not isinstance(dataset_name, str):
        if len(dataset_name) != 1:
            raise ValueError("raw crop diagnosis requires one validation dataset")
        dataset_name = dataset_name[0]
    wanted_ids = {int(row["image_id"]) for row in selected}
    records = get_detection_dataset_dicts(names=dataset_name, filter_empty=False)
    paths = {
        int(record["image_id"]): Path(record["file_name"])
        for record in records
        if int(record["image_id"]) in wanted_ids
    }
    missing = wanted_ids - paths.keys()
    if missing:
        raise ValueError(f"missing validation image ids: {sorted(missing)[:10]}")
    text_bank = model.vlm_content_query_embedding
    if text_bank.ndim != 2 or text_bank.shape[0] != model.num_classes:
        raise ValueError("CLIP text bank does not cover every detector class")

    rows_by_image = defaultdict(list)
    for row in selected:
        row["raw_crop"] = {}
        rows_by_image[int(row["image_id"])].append(row)
    pending_tensors = []
    pending_rows = []
    processed = 0
    total = len(selected) * len(args.crop_scales)

    def flush():
        nonlocal processed
        if not pending_tensors:
            return
        batch = torch.stack(pending_tensors).to(device=device, dtype=torch.float32)
        with torch.inference_mode():
            logits = raw_crop_clip_logits(model, batch)
            classes = torch.tensor(
                [row["category_id"] for row, _scale in pending_rows],
                dtype=torch.long,
                device=device,
            )
            ranks, probabilities = true_class_clip_stats(logits, classes)
            top1_classes = logits.argmax(dim=-1)
        for index, (row, scale) in enumerate(pending_rows):
            row["raw_crop"][str(scale)] = {
                "rank": int(ranks[index]),
                "true_probability": float(probabilities[index]),
                "top1_category_id": int(top1_classes[index]),
            }
        processed += len(pending_tensors)
        pending_tensors.clear()
        pending_rows.clear()
        if processed % 50 < args.batch_size or processed == total:
            print(f"[crop] {processed}/{total}", flush=True)

    for image_id in sorted(rows_by_image):
        with Image.open(paths[image_id]) as original:
            image = original.convert("RGB")
            for row in rows_by_image[image_id]:
                for scale in args.crop_scales:
                    crop = square_context_crop(
                        image, row["gt_xyxy"], scale=scale, size=args.crop_size
                    )
                    array = np.asarray(crop, dtype=np.uint8).copy()
                    pending_tensors.append(torch.from_numpy(array).permute(2, 0, 1))
                    pending_rows.append((row, scale))
                    if len(pending_tensors) >= args.batch_size:
                        flush()
    flush()
    if processed != total:
        raise RuntimeError(f"processed {processed} crops, expected {total}")
    summary = summarize_raw_crop_rows(selected, args.crop_scales)
    print_summary(summary)
    report = {
        "source_json": args.source_json,
        "checkpoint": args.checkpoint,
        "checkpoint_iteration": checkpoint_state.get("iteration"),
        "config": args.config_file,
        "crop_size": args.crop_size,
        "crop_scales": args.crop_scales,
        "batch_size": args.batch_size,
        "source_fusion_protocol": source["fusion_protocol"],
        "summary": summary,
        "rare_instances": selected,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[save] {output_path}", flush=True)


if __name__ == "__main__":
    main()
