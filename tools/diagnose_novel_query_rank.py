#!/usr/bin/env python3
"""Query-side novel-rank diagnostic.

For every rare (novel) GT box that has at least one decoder query with
IoU >= thr:
  * Take that best-IoU query's post-ensemble cls_score (BEFORE any rerank).
  * Ask: where does the GT novel class rank among the 337 novel scores?
  * Is the query novel-leaning (max novel >= max base)?

Aggregates:
  * Rank histogram of GT in novel-only score (1 / 2-5 / 6-20 / 21-100 / 101+).
  * % rare-TP queries that are novel-leaning.
  * Mean/median gap between top-1 novel score and GT-novel score.

Also dumps N sample rows: GT class, top-5 novel class names, GT rank, scores.

Why: if GT novel is already top-1 in the query's own novel scores, rerank
is useless and effort should go elsewhere (e.g. confidence calibration so
novel beats base at detection-level). If GT often sits outside top-k, the
model's cls_score itself is the bottleneck and any rerank based on
cls_score alone is downstream of a broken signal.

Usage:
  python tools/diagnose_novel_query_rank.py \\
    --config lami_dino/configs/dino_convnext_large_4scale_12ep_lvis.py \\
    --ckpt   /root/lami_convnext_large_12ep_lvis_20260419_132029/model_0042599.pth \\
    --num-images 500 \\
    --max-samples 40 \\
    --out analysis/novel_query_rank.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

import torch

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import LazyConfig, instantiate
from detectron2.data import DatasetCatalog
from detectron2.utils.logger import setup_logger

logger = logging.getLogger("diag.novel_rank")


def box_cxcywh_to_xyxy(b: torch.Tensor) -> torch.Tensor:
    cx, cy, w, h = b.unbind(-1)
    return torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=-1)


def pairwise_iou_xyxy(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    area_a = (a[:, 2] - a[:, 0]).clamp(min=0) * (a[:, 3] - a[:, 1]).clamp(min=0)
    area_b = (b[:, 2] - b[:, 0]).clamp(min=0) * (b[:, 3] - b[:, 1]).clamp(min=0)
    lt = torch.maximum(a[:, None, :2], b[None, :, :2])
    rb = torch.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    union = area_a[:, None] + area_b[None, :] - inter
    return inter / union.clamp(min=1e-6)


def _build_gt_index(dataset_name: str, device: str) -> Dict[int, Dict[str, Any]]:
    dicts = DatasetCatalog.get(dataset_name)
    gt: Dict[int, Dict[str, Any]] = {}
    for d in dicts:
        img_id = d.get("image_id")
        if img_id is None:
            continue
        boxes, classes = [], []
        for ann in d.get("annotations", []):
            if ann.get("iscrowd", 0):
                continue
            x, y, w, h = ann["bbox"]
            boxes.append([x, y, x + w, y + h])
            classes.append(int(ann["category_id"]))
        gt[img_id] = {
            "boxes_xyxy_orig": torch.tensor(boxes, dtype=torch.float32, device=device)
            if boxes else torch.zeros((0, 4), dtype=torch.float32, device=device),
            "classes": torch.tensor(classes, dtype=torch.long, device=device)
            if classes else torch.zeros((0,), dtype=torch.long, device=device),
            "orig_h": int(d.get("height", 0)),
            "orig_w": int(d.get("width", 0)),
        }
    logger.info(f"Built GT index for {dataset_name}: {len(gt)} images")
    return gt


def _install_capture(model):
    """Capture post-ensemble cls_score (and box_pred) by shadowing inference()."""
    captures: List[Dict[str, Any]] = []
    orig_inference = model.inference

    def inference_with_capture(*args, **kwargs):
        box_cls = kwargs.get("box_cls", args[0])
        box_pred = kwargs.get("box_pred", args[1])
        image_sizes = kwargs.get("image_sizes", args[2])
        captures.append({
            "cls_score": box_cls.detach().cpu(),    # [B, Q, C], already post-ensemble
            "box_pred": box_pred.detach().cpu(),    # [B, Q, 4] cxcywh in [0,1]
            "image_sizes": list(image_sizes),
        })
        return orig_inference(*args, **kwargs)

    model.inference = inference_with_capture
    return captures


def _rank_bucket(rank: int) -> str:
    if rank == 1: return "r1"
    if rank <= 5: return "r2-5"
    if rank <= 20: return "r6-20"
    if rank <= 100: return "r21-100"
    return "r101+"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--out", default="analysis/novel_query_rank.json")
    parser.add_argument("--num-images", type=int, default=500)
    parser.add_argument("--max-samples", type=int, default=40)
    parser.add_argument("--iou-thr", type=float, default=0.5)
    parser.add_argument("--show-topk", type=int, default=5)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    setup_logger(name="diag")
    logger.setLevel(logging.INFO)

    cfg = LazyConfig.load(args.config)
    cfg.model.analogical_rerank = False  # diagnostic works on clean post-ensemble
    model = instantiate(cfg.model).to(args.device).eval()
    DetectionCheckpointer(model).load(args.ckpt)

    dataloader = instantiate(cfg.dataloader.test)
    dataset_name = cfg.dataloader.test.dataset.names
    if isinstance(dataset_name, (list, tuple)):
        dataset_name = dataset_name[0]
    gt_by_img = _build_gt_index(dataset_name, args.device)

    all_classes: List[str] = list(model.all_classes)
    base_mask = model.base_idx.cpu().bool()
    novel_mask = model.novel_idx.cpu().bool()
    novel_class_ids = torch.where(novel_mask)[0].tolist()
    novel_id_to_local = {cid: i for i, cid in enumerate(novel_class_ids)}

    captures = _install_capture(model)

    bucket_counts: Counter = Counter()
    novel_leaning_hits = 0
    rare_gt_matched = 0
    rare_gt_total = 0
    gap_topnovel_vs_gt: List[float] = []
    samples: List[Dict[str, Any]] = []
    processed = 0
    limit = args.num_images if args.num_images > 0 else float("inf")

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if processed >= limit:
                break
            captures.clear()
            try:
                _ = model(batch)
            except Exception as e:
                logger.warning(f"skip batch {batch_idx}: {type(e).__name__}: {e}")
                continue
            if not captures:
                processed += len(batch)
                continue
            cap = captures[-1]
            cls_score = cap["cls_score"]          # [B, Q, C]
            box_pred = cap["box_pred"]            # [B, Q, 4]
            image_sizes = cap["image_sizes"]
            B, Q, _ = box_pred.shape

            for b in range(B):
                inp = batch[b]
                img_id = inp.get("image_id")
                entry = gt_by_img.get(img_id) if img_id is not None else None
                if entry is None or entry["classes"].numel() == 0:
                    continue

                H_resize, W_resize = image_sizes[b]
                orig_h, orig_w = entry["orig_h"], entry["orig_w"]
                if orig_h <= 0 or orig_w <= 0:
                    continue

                pred_xyxy = box_cxcywh_to_xyxy(box_pred[b])
                pred_xyxy = pred_xyxy * torch.tensor(
                    [W_resize, H_resize, W_resize, H_resize], dtype=torch.float32
                )
                sx = W_resize / float(orig_w)
                sy = H_resize / float(orig_h)
                gt_boxes = entry["boxes_xyxy_orig"].cpu() * torch.tensor(
                    [sx, sy, sx, sy], dtype=torch.float32
                )
                gt_classes = entry["classes"].cpu()

                ious = pairwise_iou_xyxy(gt_boxes, pred_xyxy)  # [num_gt, Q]

                for g in range(gt_classes.numel()):
                    gt_c = int(gt_classes[g].item())
                    if bool(base_mask[gt_c].item()):
                        continue
                    rare_gt_total += 1

                    best_iou, best_q = ious[g].max(dim=0)
                    if float(best_iou.item()) < args.iou_thr:
                        continue
                    rare_gt_matched += 1
                    q = int(best_q.item())

                    q_scores = cls_score[b, q]              # [C]
                    q_novel = q_scores[novel_mask]          # [num_novel]
                    q_base = q_scores[base_mask]            # [num_base]

                    if float(q_novel.max().item()) >= float(q_base.max().item()):
                        novel_leaning_hits += 1

                    gt_local = novel_id_to_local[gt_c]
                    gt_score = float(q_novel[gt_local].item())
                    # rank = number of novels with strictly higher score + 1
                    rank = int((q_novel > q_novel[gt_local]).sum().item()) + 1
                    bucket_counts[_rank_bucket(rank)] += 1

                    top_novel_score = float(q_novel.max().item())
                    gap_topnovel_vs_gt.append(top_novel_score - gt_score)

                    if len(samples) < args.max_samples:
                        top_vals, top_idx = q_novel.topk(args.show_topk)
                        top_names = [all_classes[novel_class_ids[int(i)]] for i in top_idx]
                        samples.append({
                            "image_id": int(img_id),
                            "query_idx": q,
                            "iou": round(float(best_iou.item()), 3),
                            "gt_class": all_classes[gt_c],
                            "gt_rank_in_novel": rank,
                            "gt_score": round(gt_score, 4),
                            "top_novel_score": round(top_novel_score, 4),
                            "top_base_score": round(float(q_base.max().item()), 4),
                            "query_is_novel_leaning": bool(
                                float(q_novel.max().item()) >= float(q_base.max().item())
                            ),
                            "top_novels": [
                                {"name": n, "score": round(float(s.item()), 4)}
                                for n, s in zip(top_names, top_vals)
                            ],
                        })

            processed += B

    # Aggregate.
    matched = max(rare_gt_matched, 1)
    gaps = torch.tensor(gap_topnovel_vs_gt) if gap_topnovel_vs_gt else torch.zeros(1)
    report = {
        "num_images_processed": processed,
        "rare_gt_total": rare_gt_total,
        "rare_gt_matched": rare_gt_matched,
        "match_rate": rare_gt_matched / max(rare_gt_total, 1),
        "novel_leaning_rate_over_matched": novel_leaning_hits / matched,
        "rank_buckets_over_matched": {
            k: {"count": bucket_counts[k], "rate": bucket_counts[k] / matched}
            for k in ["r1", "r2-5", "r6-20", "r21-100", "r101+"]
        },
        "gap_top_novel_vs_gt": {
            "mean": float(gaps.mean().item()),
            "median": float(gaps.median().item()),
            "p90": float(gaps.quantile(0.9).item()) if gaps.numel() > 1 else 0.0,
        },
        "samples": samples,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"Saved report -> {out_path}")

    print("\n=== Novel-rank query-side diagnostic ===")
    print(f"images: {processed}")
    print(f"rare GTs: total={rare_gt_total}, matched(IoU>={args.iou_thr})={rare_gt_matched}")
    print(f"  match_rate        = {report['match_rate']*100:.1f}%")
    print(f"  novel-leaning rate= {report['novel_leaning_rate_over_matched']*100:.1f}%  (of matched)")
    print("  GT rank in novel-only scores (of matched):")
    for k in ["r1", "r2-5", "r6-20", "r21-100", "r101+"]:
        r = report["rank_buckets_over_matched"][k]
        print(f"    {k:>8}: {r['count']:>5d}  ({r['rate']*100:.1f}%)")
    g = report["gap_top_novel_vs_gt"]
    print(f"  gap (top_novel - gt_novel): mean={g['mean']:.3f}  median={g['median']:.3f}  p90={g['p90']:.3f}")
    print(f"\nFirst {min(len(samples), 15)} samples:")
    for i, s in enumerate(samples[:15]):
        tail = "<< GT TOP1" if s["gt_rank_in_novel"] == 1 else ""
        print(f"[{i:02d}] img={s['image_id']} q={s['query_idx']} iou={s['iou']} "
              f"GT={s['gt_class']} rank={s['gt_rank_in_novel']} "
              f"gt_score={s['gt_score']} top_novel={s['top_novel_score']} "
              f"top_base={s['top_base_score']} novel_leaning={s['query_is_novel_leaning']} {tail}")
        topn = ", ".join(f"{t['name']}({t['score']:.3f})" for t in s["top_novels"])
        print(f"     top-{len(s['top_novels'])} novels: {topn}")


if __name__ == "__main__":
    sys.exit(main())
