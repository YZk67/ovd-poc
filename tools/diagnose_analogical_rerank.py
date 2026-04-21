#!/usr/bin/env python3
"""Analogical-rerank sanity-check diagnostic.

For predicted boxes that match a RARE (novel) GT at IoU>=thr, print:
  - GT class name
  - Top-k base "ancestors" the rerank picks (base_topk_idx + softmax weights)
  - Rerank's preferred novel class (concept_prior argmax)
  - Model's top-1 novel before and after rerank

Use this to eyeball whether the analogical base lookup is semantically sane
(e.g., GT=unicycle -> top-k base should contain bicycle/tricycle/scooter).

Usage:
  python tools/diagnose_analogical_rerank.py \\
    --config lami_dino/configs/dino_convnext_large_4scale_12ep_lvis.py \\
    --ckpt   /root/lami_convnext_large_12ep_lvis_20260419_132029/model_0042599.pth \\
    --num-images 200 \\
    --max-samples 60 \\
    --out analysis/analogical_rerank_samples.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import types
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import LazyConfig, instantiate
from detectron2.data import DatasetCatalog
from detectron2.utils.logger import setup_logger

logger = logging.getLogger("diag.analog")


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


def _install_captures(model):
    """Monkey-patch rerank + inference so we can capture per-image state."""
    captures: List[Dict[str, Any]] = []
    pending: Dict[str, Any] = {}

    orig_rerank = model._apply_analogical_rerank

    def rerank_with_capture(self, cls_score):
        if not self.analogical_rerank or self.analogical_gamma <= 0:
            return orig_rerank(cls_score)

        base_scores = cls_score[:, :, self.base_idx]
        novel_scores = cls_score[:, :, self.novel_idx]
        if base_scores.shape[-1] == 0 or novel_scores.shape[-1] == 0:
            return orig_rerank(cls_score)

        topk = min(int(self.analogical_topk), base_scores.shape[-1])
        tau_base = max(float(self.analogical_tau_base), 1e-6)
        tau_concept = max(float(self.analogical_tau_concept), 1e-6)
        base_topk_scores, base_topk_idx = torch.topk(base_scores, topk, dim=-1)
        base_weights = F.softmax(base_topk_scores / tau_base, dim=-1)

        # Build text prototypes the same way the real rerank does.
        text_prototypes = None
        class_embed = getattr(self, "class_embed", None)
        tc = class_embed[-1] if isinstance(class_embed, torch.nn.ModuleList) else None
        cached = getattr(tc, "_cached_eval", None) if tc is not None else None
        if cached is not None:
            t = cached
            if t.ndim == 3:
                t = t.mean(dim=1)
            text_prototypes = F.normalize(t.to(cls_score.device), p=2, dim=-1)
        elif tc is not None and hasattr(tc, "eval_text_feats"):
            t = tc.eval_text_feats
            if t.ndim == 3:
                t = t.mean(dim=1)
            text_prototypes = F.normalize(t.to(cls_score.device), p=2, dim=-1)
        else:
            text_prototypes = F.normalize(
                self.eval_content_query_embedding.to(cls_score.device), p=2, dim=-1
            )
        base_text = text_prototypes[self.base_idx]
        novel_text = text_prototypes[self.novel_idx]
        analogical_text = base_text[base_topk_idx]
        analogical_concept = F.normalize(
            (base_weights.unsqueeze(-1) * analogical_text).sum(dim=-2), p=2, dim=-1
        )
        concept_sim = analogical_concept @ novel_text.t()
        concept_prior = F.softmax(concept_sim / tau_concept, dim=-1)

        pending["cls_score_pre"] = cls_score.detach().cpu()
        pending["base_topk_idx"] = base_topk_idx.detach().cpu()
        pending["base_weights"] = base_weights.detach().cpu()
        pending["concept_prior"] = concept_prior.detach().cpu()

        out = orig_rerank(cls_score)
        pending["cls_score_post"] = out.detach().cpu()
        return out

    model._apply_analogical_rerank = types.MethodType(rerank_with_capture, model)

    orig_inference = model.inference

    def inference_with_capture(*args, **kwargs):
        # signature: (self, box_cls, box_pred, image_sizes, wo_sigmoid=...)
        # When called as bound method via monkey-patch, args[0] is self; but we used
        # types.MethodType below so args does NOT include self. Handle both.
        if len(args) >= 3 and isinstance(args[0], torch.Tensor):
            box_cls, box_pred, image_sizes = args[0], args[1], args[2]
        else:
            box_cls = kwargs.get("box_cls", args[1])
            box_pred = kwargs.get("box_pred", args[2])
            image_sizes = kwargs.get("image_sizes", args[3])
        pending["box_pred"] = box_pred.detach().cpu()
        pending["image_sizes"] = list(image_sizes)
        captures.append(dict(pending))
        pending.clear()
        return orig_inference(*args, **kwargs)

    model.inference = types.MethodType(inference_with_capture, model)
    return captures


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--out", default="analysis/analogical_rerank_samples.json")
    parser.add_argument("--num-images", type=int, default=200)
    parser.add_argument("--max-samples", type=int, default=60,
                        help="Stop after logging this many rare TPs.")
    parser.add_argument("--iou-thr", type=float, default=0.5)
    parser.add_argument("--score-thr", type=float, default=0.1,
                        help="Minimum post-rerank novel score to keep a query.")
    parser.add_argument("--show-topk", type=int, default=5)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    setup_logger(name="diag")
    logger.setLevel(logging.INFO)

    cfg = LazyConfig.load(args.config)
    cfg.model.analogical_rerank = True  # force on regardless of config
    model = instantiate(cfg.model).to(args.device).eval()
    DetectionCheckpointer(model).load(args.ckpt)

    dataloader = instantiate(cfg.dataloader.test)
    dataset_name = cfg.dataloader.test.dataset.names
    if isinstance(dataset_name, (list, tuple)):
        dataset_name = dataset_name[0]
    gt_by_img = _build_gt_index(dataset_name, args.device)

    # Class id -> name. model.all_classes is a list of strings (loaded in DINO.__init__).
    all_classes: List[str] = list(model.all_classes)
    base_mask = model.base_idx.to(args.device).bool()
    novel_mask = model.novel_idx.to(args.device).bool()
    base_class_ids = torch.where(base_mask.cpu())[0].tolist()
    novel_class_ids = torch.where(novel_mask.cpu())[0].tolist()

    captures = _install_captures(model)

    samples: List[Dict[str, Any]] = []
    processed = 0
    limit = args.num_images if args.num_images > 0 else float("inf")

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if processed >= limit or len(samples) >= args.max_samples:
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

            box_pred = cap["box_pred"]         # [B, Q, 4] cxcywh in [0,1]
            cls_pre = cap["cls_score_pre"]     # [B, Q, C] post-ensemble pre-rerank
            cls_post = cap["cls_score_post"]
            base_topk_idx = cap["base_topk_idx"]  # [B, Q, topk] in base-subset space
            base_weights = cap["base_weights"]    # [B, Q, topk]
            concept_prior = cap["concept_prior"]  # [B, Q, num_novel]
            image_sizes = cap["image_sizes"]

            B, Q, _ = box_pred.shape
            for b in range(B):
                if len(samples) >= args.max_samples:
                    break
                inp = batch[b]
                img_id = inp.get("image_id")
                entry = gt_by_img.get(img_id) if img_id is not None else None
                if entry is None or entry["classes"].numel() == 0:
                    continue

                H_resize, W_resize = image_sizes[b]
                orig_h, orig_w = entry["orig_h"], entry["orig_w"]
                if orig_h <= 0 or orig_w <= 0:
                    continue

                # pred boxes are normalized in [0,1]; they'll be scaled by image_size
                # by the inference module. For IoU against GT we need the same coord space.
                pred_xyxy = box_cxcywh_to_xyxy(box_pred[b])
                scale_pred = torch.tensor(
                    [W_resize, H_resize, W_resize, H_resize], dtype=torch.float32
                )
                pred_xyxy = pred_xyxy * scale_pred

                sx = W_resize / float(orig_w)
                sy = H_resize / float(orig_h)
                gt_boxes = entry["boxes_xyxy_orig"].cpu() * torch.tensor(
                    [sx, sy, sx, sy], dtype=torch.float32
                )
                gt_classes = entry["classes"].cpu()

                ious = pairwise_iou_xyxy(gt_boxes, pred_xyxy)  # [num_gt, Q]

                for g in range(gt_classes.numel()):
                    if len(samples) >= args.max_samples:
                        break
                    gt_c = int(gt_classes[g].item())
                    if bool(base_mask.cpu()[gt_c].item()):
                        continue  # only care about rare/novel GT

                    # Find best query for this GT
                    iou_g = ious[g]
                    best_iou, best_q = iou_g.max(dim=0)
                    if float(best_iou.item()) < args.iou_thr:
                        continue

                    q = int(best_q.item())

                    post_novel = cls_post[b, q][novel_mask.cpu()]
                    pre_novel = cls_pre[b, q][novel_mask.cpu()]
                    if float(post_novel.max().item()) < args.score_thr:
                        continue

                    # Translate base-subset topk indices back to global class ids
                    topk_base_local = base_topk_idx[b, q].tolist()
                    topk_base_ids = [base_class_ids[i] for i in topk_base_local]
                    topk_weights = base_weights[b, q].tolist()

                    concept_argmax_local = int(concept_prior[b, q].argmax().item())
                    concept_novel_id = novel_class_ids[concept_argmax_local]
                    concept_novel_top = float(concept_prior[b, q].max().item())

                    # Top-1 novel class (model prediction) pre/post rerank
                    pre_top_local = int(pre_novel.argmax().item())
                    post_top_local = int(post_novel.argmax().item())

                    sample = {
                        "image_id": int(img_id),
                        "query_idx": q,
                        "iou": round(float(best_iou.item()), 3),
                        "gt_class": all_classes[gt_c],
                        "gt_class_id": gt_c,
                        "top_base_ancestors": [
                            {"name": all_classes[cid], "weight": round(w, 3)}
                            for cid, w in zip(topk_base_ids, topk_weights)
                        ][: args.show_topk],
                        "concept_prior_top1": {
                            "name": all_classes[concept_novel_id],
                            "prob": round(concept_novel_top, 3),
                        },
                        "model_top1_novel_pre": {
                            "name": all_classes[novel_class_ids[pre_top_local]],
                            "score": round(float(pre_novel.max().item()), 3),
                        },
                        "model_top1_novel_post": {
                            "name": all_classes[novel_class_ids[post_top_local]],
                            "score": round(float(post_novel.max().item()), 3),
                        },
                    }
                    samples.append(sample)

            processed += B

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump({"num_samples": len(samples), "samples": samples}, f, indent=2)
    logger.info(f"Saved {len(samples)} samples -> {out_path}")

    print("\n=== Analogical rerank: rare-TP samples ===")
    for i, s in enumerate(samples[:30]):
        anc = ", ".join(f"{a['name']}({a['weight']:.2f})" for a in s["top_base_ancestors"])
        print(f"[{i:02d}] img={s['image_id']} q={s['query_idx']} iou={s['iou']}")
        print(f"     GT={s['gt_class']}")
        print(f"     ancestors: {anc}")
        print(f"     concept_prior top1 = {s['concept_prior_top1']['name']} ({s['concept_prior_top1']['prob']:.2f})")
        print(f"     model top1 novel  pre={s['model_top1_novel_pre']['name']}({s['model_top1_novel_pre']['score']:.2f}) "
              f"post={s['model_top1_novel_post']['name']}({s['model_top1_novel_post']['score']:.2f})")


if __name__ == "__main__":
    sys.exit(main())
