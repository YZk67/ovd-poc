#!/usr/bin/env python3
"""Encoder top-k class-hit diagnostic for LaMI-DETR.

For every GT box in the validation set:
  1. Run backbone + encoder; get top-k proposals (post two-stage selection)
     with logits over all classes and normalized cxcywh boxes.
  2. Assign each GT to its best-IoU encoder proposal (IoU >= --iou-thr).
  3. Check whether that proposal's encoder top-1 / top-K class == GT class.

Aggregate by bucket:
  * If --class-groups JSON is given (id_str -> group_name), use it.
  * Else if model.base_idx is available (set when score_ensemble=True),
    stratify by base / novel.
  * Else: single bucket "all".

Output: JSON with per-bucket counts and rates, plus a plain-text summary.

Why this matters:
  The current decoder content_query is initialized from the K text prototypes
  of the encoder's top-1 class. If top-1 is wrong (especially on novel/rare),
  the decoder gets contaminated initialization. A low top-1 hit rate on
  novel/rare is the empirical signal that a cross-class visual-visual retrieval
  module (ATCG-style) could help.

Usage:
  python tools/diagnose_encoder_topk.py \
    --config lami_dino/configs/dino_convnext_large_4scale_12ep_lvis.py \
    --ckpt   /path/to/model_final.pth \
    --num-images 1000 \
    --out    analysis/encoder_topk.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import LazyConfig, instantiate
from detectron2.utils.logger import setup_logger

logger = logging.getLogger("diag.encoder_topk")


def box_cxcywh_to_xyxy(b: torch.Tensor) -> torch.Tensor:
    cx, cy, w, h = b.unbind(-1)
    return torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=-1)


def pairwise_iou_xyxy(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """a:[N,4] xyxy, b:[M,4] xyxy -> [N,M] IoU."""
    area_a = (a[:, 2] - a[:, 0]).clamp(min=0) * (a[:, 3] - a[:, 1]).clamp(min=0)
    area_b = (b[:, 2] - b[:, 0]).clamp(min=0) * (b[:, 3] - b[:, 1]).clamp(min=0)
    lt = torch.maximum(a[:, None, :2], b[None, :, :2])
    rb = torch.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    union = area_a[:, None] + area_b[None, :] - inter
    return inter / union.clamp(min=1e-6)


@torch.no_grad()
def _build_content_query_embeds(model) -> torch.Tensor:
    """Replicate the content_query_embeds construction from model.forward (eval branch)."""
    text_clf = model.transformer.decoder.class_embed[0]
    if getattr(text_clf, "use_tpa", False):
        text_feats = text_clf._maybe_move_text_feats(training=False)
        proto_ckd, _ = text_clf.tpa(text_feats, with_loss=False)
        proto_ckd = model.content_layer(
            proto_ckd.reshape(-1, proto_ckd.size(-1))
        ).reshape(proto_ckd.size(0), proto_ckd.size(1), -1)
        proto_ckd = F.normalize(proto_ckd, p=2, dim=-1)
        return proto_ckd  # [C, K, D]
    emb = model.content_layer(model.content_query_embedding)
    emb = F.normalize(emb, p=2, dim=1)
    return emb.unsqueeze(1)  # [C, 1, D]


@torch.no_grad()
def run_encoder_only(model, batched_inputs) -> Tuple[torch.Tensor, torch.Tensor, List[Tuple[int, int]], Tuple[int, int]]:
    """Return (enc_logits [B, topk, C], enc_ref_norm [B, topk, 4 cxcywh], image_sizes, padded_hw).

    Mirrors model.forward up to the encoder's two-stage output; skips the decoder.
    """
    images = model.preprocess_image(batched_inputs)
    B, _, H_pad, W_pad = images.tensor.shape
    img_masks = images.tensor.new_zeros(B, H_pad, W_pad)

    if getattr(model, "score_ensemble", False):
        features, _ = model.backbone(images.tensor)
    else:
        features = model.backbone(images.tensor)

    multi_level_feats = model.neck(features)
    multi_level_masks = []
    multi_level_pe = []
    for feat in multi_level_feats:
        mask = F.interpolate(img_masks[None], size=feat.shape[-2:]).to(torch.bool).squeeze(0)
        multi_level_masks.append(mask)
        multi_level_pe.append(model.position_embedding(mask))

    raw_content = _build_content_query_embeds(model)

    # Propagate soft-attention flags like dino.py does.
    if getattr(model, "use_soft_attention", False) and raw_content.ndim == 3:
        model.transformer.use_soft_attention = True
        model.transformer.soft_attention_tau = model.soft_attention_tau

    (_inter_states, _init_ref, _inter_refs, enc_state, enc_ref, _apr) = model.transformer(
        multi_level_feats,
        multi_level_masks,
        multi_level_pe,
        (None, None),
        attn_masks=[None, None],
        content_query_embeds=raw_content,
        content_inds=None,
    )

    # Encoder-side classifier is the last entry of class_embed (shared via deepcopy at init).
    enc_logits = model.transformer.decoder.class_embed[-1](enc_state, content_inds=None)
    return enc_logits, enc_ref, images.image_sizes, (H_pad, W_pad)


def _load_class_groups(path: Optional[str]) -> Optional[Dict[int, str]]:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        logger.warning(f"--class-groups {p} does not exist, ignoring.")
        return None
    with p.open() as f:
        raw = json.load(f)
    return {int(k): str(v) for k, v in raw.items()}


def _resolve_bucket(
    cls_id: int,
    base_mask: Optional[torch.Tensor],
    class_groups: Optional[Dict[int, str]],
) -> str:
    if class_groups is not None:
        return class_groups.get(cls_id, "unknown")
    if base_mask is not None:
        return "base" if bool(base_mask[cls_id].item()) else "novel"
    return "all"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--out", default="analysis/encoder_topk.json")
    parser.add_argument("--num-images", type=int, default=500,
                        help="Stop after this many images. -1 for whole set.")
    parser.add_argument("--iou-thr", type=float, default=0.5,
                        help="GT<->proposal match threshold.")
    parser.add_argument("--topk", type=int, default=5,
                        help="K in top-K hit rate (hit iff gt class is in proposal's top-K).")
    parser.add_argument("--class-groups", type=str, default=None,
                        help="Optional JSON dict: {\"<class_id>\": \"frequent|common|rare|...\"}")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--log-interval", type=int, default=50)
    args = parser.parse_args()

    setup_logger(name="diag")
    logger.setLevel(logging.INFO)

    logger.info(f"Loading config: {args.config}")
    cfg = LazyConfig.load(args.config)

    logger.info("Instantiating model...")
    model = instantiate(cfg.model)
    model.to(args.device).eval()

    logger.info(f"Loading checkpoint: {args.ckpt}")
    DetectionCheckpointer(model).load(args.ckpt)

    logger.info("Building eval dataloader from cfg.dataloader.test ...")
    dataloader = instantiate(cfg.dataloader.test)

    # Resolve stratification.
    base_mask = None
    if hasattr(model, "base_idx") and isinstance(model.base_idx, torch.Tensor):
        base_mask = model.base_idx.to(args.device)
        logger.info(f"model.base_idx found: {int(base_mask.sum())} base / "
                    f"{int((~base_mask).sum())} novel classes.")
    class_groups = _load_class_groups(args.class_groups)
    if class_groups is not None:
        logger.info(f"Loaded class-groups for {len(class_groups)} classes from {args.class_groups}")

    buckets: Dict[str, Dict[str, int]] = defaultdict(
        lambda: {"gt_total": 0, "matched": 0, "top1_hit": 0, "topk_hit": 0}
    )

    processed = 0
    num_gt_seen = 0
    limit = args.num_images if args.num_images > 0 else float("inf")

    for batch_idx, batch in enumerate(dataloader):
        if processed >= limit:
            break
        try:
            enc_logits, enc_ref, image_sizes, (H_pad, W_pad) = run_encoder_only(model, batch)
        except Exception as e:
            logger.warning(f"Skip batch {batch_idx}, exception: {type(e).__name__}: {e}")
            continue

        B = enc_logits.shape[0]
        enc_probs = enc_logits.sigmoid()  # focal-style; monotonic w.r.t. logits for ranking

        for b in range(B):
            if processed >= limit:
                break
            processed += 1

            inp = batch[b]
            gt_inst = inp.get("instances", None)
            if gt_inst is None or len(gt_inst) == 0:
                continue

            # GT boxes are in the post-resize (pre-pad) coords: [0, img_h] x [0, img_w].
            # Encoder proposals are normalized to the padded image (H_pad, W_pad).
            # Padding is at bottom-right, so they share the same top-left origin and
            # we can bring proposals to absolute-padded coords with (W_pad, H_pad).
            gt_boxes = gt_inst.gt_boxes.tensor.to(enc_logits.device).float()  # xyxy abs
            gt_classes = gt_inst.gt_classes.to(enc_logits.device).long()

            ref_b = enc_ref[b]  # [topk, 4] normalized cxcywh
            pred_xyxy = box_cxcywh_to_xyxy(ref_b)
            scale = torch.tensor([W_pad, H_pad, W_pad, H_pad],
                                 dtype=pred_xyxy.dtype, device=pred_xyxy.device)
            pred_xyxy = pred_xyxy * scale

            ious = pairwise_iou_xyxy(gt_boxes, pred_xyxy)  # [num_gt, topk_prop]
            best_iou, best_idx = ious.max(dim=1)

            probs_b = enc_probs[b]  # [topk_prop, C]
            top1 = probs_b.argmax(dim=-1)  # [topk_prop]
            _, topk_cls = probs_b.topk(args.topk, dim=-1)  # [topk_prop, K]

            for g in range(gt_classes.numel()):
                gt_c = int(gt_classes[g].item())
                bucket = _resolve_bucket(gt_c, base_mask, class_groups)
                buckets[bucket]["gt_total"] += 1
                buckets["__all__"]["gt_total"] += 1
                num_gt_seen += 1

                if best_iou[g].item() < args.iou_thr:
                    continue

                buckets[bucket]["matched"] += 1
                buckets["__all__"]["matched"] += 1

                p = int(best_idx[g].item())
                if int(top1[p].item()) == gt_c:
                    buckets[bucket]["top1_hit"] += 1
                    buckets["__all__"]["top1_hit"] += 1
                if gt_c in topk_cls[p].tolist():
                    buckets[bucket]["topk_hit"] += 1
                    buckets["__all__"]["topk_hit"] += 1

        if processed % args.log_interval == 0 or processed >= limit:
            logger.info(f"  processed {processed} images, {num_gt_seen} GT boxes")

    # Compose report.
    report: Dict[str, Dict[str, Any]] = {}
    for bk, cnt in buckets.items():
        total = max(cnt["gt_total"], 1)
        matched = max(cnt["matched"], 1)
        report[bk] = {
            **cnt,
            "match_rate": cnt["matched"] / total,
            "top1_hit_rate_over_matched": cnt["top1_hit"] / matched,
            f"top{args.topk}_hit_rate_over_matched": cnt["topk_hit"] / matched,
            "top1_hit_rate_over_all_gt": cnt["top1_hit"] / total,
            f"top{args.topk}_hit_rate_over_all_gt": cnt["topk_hit"] / total,
        }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "config": args.config,
        "ckpt": args.ckpt,
        "num_images_processed": processed,
        "num_gt_total": num_gt_seen,
        "iou_thr": args.iou_thr,
        "topk": args.topk,
        "per_bucket": report,
    }
    with out_path.open("w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Saved report -> {out_path}")

    # Plain-text summary.
    header = (
        f"{'Bucket':<12} {'GT':>7} {'Match%':>8} "
        f"{'Top1|M%':>9} {f'Top{args.topk}|M%':>9} "
        f"{'Top1|all%':>10} {f'Top{args.topk}|all%':>10}"
    )
    print("\n=== Encoder top-k diagnostic ===")
    print(f"config={args.config}")
    print(f"ckpt={args.ckpt}")
    print(f"images={processed}  GT={num_gt_seen}  IoU>={args.iou_thr}")
    print(header)
    print("-" * len(header))
    for bk in sorted(report.keys(), key=lambda s: (s != "__all__", s)):
        r = report[bk]
        print(
            f"{bk:<12} {r['gt_total']:>7d} "
            f"{r['match_rate'] * 100:>7.2f}% "
            f"{r['top1_hit_rate_over_matched'] * 100:>8.2f}% "
            f"{r[f'top{args.topk}_hit_rate_over_matched'] * 100:>8.2f}% "
            f"{r['top1_hit_rate_over_all_gt'] * 100:>9.2f}% "
            f"{r[f'top{args.topk}_hit_rate_over_all_gt'] * 100:>9.2f}%"
        )


if __name__ == "__main__":
    sys.exit(main())
