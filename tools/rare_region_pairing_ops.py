"""Score saved FP/TP regions with each checkpoint's native cached queries."""

from __future__ import annotations

from collections import defaultdict
import math

import torch
import torch.nn.functional as F

from lami_dino.diagnostic_ops import fuse_detector_vlm_scores, pairwise_box_iou_xyxy
from lami_dino.pairing_diagnostic_ops import replay_classifier


def collect_regions(details, names, iou_key="0.50"):
    """Select new leading FP plus both checkpoints' TP controls by saved identity."""
    regions, seen = [], set()
    for name in names:
        if name not in details["classes"]:
            raise ValueError(f"Missing FP detail category: {name}")
        for side, kind, field in (
            ("new", "fp", "leading_false_positives"),
            ("new", "tp", "true_positives"),
            ("old", "tp", "true_positives"),
        ):
            for row in details["classes"][name][side][iou_key][field]:
                identity = (name, side, row["image_id"], row["detection_id"])
                if identity in seen:
                    continue
                seen.add(identity)
                box = row["bbox"]
                if len(box) != 4 or not all(math.isfinite(v) for v in box) or min(box[2:]) <= 0:
                    raise ValueError("FP/TP region must have a finite positive xywh box")
                if not math.isfinite(row["score"]) or row["score"] < 0:
                    raise ValueError("Saved score must be finite and nonnegative")
                regions.append({
                    "category": name, "source_side": side, "kind": kind,
                    "image_id": row["image_id"], "detection_id": row["detection_id"],
                    "class_global_rank": row["rank"], "score": row["score"],
                    "box_xyxy": [box[0], box[1], box[0] + box[2], box[1] + box[3]],
                    "matched_gt_id": row.get("matched_gt_id"),
                    "overlap_label": row.get("overlap_label"),
                    "nearest_other_labeled_gt": row.get("nearest_other_labeled_gt"),
                })
    return regions


def fp_overlap_report(regions):
    groups = defaultdict(list)
    for region in regions:
        if region["kind"] == "fp" and region["source_side"] == "new":
            groups[region["category"], region["image_id"]].append(region)
    report = []
    for (category, image_id), rows in sorted(groups.items()):
        boxes = torch.tensor([row["box_xyxy"] for row in rows], dtype=torch.float64)
        ious = pairwise_box_iou_xyxy(boxes, boxes)
        pairs = [{"detection_ids": [rows[i]["detection_id"], rows[j]["detection_id"]],
                  "iou": float(ious[i, j])}
                 for i in range(len(rows)) for j in range(i + 1, len(rows))]
        report.append({
            "category": category, "image_id": image_id,
            "detection_ids": [row["detection_id"] for row in rows],
            "boxes_xyxy": boxes.tolist(), "iou_matrix": ious.tolist(), "pairs": pairs,
            "pairs_iou_ge_050": sum(pair["iou"] >= .5 for pair in pairs),
            "pairs_iou_ge_075": sum(pair["iou"] >= .75 for pair in pairs),
            "maximum_pair_iou": max((pair["iou"] for pair in pairs), default=None),
        })
    return report


@torch.no_grad()
def replay_sample(sample, bank, protocol, device="cpu"):
    features = sample["features"].float().to(device)
    boxes = sample["query_boxes"].float().to(device)
    roi = sample["roi_features"].float().to(device)
    if features.shape[0] != roi.shape[0] or boxes.shape != (features.shape[0], 4):
        raise ValueError("Cached query/ROI/box counts disagree")
    logits = replay_classifier(
        features, bank["prototypes"].to(device), temperature=bank["temperature"],
        logit_scale=bank["logit_scale"], cls_bias=bank["cls_bias"], query_chunk_size=128,
    )
    clip_logits = roi @ bank["vlm_text"].float().to(device).t() * bank["vlm_temperature"]
    log_scores = fuse_detector_vlm_scores(
        logits, clip_logits, bank["novel_mask"].to(device),
        base_weight=protocol["alpha"], novel_weight=protocol["beta"],
        novel_scale=protocol["novel_scale"],
    )
    scores = log_scores.exp()
    if not torch.isfinite(scores).all() or not torch.isfinite(boxes).all():
        raise ValueError("Nonfinite replay scores/boxes")
    count = min(protocol["max_dets"], scores.numel())
    top = scores.flatten().topk(count)
    selected = torch.zeros(scores.numel(), dtype=torch.bool, device=scores.device)
    selected[top.indices] = True
    return {
        "boxes": boxes, "det_logits": logits, "clip_logits": clip_logits,
        "det_logp": F.logsigmoid(logits), "clip_logp": F.log_softmax(clip_logits, dim=-1),
        "scores": scores, "log_scores": log_scores,
        "selected": selected.reshape_as(scores),
        "cutoff": float(top.values[-1]) if count else None,
        "category_ids": bank["category_ids"], "novel_mask": bank["novel_mask"],
    }


def query_components(replay, query_id, class_index, categories, region_iou):
    det = replay["det_logits"][query_id]
    clip = replay["clip_logits"][query_id]
    det_top, clip_top = int(det.argmax()), int(clip.argmax())
    score = float(replay["scores"][query_id, class_index])
    ids = replay["category_ids"]
    return {
        "query_id": query_id, "box_xyxy": replay["boxes"][query_id].tolist(),
        "region_iou": float(region_iou),
        "detector_probability": float(det[class_index].sigmoid()),
        "clip_probability": float(clip.softmax(-1)[class_index]),
        "detector_log_probability": float(replay["det_logp"][query_id, class_index]),
        "clip_log_probability": float(replay["clip_logp"][query_id, class_index]),
        "detector_rank": 1 + int((det > det[class_index]).sum()),
        "clip_rank": 1 + int((clip > clip[class_index]).sum()),
        "detector_top1": categories[ids[det_top]]["name"],
        "clip_top1": categories[ids[clip_top]]["name"],
        "fused_score": score, "fused_log_score": float(replay["log_scores"][query_id, class_index]),
        "in_image_topk": bool(replay["selected"][query_id, class_index]),
        "image_topk_threshold": replay["cutoff"],
        "score_threshold_ratio": score / replay["cutoff"] if replay["cutoff"] else None,
    }


def compare_region(region, replays, categories, protocol, *, match_iou=.5,
                   box_atol=.05, score_atol=5e-5):
    """Recover source detection by box AND score; pair the other model geometrically.

    Detector queries cannot be evaluated at an arbitrary identical ROI using
    this cache. The other model's nearest box and highest-scoring eligible box
    are both reported. Exact source ambiguities are retained, never hidden.
    """
    source_side = region["source_side"]
    other_side = "old" if source_side == "new" else "new"
    source, other = replays[source_side], replays[other_side]
    category_id = next(key for key, row in categories.items() if row["name"] == region["category"])
    class_index = source["category_ids"].index(category_id)
    if other["category_ids"] != source["category_ids"]:
        raise ValueError("Old/new class ordering differs")
    target = source["boxes"].new_tensor(region["box_xyxy"])
    box_error = (source["boxes"] - target).abs().amax(-1)
    score_error = (source["scores"][:, class_index] - region["score"]).abs()
    matches = torch.where((box_error <= box_atol) & (score_error <= score_atol))[0]
    if matches.numel() == 0:
        raise ValueError(
            f"Saved detection does not match source replay: {source_side} "
            f"{region['category']} image={region['image_id']} det={region['detection_id']}; "
            "check checkpoint, mapper, protocol and cache provenance"
        )
    source_ious = pairwise_box_iou_xyxy(target[None], source["boxes"])[0]
    components = [query_components(source, int(q), class_index, categories, source_ious[q]) for q in matches]
    for q, item in zip(matches, components):
        item["saved_box_max_abs_error"] = float(box_error[q])
        item["saved_score_abs_error"] = float(score_error[q])
    other_ious = pairwise_box_iou_xyxy(target.to(other["boxes"].device)[None], other["boxes"])[0]
    nearest_id = int(other_ious.argmax()) if other_ious.numel() else None
    eligible = torch.where(other_ious >= match_iou)[0]
    best_id = int(eligible[other["scores"][eligible, class_index].argmax()]) if eligible.numel() else None
    nearest = (query_components(other, nearest_id, class_index, categories, other_ious[nearest_id])
               if nearest_id is not None else None)
    best = (query_components(other, best_id, class_index, categories, other_ious[best_id])
            if best_id is not None else None)
    labeled = region.get("nearest_other_labeled_gt")
    if labeled is not None and labeled["iou"] >= match_iou:
        competitor_id = labeled["category_id"]
        competitor_index = source["category_ids"].index(competitor_id)
        for replay, entries in ((source, components), (other, [nearest, best])):
            for entry in entries:
                if entry is None:
                    continue
                q = entry["query_id"]
                entry["overlapping_annotation_class"] = {
                    "category_id": competitor_id, "name": categories[competitor_id]["name"],
                    "detector_probability": float(replay["det_logp"][q, competitor_index].exp()),
                    "clip_probability": float(replay["clip_logp"][q, competitor_index].exp()),
                }
    # Algebraic decomposition, explicitly conditional on geometrically paired
    # native boxes. It does not isolate a causal feature/classifier effect.
    contributions = None
    if len(components) == 1 and best is not None:
        native = components[0]
        old, new = (best, native) if source_side == "new" else (native, best)
        weight = protocol["beta"] if bool(source["novel_mask"][class_index]) else protocol["alpha"]
        det_delta = (1 - weight) * (new["detector_log_probability"] - old["detector_log_probability"])
        clip_delta = weight * (new["clip_log_probability"] - old["clip_log_probability"])
        total = new["fused_log_score"] - old["fused_log_score"]
        contributions = {"detector": det_delta, "clip": clip_delta, "total": total,
                         "closure_abs_error": abs(det_delta + clip_delta - total)}
        if contributions["closure_abs_error"] > 1e-5:
            raise ValueError("Weighted log-score difference does not close")
    return {
        **region, "source_query_identity": "unique" if len(components) == 1 else "ambiguous",
        "source_candidates": components, "other_side": other_side,
        "other_nearest_box": nearest, "other_eligible_query_count": int(eligible.numel()),
        "source_retained_query_count": int(((source_ious >= match_iou) & source["selected"][:, class_index]).sum()),
        "other_retained_query_count": int(((other_ious >= match_iou) & other["selected"][:, class_index]).sum()),
        "other_best_score_box": best, "match_iou_threshold": match_iou,
        "new_minus_old_log_score_components": contributions,
    }
