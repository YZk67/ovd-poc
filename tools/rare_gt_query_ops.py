"""Single-GT coverage before selection; no training or AP estimation."""

from __future__ import annotations

import math

import torch

from lami_dino.diagnostic_ops import pairwise_box_iou_xyxy
from tools.rare_region_pairing_ops import query_components


def selected_predictions(replay, image_id):
    """Native all-class top-k first, then remove empty clipped boxes; no refill."""
    rows = []
    for query, category in replay["selected"].nonzero().tolist():
        x0, y0, x1, y1 = replay["boxes"][query].tolist()
        if x1 <= x0 or y1 <= y0:
            continue
        rows.append({"image_id": image_id, "category_id": replay["category_ids"][category],
                     "bbox": [x0, y0, x1 - x0, y1 - y0],
                     "score": float(replay["scores"][query, category]), "query_id": query})
    return rows


def verify_predictions(actual, saved, *, box_atol=.05, score_atol=5e-5):
    """Match the complete prediction multiset, including ties/duplicate boxes.

    A maximum bipartite matching avoids a greedy ambiguous duplicate consuming
    the only match available to another row. Never match query IDs across models.
    """
    if len(actual) != len(saved):
        raise ValueError(f"Saved/native prediction count mismatch: {len(saved)} vs {len(actual)}")
    for row in actual + saved:
        if (len(row["bbox"]) != 4 or min(row["bbox"][2:]) <= 0
                or not all(math.isfinite(v) for v in row["bbox"] + [row["score"]])
                or row["score"] < 0):
            raise ValueError("Invalid saved/native prediction")
    edges = []
    for row in actual:
        edges.append([j for j, ref in enumerate(saved)
                      if (row["image_id"], row["category_id"]) == (ref["image_id"], ref["category_id"])
                      and abs(row["score"] - ref["score"]) <= score_atol
                      and max(abs(a-b) for a, b in zip(row["bbox"], ref["bbox"])) <= box_atol])
    assigned = {}

    def augment(i, visited):
        for j in edges[i]:
            if j in visited:
                continue
            visited.add(j)
            if j not in assigned or augment(assigned[j], visited):
                assigned[j] = i
                return True
        return False

    for i in sorted(range(len(actual)), key=lambda n: len(edges[n])):
        if not augment(i, set()):
            raise ValueError("Cannot reproduce ALL saved top-300 predictions. Stop: check checkpoint, "
                             "mapper, assets and inference protocol; no miss diagnosis is valid yet.")
    pairs = [(actual[i], saved[j]) for j, i in assigned.items()]
    return {"all_predictions_reproduced": True, "matched": len(pairs),
            "box_atol_pixels": box_atol, "score_atol": score_atol,
            "max_box_error": max((abs(a-b) for x, y in pairs
                                   for a, b in zip(x["bbox"], y["bbox"])), default=0.),
            "max_score_error": max((abs(x["score"]-y["score"]) for x, y in pairs), default=0.),
            "ambiguous_rows": sum(len(e) > 1 for e in edges)}


@torch.no_grad()
def analyze_queries(replay, gt, categories, thresholds=(.5, .75, .9)):
    boxes, scores = replay["boxes"], replay["scores"]
    if not len(boxes) or not torch.isfinite(boxes).all() or not torch.isfinite(scores).all():
        raise ValueError("Empty/nonfinite raw query bank")
    category = replay["category_ids"].index(gt["category_id"])
    x, y, w, h = gt["bbox"]
    ious = pairwise_box_iou_xyxy(boxes.new_tensor([[x, y, x+w, y+h]]), boxes)[0]
    positive_box = (boxes[:, 2:] > boxes[:, :2]).all(-1)
    kept = replay["selected"] & positive_box[:, None]
    # Sort once, rather than rescanning Q*C scores for every query.
    sorted_scores = scores.flatten().sort().values.contiguous()
    true_scores = scores[:, category].contiguous()
    first = scores.numel() - torch.searchsorted(sorted_scores, true_scores, right=True) + 1
    last = scores.numel() - torch.searchsorted(sorted_scores, true_scores, right=False)
    rows = []
    for q in range(len(boxes)):
        row = query_components(replay, q, category, categories, ious[q])
        row.update({"image_pair_rank_interval": [int(first[q]), int(last[q])],
                    "any_class_in_saved_topk": bool(kept[q].any()),
                    "true_class_in_saved_topk": bool(kept[q, category]),
                    "retained_category_ids": [replay["category_ids"][c]
                                              for c in kept[q].nonzero().flatten().tolist()]})
        rows.append(row)
    by_iou = {}
    for threshold in thresholds:
        eligible = (ious >= threshold) & positive_box
        ids = eligible.nonzero().flatten()
        best = int(ids[true_scores[ids].argmax()]) if len(ids) else None
        correct = int(kept[eligible, category].sum())
        any_query = int(kept[eligible].any(-1).sum())
        reason = ("no_eligible_final_query_box" if best is None else
                  "eligible_true_class_retained" if correct else
                  "eligible_query_retained_under_other_categories" if any_query else
                  "eligible_queries_all_excluded_from_top300")
        by_iou[f"{threshold:.2f}"] = {
            "reason": reason, "raw_eligible_queries": len(ids),
            "retained_any_class_queries": any_query, "retained_any_class_pairs": int(kept[eligible].sum()),
            "retained_true_class_queries": correct,
            "best_fused_true_class_eligible": rows[best] if best is not None else None,
            "best_detector_true_class_eligible": rows[int(ids[replay["det_logits"][ids, category].argmax()])]
            if best is not None else None,
            "best_clip_true_class_eligible": rows[int(ids[replay["clip_logp"][ids, category].argmax()])]
            if best is not None else None,
        }
    return {"raw_queries": len(rows), "vocabulary_size": scores.shape[1],
            "image_cutoff": replay["cutoff"], "best_iou_query": rows[int(ious.argmax())],
            "by_iou": by_iou, "all_queries": rows}
