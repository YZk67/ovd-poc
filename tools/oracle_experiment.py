"""Oracle experiment: test if better HAUF signals can improve unknown detection.

Three modes:
  A) Baseline: original HAUF (att_score + uncertainty)
  B) Oracle-Uncertainty: replace uncertainty with GT-based perfect signal
  C) Oracle-Full: replace both att_score and uncertainty with GT-based signals

If C >> A, there's room for VLM distillation.
If C ≈ A, HAUF is already near ceiling, VLM won't help.

Usage:
    python tools/oracle_experiment.py \
        --config-file lami_dino/configs/dino_convnext_large_owovd.py \
        --checkpoint output/dino_convnext_large_owovd/model_0005949.pth

The script runs inference once, saves intermediate features, then evaluates
all three oracle modes offline (no re-inference needed).
"""

import argparse
import json
import logging
import os
import sys
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from pycocotools.coco import COCO

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from detectron2.config import LazyConfig, instantiate
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import MetadataCatalog
from detectron2.structures import Boxes, Instances

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Oracle experiment for HAUF")
    parser.add_argument("--config-file", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", default="output/oracle_experiment")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def load_model(cfg, checkpoint_path, device):
    """Load model with HAUF enabled."""
    # Force enable HAUF and score_ensemble (needed for ROI feature extraction)
    cfg.model.hauf_enabled = True
    cfg.model.hauf_att_path = "dataset/metadata/coco_vsas_selected_v2.pth"
    cfg.model.backbone.score_ensemble = True
    cfg.model.score_ensemble = False  # We use HAUF path, not score_ensemble path

    model = instantiate(cfg.model)
    model.to(device)
    model.eval()

    checkpointer = DetectionCheckpointer(model)
    checkpointer.load(checkpoint_path)
    return model


def load_gt_info(cfg):
    """Load GT annotations and build unknown/known category maps."""
    test_dataset = cfg.dataloader.test.dataset.names
    metadata = MetadataCatalog.get(test_dataset)
    json_file = metadata.json_file
    coco_gt = COCO(json_file)

    # Get known cat IDs from JSON info
    with open(json_file) as f:
        test_info = json.load(f).get("info", {})
    known_cat_ids_orig = test_info.get("known_cat_ids", [])

    id_map = getattr(metadata, "thing_dataset_id_to_contiguous_id", {})
    known_cat_ids = set(id_map.get(cid, cid) for cid in known_cat_ids_orig)

    all_cat_ids = set(id_map.get(cid, cid) for cid in coco_gt.getCatIds())
    unknown_cat_ids = all_cat_ids - known_cat_ids

    # Build per-image GT: {image_id: [(box_xyxy, is_unknown, contiguous_cat_id), ...]}
    gt_by_image = defaultdict(list)
    for ann in coco_gt.loadAnns(coco_gt.getAnnIds()):
        if ann.get("iscrowd", 0):
            continue
        x, y, w, h = ann["bbox"]
        box = [x, y, x + w, y + h]
        cat_id_orig = ann["category_id"]
        cat_id = id_map.get(cat_id_orig, cat_id_orig)
        is_unknown = cat_id in unknown_cat_ids
        gt_by_image[ann["image_id"]].append({
            "box": box,
            "is_unknown": is_unknown,
            "cat_id": cat_id,
        })

    return coco_gt, known_cat_ids, unknown_cat_ids, gt_by_image, test_dataset


def compute_iou_matrix(boxes_a, boxes_b):
    """Compute IoU between two sets of boxes. [N,4] x [M,4] -> [N,M]"""
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.zeros((len(boxes_a), len(boxes_b)))

    boxes_a = np.array(boxes_a)
    boxes_b = np.array(boxes_b)

    x1 = np.maximum(boxes_a[:, 0:1], boxes_b[:, 0:1].T)
    y1 = np.maximum(boxes_a[:, 1:2], boxes_b[:, 1:2].T)
    x2 = np.minimum(boxes_a[:, 2:3], boxes_b[:, 2:3].T)
    y2 = np.minimum(boxes_a[:, 3:4], boxes_b[:, 3:4].T)

    inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    area_a = (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter

    return inter / np.maximum(union, 1e-8)


def run_inference_and_collect(model, dataloader, gt_by_image, device):
    """Run inference with HAUF, collect predictions + GT matching info.

    Uses monkey-patching on HAUF.forward to capture intermediate signals
    without modifying the model's forward pass.
    """
    hauf = model.hauf_module
    all_results = []
    collected_signals = {}

    # Patch HAUF to capture intermediate signals
    original_forward = hauf.forward.__func__  # unbound method

    def patched_forward(self, known_cls_scores, roi_features, att_emb, temperature=100.0):
        att_scores = self.compute_attribute_scores(roi_features, att_emb, temperature)
        top_k_att = self.compute_weighted_top_k(att_scores)
        uncertainty = self.compute_uncertainty(known_cls_scores)
        max_known, _ = known_cls_scores.max(dim=-1, keepdim=True)

        collected_signals["known_cls_scores"] = known_cls_scores.cpu()
        collected_signals["top_k_att"] = top_k_att.cpu()
        collected_signals["uncertainty"] = uncertainty.cpu()
        collected_signals["max_known"] = max_known.cpu()

        # Compute original result
        w = self.att_weight
        unknown_scores = (w * top_k_att + (1 - w) * uncertainty) * (1.0 - max_known)
        combined_scores = torch.cat([known_cls_scores, unknown_scores], dim=-1)
        return unknown_scores, combined_scores

    # Apply patch
    import types
    hauf.forward = types.MethodType(patched_forward, hauf)

    for batch_idx, batched_inputs in enumerate(dataloader):
        if batch_idx % 500 == 0:
            print(f"Processing batch {batch_idx}/{len(dataloader)}...")

        collected_signals.clear()

        with torch.no_grad():
            outputs = model(batched_inputs)

        # Process each image in the batch
        for i, (inp, out) in enumerate(zip(batched_inputs, outputs)):
            image_id = inp["image_id"]
            instances = out["instances"]

            pred_boxes = instances.pred_boxes.tensor.cpu().numpy()
            pred_scores = instances.scores.cpu().numpy()
            pred_classes = instances.pred_classes.cpu().numpy()

            # GT matching
            gt_entries = gt_by_image.get(image_id, [])
            gt_boxes = [g["box"] for g in gt_entries]
            gt_is_unk = [g["is_unknown"] for g in gt_entries]

            pred_matched_unknown = np.zeros(len(pred_boxes), dtype=bool)
            pred_matched_known = np.zeros(len(pred_boxes), dtype=bool)

            if len(gt_boxes) > 0 and len(pred_boxes) > 0:
                iou_matrix = compute_iou_matrix(pred_boxes, gt_boxes)
                for pred_idx in range(len(pred_boxes)):
                    best_gt = iou_matrix[pred_idx].argmax()
                    if iou_matrix[pred_idx, best_gt] >= 0.5:
                        if gt_is_unk[best_gt]:
                            pred_matched_unknown[pred_idx] = True
                        else:
                            pred_matched_known[pred_idx] = True

            all_results.append({
                "image_id": image_id,
                "pred_boxes": pred_boxes,
                "pred_scores": pred_scores,
                "pred_classes": pred_classes,
                "matched_unknown": pred_matched_unknown,
                "matched_known": pred_matched_known,
            })

    # Restore original forward
    hauf.forward = types.MethodType(original_forward, hauf)

    return all_results


def evaluate_unknown_detection(predictions, unknown_class_id, gt_by_image, unknown_cat_ids, mode="baseline"):
    """Evaluate U-Recall and U-mAP for a set of predictions.

    Args:
        predictions: list of dicts with boxes/scores/classes per image.
        mode: label for logging.
    """
    # U-Recall
    total_unknown_gt = 0
    recalled = 0

    # U-mAP
    all_scores = []
    all_tp = []

    for pred in predictions:
        image_id = pred["image_id"]
        gt_entries = gt_by_image.get(image_id, [])
        gt_unknown_boxes = [g["box"] for g in gt_entries if g["is_unknown"]]
        total_unknown_gt += len(gt_unknown_boxes)

        unknown_mask = pred["classes"] == unknown_class_id
        if not unknown_mask.any() or not gt_unknown_boxes:
            # Still need to count false positives for U-mAP
            if unknown_mask.any():
                for s in pred["scores"][unknown_mask]:
                    all_scores.append(s)
                    all_tp.append(0)
            continue

        pred_boxes = pred["boxes"][unknown_mask]
        scores = pred["scores"][unknown_mask]
        gt_boxes_arr = np.array(gt_unknown_boxes)

        matched_gt = set()
        order = scores.argsort()[::-1]
        for idx in order:
            ious = compute_iou_matrix(pred_boxes[idx:idx+1], gt_boxes_arr)[0]
            best_gt = ious.argmax()
            all_scores.append(scores[idx])
            if ious[best_gt] >= 0.5 and best_gt not in matched_gt:
                matched_gt.add(best_gt)
                all_tp.append(1)
                recalled += 1
            else:
                all_tp.append(0)

    u_recall = recalled / max(total_unknown_gt, 1)

    # VOC 11-point AP
    u_map = 0.0
    if all_scores:
        order = np.argsort(-np.array(all_scores))
        tp = np.array(all_tp)[order]
        tp_cum = tp.cumsum()
        fp_cum = (1 - tp).cumsum()
        recall_curve = tp_cum / max(total_unknown_gt, 1)
        precision_curve = tp_cum / (tp_cum + fp_cum)
        for t in np.arange(0, 1.1, 0.1):
            mask = recall_curve >= t
            if mask.any():
                u_map += precision_curve[mask].max() / 11.0

    return {
        "mode": mode,
        "U-Recall": u_recall,
        "U-mAP": u_map,
        "total_unknown_gt": total_unknown_gt,
        "total_unknown_pred": sum(
            (p["classes"] == unknown_class_id).sum() for p in predictions
        ),
    }


def oracle_rescore(all_results, gt_by_image, unknown_class_id=80, mode="oracle_full"):
    """Re-score predictions using oracle (GT) information.

    Modes:
    - baseline: use original HAUF scores (no change)
    - oracle_uncertainty: replace uncertainty with GT-based signal
      (1.0 if proposal matches unknown GT, 0.0 otherwise)
    - oracle_full: replace both att_score and uncertainty with GT-based signal
      (unknown_score = 1.0 if matches unknown GT, 0.0 if matches known, 0.0 if no match)
    """
    rescored_predictions = []

    for result in all_results:
        image_id = result["image_id"]
        pred_boxes = result["pred_boxes"]
        pred_scores = result["pred_scores"]
        pred_classes = result["pred_classes"].copy()

        if mode == "baseline":
            # No change
            rescored_predictions.append({
                "image_id": image_id,
                "boxes": pred_boxes,
                "scores": pred_scores,
                "classes": pred_classes,
            })
            continue

        # For oracle modes: re-match predictions to GT and override unknown scores
        gt_entries = gt_by_image.get(image_id, [])
        gt_boxes = [g["box"] for g in gt_entries]
        gt_is_unk = [g["is_unknown"] for g in gt_entries]

        new_scores = pred_scores.copy()
        new_classes = pred_classes.copy()

        if len(gt_boxes) > 0 and len(pred_boxes) > 0:
            iou_matrix = compute_iou_matrix(pred_boxes, gt_boxes)

            for pred_idx in range(len(pred_boxes)):
                best_gt = iou_matrix[pred_idx].argmax()
                best_iou = iou_matrix[pred_idx, best_gt]

                if mode == "oracle_uncertainty":
                    # If prediction overlaps unknown GT → force unknown class
                    if best_iou >= 0.5 and gt_is_unk[best_gt]:
                        new_classes[pred_idx] = unknown_class_id
                        new_scores[pred_idx] = max(new_scores[pred_idx], 0.9)

                elif mode == "oracle_full":
                    # Perfect oracle: anything matching unknown GT → unknown class
                    # anything matching known GT → keep as is
                    # anything not matching → keep as is
                    if best_iou >= 0.5 and gt_is_unk[best_gt]:
                        new_classes[pred_idx] = unknown_class_id
                        new_scores[pred_idx] = 0.95
                    elif best_iou >= 0.5 and not gt_is_unk[best_gt]:
                        # Keep known class, slightly boost confidence
                        pass
                    else:
                        # No match - if currently unknown, reduce score
                        if new_classes[pred_idx] == unknown_class_id:
                            new_scores[pred_idx] *= 0.1  # suppress false unknown

                elif mode == "oracle_att_only":
                    # Only replace attribute signal: boost unknown score for
                    # proposals that overlap unknown GT
                    if best_iou >= 0.5 and gt_is_unk[best_gt]:
                        # If not already unknown, make it unknown with moderate score
                        if new_classes[pred_idx] != unknown_class_id:
                            new_classes[pred_idx] = unknown_class_id
                            new_scores[pred_idx] = 0.5

        rescored_predictions.append({
            "image_id": image_id,
            "boxes": pred_boxes,
            "scores": new_scores,
            "classes": new_classes,
        })

    return rescored_predictions


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO)

    os.makedirs(args.output_dir, exist_ok=True)

    # Load config
    cfg = LazyConfig.load(args.config_file)

    # Load GT info
    print("Loading GT annotations...")
    coco_gt, known_cat_ids, unknown_cat_ids, gt_by_image, test_dataset = load_gt_info(cfg)
    print(f"Known classes: {len(known_cat_ids)}, Unknown classes: {len(unknown_cat_ids)}")

    # Load model
    print("Loading model...")
    model = load_model(cfg, args.checkpoint, args.device)

    # Run inference and collect HAUF signals
    print("Running inference...")
    dataloader = instantiate(cfg.dataloader.test)
    all_results = run_inference_and_collect(model, dataloader, gt_by_image, args.device)

    # Save intermediate results
    print(f"Collected {len(all_results)} images. Saving...")
    # Don't save full results (too large), just save what we need for re-scoring
    lightweight_results = [{
        "image_id": r["image_id"],
        "pred_boxes": r["pred_boxes"],
        "pred_scores": r["pred_scores"],
        "pred_classes": r["pred_classes"],
        "matched_unknown": r["matched_unknown"],
        "matched_known": r["matched_known"],
    } for r in all_results]
    torch.save(lightweight_results, os.path.join(args.output_dir, "inference_results.pth"))

    # Evaluate all modes
    print("\n" + "=" * 60)
    print("ORACLE EXPERIMENT RESULTS")
    print("=" * 60)

    modes = ["baseline", "oracle_uncertainty", "oracle_att_only", "oracle_full"]
    all_metrics = {}

    for mode in modes:
        rescored = oracle_rescore(all_results, gt_by_image, unknown_class_id=80, mode=mode)
        metrics = evaluate_unknown_detection(
            rescored, unknown_class_id=80,
            gt_by_image=gt_by_image,
            unknown_cat_ids=unknown_cat_ids,
            mode=mode,
        )
        all_metrics[mode] = metrics

        print(f"\n[{mode}]")
        print(f"  U-Recall: {metrics['U-Recall']:.4f}")
        print(f"  U-mAP:    {metrics['U-mAP']:.4f}")
        print(f"  Unknown GT: {metrics['total_unknown_gt']}")
        print(f"  Unknown Pred: {metrics['total_unknown_pred']}")

    # Analysis
    print("\n" + "=" * 60)
    print("ANALYSIS")
    print("=" * 60)

    baseline = all_metrics["baseline"]
    oracle_full = all_metrics["oracle_full"]
    oracle_unc = all_metrics["oracle_uncertainty"]
    oracle_att = all_metrics["oracle_att_only"]

    gap_full = oracle_full["U-mAP"] - baseline["U-mAP"]
    gap_unc = oracle_unc["U-mAP"] - baseline["U-mAP"]
    gap_att = oracle_att["U-mAP"] - baseline["U-mAP"]

    print(f"\nU-mAP improvement over baseline:")
    print(f"  Oracle uncertainty only: +{gap_unc:.4f}")
    print(f"  Oracle attribute only:   +{gap_att:.4f}")
    print(f"  Oracle full:             +{gap_full:.4f}")

    if gap_full < 0.01:
        print("\n>> CONCLUSION: HAUF is near ceiling. VLM distillation unlikely to help.")
    elif gap_full >= 0.01 and gap_unc > gap_att:
        print("\n>> CONCLUSION: Room for improvement. Bottleneck is UNCERTAINTY.")
        print("   → VLM uncertainty signal most valuable to distill.")
    elif gap_full >= 0.01 and gap_att > gap_unc:
        print("\n>> CONCLUSION: Room for improvement. Bottleneck is ATTRIBUTE QUALITY.")
        print("   → VLM attribute descriptions most valuable to distill.")
    else:
        print("\n>> CONCLUSION: Room for improvement. Both signals contribute.")
        print("   → Full VLM distillation (attributes + uncertainty) recommended.")

    # Save summary
    summary = {
        "metrics": {k: {kk: float(vv) if isinstance(vv, (float, np.floating)) else int(vv)
                        for kk, vv in v.items()} for k, v in all_metrics.items()},
        "gaps": {
            "oracle_uncertainty": float(gap_unc),
            "oracle_att_only": float(gap_att),
            "oracle_full": float(gap_full),
        },
    }
    with open(os.path.join(args.output_dir, "oracle_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to {args.output_dir}/oracle_summary.json")


if __name__ == "__main__":
    main()
