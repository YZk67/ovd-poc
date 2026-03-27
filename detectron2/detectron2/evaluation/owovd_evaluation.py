"""OW-OVD Evaluation: U-Recall, U-mAP, WI, A-OSE metrics.

Evaluates open-world open-vocabulary detection by measuring both
known-class performance and unknown object detection capability.
"""

import copy
import itertools
import json
import logging
import os
from collections import OrderedDict, defaultdict

import numpy as np
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from detectron2.evaluation import DatasetEvaluator
from detectron2.utils import comm

logger = logging.getLogger(__name__)


class OWOVDEvaluator(DatasetEvaluator):
    """Evaluator for Open-World Open-Vocabulary Detection.

    Computes:
        - Known mAP: standard COCO mAP on known classes
        - U-Recall: recall of unknown objects at IoU=0.5
        - U-mAP: AP for unknown objects (treated as single class)
        - WI: Wilderness Impact (how unknowns affect known precision)
        - A-OSE: unknown objects misclassified as known

    Args:
        dataset_name: Registered dataset name.
        known_cat_ids: Set of category IDs that are "known" for this task.
        unknown_class_id: The class ID used for unknown predictions.
        output_dir: Directory to save results.
        iou_threshold: IoU threshold for matching (default 0.5).
    """

    def __init__(
        self,
        dataset_name,
        known_cat_ids,
        unknown_class_id=80,
        output_dir=None,
        iou_threshold=0.5,
    ):
        self.dataset_name = dataset_name
        self.known_cat_ids = set(known_cat_ids)
        self.unknown_class_id = unknown_class_id
        self.output_dir = output_dir
        self.iou_threshold = iou_threshold
        self._predictions = []

    def reset(self):
        self._predictions = []

    def process(self, inputs, outputs):
        """Process a batch of predictions."""
        for input_dict, output in zip(inputs, outputs):
            image_id = input_dict["image_id"]
            instances = output["instances"].to("cpu")

            prediction = {"image_id": image_id}
            if len(instances) > 0:
                prediction["boxes"] = instances.pred_boxes.tensor.numpy()
                prediction["scores"] = instances.scores.numpy()
                prediction["classes"] = instances.pred_classes.numpy()
            else:
                prediction["boxes"] = np.zeros((0, 4))
                prediction["scores"] = np.zeros(0)
                prediction["classes"] = np.zeros(0, dtype=int)

            self._predictions.append(prediction)

    def evaluate(self):
        """Compute all OW-OVD metrics."""
        if comm.get_world_size() > 1:
            comm.synchronize()
            all_predictions = comm.gather(self._predictions, dst=0)
            if not comm.is_main_process():
                return {}
            self._predictions = list(itertools.chain(*all_predictions))

        if len(self._predictions) == 0:
            logger.warning("No predictions to evaluate.")
            return {}

        results = OrderedDict()

        # Load GT annotations
        from detectron2.data import DatasetCatalog, MetadataCatalog
        metadata = MetadataCatalog.get(self.dataset_name)
        json_file = metadata.json_file
        coco_gt = COCO(json_file)

        # Convert GT category IDs to contiguous IDs to match model predictions
        if hasattr(metadata, "thing_dataset_id_to_contiguous_id"):
            self._id_map = metadata.thing_dataset_id_to_contiguous_id
        else:
            self._id_map = None

        all_cat_ids_orig = set(coco_gt.getCatIds())
        if self._id_map:
            all_cat_ids = set(self._id_map.get(cid, cid) for cid in all_cat_ids_orig)
        else:
            all_cat_ids = all_cat_ids_orig
        unknown_cat_ids = all_cat_ids - self.known_cat_ids

        logger.info(f"Known classes: {len(self.known_cat_ids)}, "
                     f"Unknown classes: {len(unknown_cat_ids)}")

        # Compute metrics
        u_recall = self._compute_unknown_recall(coco_gt, unknown_cat_ids)
        u_map = self._compute_unknown_map(coco_gt, unknown_cat_ids)
        a_ose = self._compute_a_ose(coco_gt, unknown_cat_ids)
        known_map = self._compute_known_map(coco_gt)

        results["U-Recall"] = u_recall
        results["U-mAP"] = u_map
        results["A-OSE"] = a_ose
        results["Known-mAP"] = known_map

        logger.info(f"OW-OVD Results: U-Recall={u_recall:.4f}, U-mAP={u_map:.4f}, "
                     f"A-OSE={a_ose}, Known-mAP={known_map:.4f}")

        if self.output_dir:
            os.makedirs(self.output_dir, exist_ok=True)
            with open(os.path.join(self.output_dir, "owovd_results.json"), "w") as f:
                json.dump(results, f, indent=2)

        return {"owovd": results}

    def _contiguous_to_coco_ids(self, contiguous_ids):
        """Convert contiguous IDs back to COCO original IDs for GT queries."""
        if self._id_map is None:
            return list(contiguous_ids)
        reverse_map = {v: k for k, v in self._id_map.items()}
        return [reverse_map[cid] for cid in contiguous_ids if cid in reverse_map]

    def _compute_unknown_recall(self, coco_gt, unknown_cat_ids):
        """Compute recall for unknown objects at IoU threshold."""
        if not unknown_cat_ids:
            return 0.0

        # Get all unknown GT boxes (need COCO original IDs for query)
        unknown_coco_ids = self._contiguous_to_coco_ids(unknown_cat_ids)
        if not unknown_coco_ids:
            return 0.0
        unknown_ann_ids = coco_gt.getAnnIds(catIds=unknown_coco_ids)
        unknown_anns = coco_gt.loadAnns(unknown_ann_ids)

        # Group by image
        gt_by_image = defaultdict(list)
        for ann in unknown_anns:
            if ann.get("iscrowd", 0):
                continue
            x, y, w, h = ann["bbox"]
            gt_by_image[ann["image_id"]].append([x, y, x + w, y + h])

        total_unknown = sum(len(v) for v in gt_by_image.values())
        if total_unknown == 0:
            return 0.0

        recalled = 0
        for pred in self._predictions:
            image_id = pred["image_id"]
            gt_boxes = gt_by_image.get(image_id, [])
            if not gt_boxes:
                continue

            # Unknown predictions: class == unknown_class_id
            unknown_mask = pred["classes"] == self.unknown_class_id
            if not unknown_mask.any():
                continue

            pred_boxes = pred["boxes"][unknown_mask]
            gt_boxes_arr = np.array(gt_boxes)

            # Compute IoU
            matched_gt = set()
            # Sort by score descending
            scores = pred["scores"][unknown_mask]
            order = scores.argsort()[::-1]

            for idx in order:
                pb = pred_boxes[idx]
                ious = self._compute_iou(pb, gt_boxes_arr)
                best_gt = ious.argmax()
                if ious[best_gt] >= self.iou_threshold and best_gt not in matched_gt:
                    matched_gt.add(best_gt)
                    recalled += 1

        return recalled / total_unknown

    def _compute_unknown_map(self, coco_gt, unknown_cat_ids):
        """Compute mAP for unknown objects (treated as single class)."""
        if not unknown_cat_ids:
            return 0.0

        # Collect all unknown GT
        unknown_coco_ids = self._contiguous_to_coco_ids(unknown_cat_ids)
        if not unknown_coco_ids:
            return 0.0
        unknown_ann_ids = coco_gt.getAnnIds(catIds=unknown_coco_ids)
        unknown_anns = coco_gt.loadAnns(unknown_ann_ids)

        gt_by_image = defaultdict(list)
        for ann in unknown_anns:
            if ann.get("iscrowd", 0):
                continue
            x, y, w, h = ann["bbox"]
            gt_by_image[ann["image_id"]].append([x, y, x + w, y + h])

        total_gt = sum(len(v) for v in gt_by_image.values())
        if total_gt == 0:
            return 0.0

        # Collect all unknown predictions sorted by score
        all_scores = []
        all_tp = []

        for pred in self._predictions:
            image_id = pred["image_id"]
            gt_boxes = gt_by_image.get(image_id, [])

            unknown_mask = pred["classes"] == self.unknown_class_id
            if not unknown_mask.any():
                continue

            pred_boxes = pred["boxes"][unknown_mask]
            scores = pred["scores"][unknown_mask]

            gt_boxes_arr = np.array(gt_boxes) if gt_boxes else np.zeros((0, 4))
            matched_gt = set()

            order = scores.argsort()[::-1]
            for idx in order:
                all_scores.append(scores[idx])
                if gt_boxes_arr.shape[0] > 0:
                    ious = self._compute_iou(pred_boxes[idx], gt_boxes_arr)
                    best_gt = ious.argmax()
                    if ious[best_gt] >= self.iou_threshold and best_gt not in matched_gt:
                        matched_gt.add(best_gt)
                        all_tp.append(1)
                    else:
                        all_tp.append(0)
                else:
                    all_tp.append(0)

        if not all_scores:
            return 0.0

        # Sort by score and compute AP
        order = np.argsort(-np.array(all_scores))
        tp = np.array(all_tp)[order]
        fp = 1 - tp
        tp_cum = tp.cumsum()
        fp_cum = fp.cumsum()
        recall = tp_cum / total_gt
        precision = tp_cum / (tp_cum + fp_cum)

        # VOC-style AP (11-point interpolation)
        ap = 0.0
        for t in np.arange(0, 1.1, 0.1):
            mask = recall >= t
            if mask.any():
                ap += precision[mask].max() / 11.0

        return ap

    def _compute_a_ose(self, coco_gt, unknown_cat_ids):
        """Count unknown objects misclassified as known classes."""
        if not unknown_cat_ids:
            return 0

        unknown_coco_ids = self._contiguous_to_coco_ids(unknown_cat_ids)
        if not unknown_coco_ids:
            return 0
        unknown_ann_ids = coco_gt.getAnnIds(catIds=unknown_coco_ids)
        unknown_anns = coco_gt.loadAnns(unknown_ann_ids)

        gt_by_image = defaultdict(list)
        for ann in unknown_anns:
            if ann.get("iscrowd", 0):
                continue
            x, y, w, h = ann["bbox"]
            gt_by_image[ann["image_id"]].append([x, y, x + w, y + h])

        a_ose = 0
        for pred in self._predictions:
            image_id = pred["image_id"]
            gt_boxes = gt_by_image.get(image_id, [])
            if not gt_boxes:
                continue

            # Known predictions (not unknown class)
            known_mask = pred["classes"] != self.unknown_class_id
            if not known_mask.any():
                continue

            pred_boxes = pred["boxes"][known_mask]
            gt_boxes_arr = np.array(gt_boxes)

            for pb in pred_boxes:
                ious = self._compute_iou(pb, gt_boxes_arr)
                if ious.max() >= self.iou_threshold:
                    a_ose += 1  # Unknown GT matched by known prediction

        return a_ose

    def _compute_known_map(self, coco_gt):
        """Standard COCO mAP on known classes only."""
        # Convert predictions to COCO format, filtering to known classes only
        coco_results = []
        for pred in self._predictions:
            known_mask = pred["classes"] != self.unknown_class_id
            if not known_mask.any():
                continue
            boxes = pred["boxes"][known_mask]
            scores = pred["scores"][known_mask]
            classes = pred["classes"][known_mask]

            for box, score, cls in zip(boxes, scores, classes):
                x1, y1, x2, y2 = box
                coco_results.append({
                    "image_id": pred["image_id"],
                    "category_id": int(cls),
                    "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                    "score": float(score),
                })

        if not coco_results:
            return 0.0

        # Use COCO eval
        coco_dt = coco_gt.loadRes(coco_results)
        coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
        coco_eval.params.catIds = sorted(self.known_cat_ids)
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        return coco_eval.stats[0]  # mAP@[.5:.95]

    @staticmethod
    def _compute_iou(box, boxes):
        """Compute IoU between one box and an array of boxes.

        Args:
            box: [4] - (x1, y1, x2, y2)
            boxes: [N, 4] - (x1, y1, x2, y2)

        Returns:
            ious: [N]
        """
        x1 = np.maximum(box[0], boxes[:, 0])
        y1 = np.maximum(box[1], boxes[:, 1])
        x2 = np.minimum(box[2], boxes[:, 2])
        y2 = np.minimum(box[3], boxes[:, 3])

        inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
        area_box = (box[2] - box[0]) * (box[3] - box[1])
        area_boxes = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        union = area_box + area_boxes - inter

        return inter / np.maximum(union, 1e-8)
