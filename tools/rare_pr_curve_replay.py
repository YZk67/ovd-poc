"""Fill missing rare-class curves by replaying saved, all-class LVIS results.

This is CPU-only official bbox evaluation on every image in the supplied LVIS
annotation file. Only selected categories are accumulated; the original report's
full rare-category APr is retained, never replaced by a selected-category mean.
"""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from tools.pairing_lvis_support import _lvis_from_dataset
from tools.report_lvis_rare_pr import focus_report, per_class_ap


AP_TOLERANCE = 1e-5  # Absolute AP points (0..100), including serialization noise.
REQUIRED_IOUS = ("0.50", "0.75")


def _missing_curves(report, names):
    focus = report.get("focus", {})
    if not isinstance(focus, dict):
        raise ValueError("report focus must be an object")
    missing = {}
    for name in names:
        entry = focus.get(name)
        if entry is None:
            missing[name] = list(REQUIRED_IOUS)
            continue
        if not isinstance(entry, dict):
            raise ValueError(f"focus entry for {name!r} must be an object")
        curves = entry.get("iou_curves", {})
        if not isinstance(curves, dict):
            raise ValueError(f"iou_curves for {name!r} must be an object")
        absent = [key for key in REQUIRED_IOUS if curves.get(key) is None]
        if absent:
            missing[name] = absent
    return missing


def _signature(path):
    stat = path.stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns


def _source(path):
    path = Path(path).expanduser().resolve()
    before = _signature(path)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    if _signature(path) != before:
        raise ValueError(f"source changed while hashing: {path}")
    return path, before, {
        "path": str(path), "bytes": before[2], "sha256": digest.hexdigest(),
    }


def _check_metric(name, metric, expected, actual, where="per_class"):
    if expected is None or actual is None:
        equal = expected is None and actual is None
    else:
        equal = (
            isinstance(expected, (float, int)) and not isinstance(expected, bool)
            and math.isfinite(expected) and math.isfinite(actual)
            and abs(expected - actual) <= AP_TOLERANCE
        )
    if not equal:
        raise ValueError(
            f"replayed {name} {metric}={actual!r} disagrees with "
            f"report {where} value {expected!r}; check prediction and annotation sources"
        )


def _validate_categories(report, dataset, requested):
    rows = report.get("per_class")
    if not isinstance(rows, list) or not rows:
        raise ValueError("report per_class must contain rare-category rows")
    categories = {category["id"]: category for category in dataset["categories"]}
    if len(categories) != len(dataset["categories"]):
        raise ValueError("annotations contain duplicate category IDs")
    counts = Counter(item["category_id"] for item in dataset["annotations"])
    by_name, seen_ids = {}, set()
    for row in rows:
        name, category_id = row["name"], row["category_id"]
        if name in by_name or category_id in seen_ids:
            raise ValueError("report per_class contains duplicate names or category IDs")
        category = categories.get(category_id)
        if category is None or category.get("name") != name:
            raise ValueError(f"report category ID/name for {name!r} disagrees with annotations")
        if category.get("frequency") != "r" or row.get("frequency", "r") != "r":
            raise ValueError(f"report category {name!r} is not rare in annotations")
        if row.get("gt_annotations") != counts[category_id]:
            raise ValueError(f"report raw GT count for {name!r} disagrees with annotations")
        for key in ("AP", "AP50", "AP75"):
            if key not in row:
                raise ValueError(f"report per_class {name!r} lacks {key}")
        by_name[name] = row
        seen_ids.add(category_id)
    rare_ids = {key for key, category in categories.items() if category.get("frequency") == "r"}
    if seen_ids != rare_ids:
        raise ValueError("report per_class rare-category IDs disagree with annotations")
    if report.get("rare_category_count", len(rows)) != len(rare_ids):
        raise ValueError("report rare_category_count disagrees with annotations")
    unknown = [name for name in requested if name not in by_name]
    if unknown:
        raise ValueError(f"focus classes are not rare report categories: {unknown}")
    return by_name


def _accumulate(evaluator):
    # LVIS 0.5.3 uses np.float, whose historical value is precisely built-in
    # float. Restore NumPy immediately, even if official accumulation raises.
    needs_alias = not hasattr(np, "float")
    if needs_alias:
        np.float = float
    try:
        evaluator.accumulate()
    finally:
        if needs_alias:
            del np.float
    return needs_alias


def fill_report_curves(report, focus_names, *, predictions, annotations):
    """Return ``(copied_report, replay_metadata)`` without writing source files.

    ``predictions`` is the authoritative path to an unfiltered, all-category
    prediction JSON; ``annotations`` is the original full-validation LVIS JSON.
    A caller may supply relocated paths. All report category identities and raw
    GT counts must match; replayed AP/AP50/AP75 (and AR when present) must match
    the original selected-category rows within 1e-5 absolute AP points.

    Existing IoU entries, including genuine empty detection curves, are retained.
    No files are opened and LVIS is not imported when no required entry is absent.
    Actual input hashes identify this replay, but cannot establish the original
    full-report provenance if that report did not store hashes. In particular,
    matching selected-category AP does not verify the original aggregate APr.
    """
    names = list(dict.fromkeys(focus_names))
    if any(not isinstance(name, str) or not name for name in names):
        raise ValueError("focus_names must contain nonempty category names")
    missing = _missing_curves(report, names)
    metadata = {
        "status": "skipped" if not missing else "filled",
        "requested_names": names,
        "missing_iou_curves": missing,
        "filled_names": [],
    }
    if not missing:
        metadata["reason"] = "all_requested_iou_curves_already_present"
        return deepcopy(report), metadata

    max_dets = report.get("max_dets")
    if not isinstance(max_dets, int) or isinstance(max_dets, bool) or max_dets <= 0:
        raise ValueError("report max_dets must be a positive integer")
    print(f"[replay] hashing predictions={predictions} annotations={annotations}", flush=True)
    prediction_path, prediction_signature, prediction_source = _source(predictions)
    annotation_path, annotation_signature, annotation_source = _source(annotations)
    original_path = report.get("predictions")
    same_prediction_path = (
        bool(original_path)
        and Path(original_path).expanduser().resolve() == prediction_path
    )
    expected_bytes = report.get("prediction_bytes")
    size_matches = None if expected_bytes is None else expected_bytes == prediction_source["bytes"]
    if same_prediction_path and size_matches is False:
        raise ValueError("prediction_bytes differs from report for the same prediction path")

    from lvis import LVIS, LVISEval, LVISResults

    print(f"[replay] loading full-validation annotations={annotation_path}", flush=True)
    ground_truth = LVIS(str(annotation_path))
    rows = _validate_categories(report, ground_truth.dataset, names)
    print(f"[replay] loading all-class predictions={prediction_path}", flush=True)
    with prediction_path.open(encoding="utf-8") as stream:
        saved_predictions = json.load(stream)
    if not isinstance(saved_predictions, list):
        raise ValueError("prediction JSON must be an all-class list of LVIS bbox results")
    known_images = set(ground_truth.get_img_ids())
    known_categories = set(ground_truth.get_cat_ids())
    for prediction in saved_predictions:
        if not isinstance(prediction, dict) or "bbox" not in prediction:
            raise ValueError("predictions must contain LVIS bbox result objects")
        if prediction.get("image_id") not in known_images:
            raise ValueError("predictions reference an image absent from annotations")
        if prediction.get("category_id") not in known_categories:
            raise ValueError("predictions reference a category absent from annotations")
    input_prediction_count = len(saved_predictions)
    # The cap must see every category. Do not prefilter by category or GT images.
    if saved_predictions:
        detections = LVISResults(ground_truth, saved_predictions, max_dets=max_dets)
    else:
        # Upstream LVISResults indexes results[0] even when the list is empty.
        detections = _lvis_from_dataset(LVISResults, {
            **deepcopy(ground_truth.dataset), "annotations": [],
        })
    selected = sorted((rows[name] for name in missing), key=lambda row: row["category_id"])
    evaluator = LVISEval(ground_truth, detections, "bbox")
    evaluator.params.cat_ids = [row["category_id"] for row in selected]
    evaluator.params.img_ids = sorted(known_images)
    evaluator.params.max_dets = max_dets
    evaluator.params.iou_thrs = np.linspace(0.50, 0.95, 10)
    print(
        f"[replay] matching images={len(known_images)} rare_categories={len(selected)} "
        f"max_dets={max_dets}", flush=True,
    )
    evaluator.evaluate()
    print("[replay] accumulating official IoU 0.50:0.95 precision", flush=True)
    used_numpy_alias = _accumulate(evaluator)
    patched = deepcopy(report)
    patched_focus = patched.setdefault("focus", {})
    metric_checks = {}
    for index, row in enumerate(selected):
        name = row["name"]
        replay_ap = per_class_ap(
            evaluator.eval["precision"], evaluator.eval["recall"], index, evaluator.params,
        )
        for metric in ("AP", "AP50", "AP75", "AR"):
            if metric in row:
                _check_metric(name, metric, row[metric], replay_ap[metric])
        existing = patched_focus.get(name)
        if existing is not None:
            for key in ("category_id", "name", "gt_annotations"):
                if key in existing and existing[key] != row[key]:
                    raise ValueError(f"report focus {name!r} {key} disagrees with per_class")
            for metric in ("AP", "AP50", "AP75", "AR"):
                if metric in existing:
                    _check_metric(name, metric, existing[metric], replay_ap[metric], "focus")
        generated = focus_report(evaluator, row["category_id"], index, row)
        if existing is None:
            patched_focus[name] = generated
        else:
            for key, value in generated.items():
                if key != "iou_curves":
                    existing.setdefault(key, value)
            curves = existing.setdefault("iou_curves", {})
            for key in missing[name]:
                curves[key] = generated["iou_curves"][key]
        metric_checks[name] = replay_ap
    for path, signature in (
        (prediction_path, prediction_signature), (annotation_path, annotation_signature),
    ):
        if _signature(path) != signature:
            raise ValueError(f"source changed during replay: {path}")
    metadata.update({
        "filled_names": [row["name"] for row in selected],
        "scope": "official_LVIS_bbox_selected_rare_categories_full_validation_images",
        "sources": {"predictions": prediction_source, "annotations": annotation_source},
        "max_dets": max_dets,
        "evaluation": {
            "image_count": len(known_images),
            "category_ids": evaluator.params.cat_ids,
            "iou_thresholds": evaluator.params.iou_thrs.tolist(),
            "input_predictions_all_categories": input_prediction_count,
            "capped_predictions_all_categories": len(detections.dataset["annotations"]),
            "detection_limit_policy": "per_image_across_all_categories_before_focus_filtering",
            "numpy_legacy_float_alias_during_accumulation": used_numpy_alias,
        },
        "consistency_checks": {
            "category_identity_frequency_raw_gt_count_rows": len(rows),
            "selected_category_replayed_metrics": metric_checks,
            "absolute_tolerance_ap_points": AP_TOLERANCE,
            "same_prediction_path_as_report": same_prediction_path,
            "prediction_bytes_matches_report": size_matches,
            "full_report_official_apr_preserved": True,
            "aggregate_apr_recomputed": False,
            "limitation": (
                "Actual source hashes and selected-category AP agreement do not verify "
                "the original full-report source hashes or aggregate APr."
            ),
        },
    })
    return patched, metadata
