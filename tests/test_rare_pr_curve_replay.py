from copy import deepcopy
import hashlib
import json

import numpy as np
import pytest

from tools.rare_pr_curve_replay import AP_TOLERANCE, fill_report_curves
from tools.report_lvis_rare_pr import focus_report, per_class_ap


def _image(image_id, *, negatives=(), not_exhaustive=()):
    return {
        "id": image_id, "width": 100, "height": 100,
        "neg_category_ids": list(negatives),
        "not_exhaustive_category_ids": list(not_exhaustive),
    }


def _annotation(ann_id, image_id, category_id=11, **extra):
    return {
        "id": ann_id, "image_id": image_id, "category_id": category_id,
        "bbox": [0, 0, 10, 10], "area": 100, **extra,
    }


def _prediction(image_id, score=0.9, category_id=11, bbox=None):
    return {
        "image_id": image_id, "category_id": category_id, "score": score,
        "bbox": [0, 0, 10, 10] if bbox is None else bbox,
    }


def _dataset():
    # Sparse raw category IDs catch accidental contiguous-index substitutions.
    return {
        "images": [_image(1), _image(2, negatives=[11])],
        "annotations": [_annotation(1, 1)],
        "categories": [
            {"id": 11, "name": "rare_one", "frequency": "r"},
            {"id": 42, "name": "rare_two", "frequency": "r"},
            {"id": 99, "name": "frequent", "frequency": "f"},
        ],
    }


def _write_inputs(tmp_path, source, predictions):
    annotation_path = tmp_path / "annotations.json"
    prediction_path = tmp_path / "predictions.json"
    annotation_path.write_text(json.dumps(source), encoding="utf-8")
    prediction_path.write_text(json.dumps(predictions), encoding="utf-8")
    return prediction_path, annotation_path


def _original_report(tmp_path, source=None, predictions=None, *, max_dets=300, focus=()):
    """Produce the existing full rare report using the actual upstream evaluator."""
    lvis = pytest.importorskip("lvis")
    source = _dataset() if source is None else source
    predictions = [_prediction(1)] if predictions is None else predictions
    prediction_path, annotation_path = _write_inputs(tmp_path, source, predictions)
    ground_truth = lvis.LVIS(str(annotation_path))
    if predictions:
        detections = lvis.LVISResults(ground_truth, deepcopy(predictions), max_dets=max_dets)
    else:
        from tools.pairing_lvis_support import _lvis_from_dataset
        detections = _lvis_from_dataset(lvis.LVISResults, {
            **deepcopy(source), "annotations": [],
        })
    evaluator = lvis.LVISEval(ground_truth, detections, "bbox")
    rare = sorted(
        (category for category in source["categories"] if category["frequency"] == "r"),
        key=lambda category: category["id"],
    )
    evaluator.params.cat_ids = [category["id"] for category in rare]
    evaluator.params.max_dets = max_dets
    evaluator.evaluate()
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(np, "float", float, raising=False)
        evaluator.accumulate()
    evaluator.summarize()
    per_class = [
        {
            "category_id": category["id"], "name": category["name"],
            "gt_annotations": sum(
                item["category_id"] == category["id"] for item in source["annotations"]
            ),
            **per_class_ap(evaluator.eval["precision"], evaluator.eval["recall"], index, evaluator.params),
        }
        for index, category in enumerate(rare)
    ]
    return {
        "predictions": str(prediction_path),
        "prediction_bytes": prediction_path.stat().st_size,
        "annotations": str(annotation_path),
        "max_dets": max_dets,
        "official_apr": 100 * evaluator.get_results()["APr"],
        "rare_category_count": len(rare),
        "per_class": per_class,
        "focus": {
            row["name"]: focus_report(evaluator, row["category_id"], index, row)
            for index, row in enumerate(per_class) if row["name"] in focus
        },
        "custom_metadata": {"retain": ["original"]},
    }, prediction_path, annotation_path


def _raw_101_point_ap(ranked):
    """Independent definition: right-hand precision envelope, 101 recall levels."""
    if ranked["num_gt"] == 0:
        return None
    values = []
    for recall in np.linspace(0, 1, 101):
        eligible = [point["precision"] for point in ranked["curve"] if point["recall"] >= recall]
        values.append(max(eligible, default=0))
    return 100 * sum(values) / 101


def test_complete_focus_is_deepcopied_without_opening_any_sources(monkeypatch):
    report = {"focus": {"rare_one": {"iou_curves": {
        "0.50": {"curve": []}, "0.75": {"curve": []},
    }}}}
    def cannot_open(*args, **kwargs):
        raise AssertionError("complete report must not read files")
    monkeypatch.setattr("tools.rare_pr_curve_replay._source", cannot_open)
    copied, metadata = fill_report_curves(
        report, ["rare_one"], predictions="absent-pred.json", annotations="absent-gt.json",
    )
    assert copied == report and copied is not report
    copied["focus"]["rare_one"]["iou_curves"]["0.50"]["curve"].append({})
    assert report["focus"]["rare_one"]["iou_curves"]["0.50"]["curve"] == []
    assert metadata["status"] == "skipped"


def test_official_replay_keeps_full_validation_negatives_and_closes_ap50_ap75(tmp_path):
    source = _dataset()
    source["images"].extend([
        _image(3), _image(4, not_exhaustive=[11]), _image(5),
    ])
    source["annotations"].extend([
        _annotation(2, 3, ignore=1), _annotation(3, 4), _annotation(4, 5, 42),
    ])
    predictions = [
        _prediction(2, 0.99),  # Verified-negative image with no rare GT: FP.
        _prediction(1, 0.9, bbox=[2, 0, 10, 10]),  # IoU 2/3: TP50, FP75.
        _prediction(4, 0.8),
        _prediction(4, 0.7, bbox=[30, 30, 10, 10]),  # Not exhaustive: ignored.
        _prediction(3, 0.6),  # Matches ignored GT: ignored.
        _prediction(2, 0.5, category_id=42),  # Unverified absence: dropped by LVIS.
    ]
    report, prediction_path, annotation_path = _original_report(tmp_path, source, predictions)
    before = deepcopy(report)
    file_bytes = [prediction_path.read_bytes(), annotation_path.read_bytes()]
    had_float = hasattr(np, "float")
    patched, metadata = fill_report_curves(
        report, ["rare_one"], predictions=prediction_path, annotations=annotation_path,
    )
    assert hasattr(np, "float") == had_float
    assert report == before
    assert [prediction_path.read_bytes(), annotation_path.read_bytes()] == file_bytes
    assert patched["official_apr"] == report["official_apr"]
    assert patched["per_class"] == report["per_class"]
    assert patched["custom_metadata"] == report["custom_metadata"]
    assert list(patched["focus"]) == ["rare_one"]
    focus = patched["focus"]["rare_one"]
    assert focus["AP"] != pytest.approx(patched["official_apr"])
    curve50, curve75 = [focus["iou_curves"][key] for key in ("0.50", "0.75")]
    assert curve50["num_gt"] == 2  # Report raw GT count is 3 due to ignore=1.
    assert focus["gt_annotations"] == 3
    assert (curve50["true_positives"], curve50["false_positives"]) == (2, 1)
    assert (curve75["true_positives"], curve75["false_positives"]) == (1, 2)
    assert curve50["ignored_detections"] == curve75["ignored_detections"] == 2
    assert curve50["curve"][0]["fp"] == 1
    for metric, ranked in (("AP50", curve50), ("AP75", curve75)):
        assert _raw_101_point_ap(ranked) == pytest.approx(focus[metric], abs=AP_TOLERANCE)
        assert 100 * np.mean([p["precision"] for p in ranked["official_interpolated_pr"]]) == pytest.approx(focus[metric])
    assert focus["AP50"] == pytest.approx(100 * 2 / 3)
    assert focus["AP75"] == pytest.approx(100 * 51 / 303)
    assert metadata["evaluation"]["image_count"] == 5
    assert metadata["evaluation"]["category_ids"] == [11]
    assert metadata["evaluation"]["iou_thresholds"] == pytest.approx(np.linspace(0.50, 0.95, 10))
    assert metadata["consistency_checks"]["aggregate_apr_recomputed"] is False
    assert metadata["sources"]["predictions"]["sha256"] == hashlib.sha256(file_bytes[0]).hexdigest()
    assert metadata["sources"]["annotations"]["sha256"] == hashlib.sha256(file_bytes[1]).hexdigest()
    json.dumps(metadata, allow_nan=False)
    json.dumps(patched, allow_nan=False)


def test_replay_caps_all_categories_before_filtering_focus(tmp_path):
    report, predictions, annotations = _original_report(
        tmp_path, predictions=[_prediction(1, 0.99, 99), _prediction(1, 0.9)], max_dets=1,
    )
    patched, metadata = fill_report_curves(
        report, ["rare_one"], predictions=predictions, annotations=annotations,
    )
    ranked = patched["focus"]["rare_one"]["iou_curves"]["0.50"]
    assert ranked["curve"] == []
    assert ranked["num_gt"] == 1 and ranked["true_positives"] == 0
    assert patched["focus"]["rare_one"]["AP"] == 0
    assert metadata["evaluation"]["input_predictions_all_categories"] == 2
    assert metadata["evaluation"]["capped_predictions_all_categories"] == 1


def test_partial_replay_preserves_existing_iou_curve_and_unrequested_focus(tmp_path):
    report, predictions, annotations = _original_report(tmp_path, focus=("rare_one", "rare_two"))
    del report["focus"]["rare_one"]["iou_curves"]["0.75"]
    report["focus"]["rare_one"]["iou_curves"]["0.50"]["custom"] = "preserved"
    before = deepcopy(report)
    patched, metadata = fill_report_curves(
        report, ["rare_one", "rare_two"], predictions=predictions, annotations=annotations,
    )
    assert report == before
    assert patched["focus"]["rare_two"] == report["focus"]["rare_two"]
    assert patched["focus"]["rare_one"]["iou_curves"]["0.50"] == report["focus"]["rare_one"]["iou_curves"]["0.50"]
    assert "0.75" in patched["focus"]["rare_one"]["iou_curves"]
    assert metadata["missing_iou_curves"] == {"rare_one": ["0.75"]}
    assert metadata["filled_names"] == ["rare_one"]


def test_genuine_empty_predictions_have_zero_ap_without_dummy_false_positive(tmp_path):
    report, predictions, annotations = _original_report(tmp_path, predictions=[])
    patched, metadata = fill_report_curves(
        report, ["rare_one", "rare_two"], predictions=predictions, annotations=annotations,
    )
    assert patched["focus"]["rare_one"]["AP"] == 0
    assert patched["focus"]["rare_two"]["AP"] is None
    assert metadata["evaluation"]["input_predictions_all_categories"] == 0
    for focus in patched["focus"].values():
        for ranked in focus["iou_curves"].values():
            assert ranked["curve"] == []
            assert ranked["true_positives"] == ranked["false_positives"] == 0


@pytest.mark.parametrize("metric", ["AP", "AP50", "AP75", "AR"])
def test_mismatched_source_selected_category_ap_is_rejected(tmp_path, metric):
    report, predictions, annotations = _original_report(tmp_path)
    report["per_class"][0][metric] -= 1
    before = deepcopy(report)
    with pytest.raises(ValueError, match=f"rare_one {metric}=.*disagrees"):
        fill_report_curves(report, ["rare_one"], predictions=predictions, annotations=annotations)
    assert report == before


@pytest.mark.parametrize("field,value,match", [
    ("category_id", 1, "ID/name"),
    ("name", "renamed", "ID/name"),
    ("gt_annotations", 99, "raw GT count"),
    ("frequency", "c", "not rare"),
])
def test_annotation_identity_frequency_and_raw_counts_must_match_report(tmp_path, field, value, match):
    report, predictions, annotations = _original_report(tmp_path)
    report["per_class"][0][field] = value
    with pytest.raises(ValueError, match=match):
        fill_report_curves(report, ["rare_one"], predictions=predictions, annotations=annotations)


def test_relocated_explicit_predictions_are_hashed_and_size_is_not_identity(tmp_path):
    report, predictions, annotations = _original_report(tmp_path)
    relocated = tmp_path / "relocated.json"
    relocated.write_text(json.dumps(json.loads(predictions.read_text()), indent=2), encoding="utf-8")
    _, metadata = fill_report_curves(report, ["rare_one"], predictions=relocated, annotations=annotations)
    assert metadata["sources"]["predictions"]["path"] == str(relocated)
    assert metadata["consistency_checks"]["same_prediction_path_as_report"] is False
    assert metadata["consistency_checks"]["prediction_bytes_matches_report"] is False


def test_changed_bytes_at_original_prediction_path_are_rejected(tmp_path):
    report, predictions, annotations = _original_report(tmp_path)
    report["prediction_bytes"] += 1
    with pytest.raises(ValueError, match="prediction_bytes"):
        fill_report_curves(report, ["rare_one"], predictions=predictions, annotations=annotations)


def test_numpy_compatibility_is_restored_after_failed_accumulation(tmp_path, monkeypatch):
    report, predictions, annotations = _original_report(tmp_path)
    import lvis
    def fail(self):
        assert np.float is float
        raise RuntimeError("accumulation failure")
    monkeypatch.delattr(np, "float", raising=False)
    monkeypatch.setattr(lvis.LVISEval, "accumulate", fail)
    with pytest.raises(RuntimeError, match="accumulation failure"):
        fill_report_curves(report, ["rare_one"], predictions=predictions, annotations=annotations)
    assert not hasattr(np, "float")
