from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import pytest

from tools import decoder_aux_postmortem_ops as ops
from tools import review_decoder_aux_ablation as cli
from tools.rare_pr_comparison_ops import summarize_ranked_curve


def curve(matches, gt=2):
    tp = fp = 0
    points = []
    for index, matched in enumerate(matches):
        tp += int(matched)
        fp += int(not matched)
        points.append({"tp": tp, "fp": fp, "score": 1-index*.01,
                       "precision": tp/(tp+fp), "recall": tp/gt})
    return {"num_gt": gt, "valid_detections": len(points), "ignored_detections": 3,
            "true_positives": tp, "false_positives": fp, "curve": points}


def reports():
    # Original focus helps, outside classes lose, and explicit unchanged/zero-GT controls.
    scenarios = [("bass_horn", [False, True, False, True], [True, True, False, False], 2),
                 ("keg", [True], [], 1), ("lasagna", [True, True], [True, True], 6),
                 ("other", [True, True], [False, False, True, True], 2),
                 ("no_tp", [], [], 12)]
    result = []
    for side in (1, 2):
        rows, focus = [], {}
        for idx, entry in enumerate(scenarios, 1):
            name, gt = entry[0], entry[3]
            raw = curve(entry[side], gt)
            ap = summarize_ranked_curve(raw)["interpolated_AP_points"]
            row = {"category_id": idx, "name": name, "gt_annotations": gt,
                   "AP": ap, "AP50": ap, "AP75": ap}
            rows.append(row)
            focus[name] = {**row, "iou_curves": {iou: deepcopy(raw) for iou in ops.IOUS}}
        rows.append({"category_id": 100, "name": "unobserved", "gt_annotations": 0,
                     "AP": None, "AP50": None, "AP75": None})
        result.append({"max_dets": 300, "rare_category_count": len(rows),
                       "official_apr": sum(r["AP"] for r in rows[:-1])/len(rows[:-1]),
                       "curve_scope": "all_rare_categories", "per_class": rows, "focus": focus})
    return result


def test_full_iou_macro_closure_focus_vs_rest_and_source_immutability():
    a, b = reports()
    before = deepcopy((a, b))
    result = ops.analyze(a, b)
    assert result["complete"] and result["valid_classes"] == 5
    assert result["all_rare_categories"] == 6  # absent rare class excluded from denominator
    assert sum(result["global"]["partition_contribution"].values()) == pytest.approx(result["delta_apr"])
    assert result["global"]["gains"] == 1
    assert result["global"]["declines"] == 2
    assert result["global"]["unchanged"] == 2
    original = result["original_focus"]
    assert [r["outcome"] for r in original["classes"]] == ["gain", "decline", "unchanged"]
    assert original["cohort"]["apr_contribution"] + original["rest"]["apr_contribution"] == pytest.approx(result["delta_apr"])
    assert (a, b) == before
    json.dumps(result, allow_nan=False)


def test_precision_loss_distinct_from_recall_loss_and_no_tp_not_missing():
    result = ops.analyze(*reports())
    rows = {r["name"]: r for r in result["per_class"]}
    ranking = rows["other"]["iou"]["0.50"]
    assert ranking["A"]["tp"] == ranking["B"]["tp"] == 2
    assert ranking["change"]["ap_partition_points"]["lost_recall_support"] == 0
    assert ranking["change"]["same_recall_mean_delta_fp_before"] == 2
    assert ranking["change"]["ap_partition_points"]["shared_recall_precision"] < 0
    assert rows["keg"]["ap_partition_points"]["lost_recall_support"] < 0
    assert rows["no_tp"]["iou"]["0.50"]["change"]["same_recall_mean_delta_fp_before"] is None


def test_all_ten_ious_not_just_50_and_75():
    a, b = reports()
    # Change only .95, leaving AP50/AP75 untouched; update official full AP metadata.
    b["focus"]["lasagna"]["iou_curves"]["0.95"] = curve([], 6)
    original_ap = b["focus"]["lasagna"]["AP"]
    for r in b["per_class"]:
        if r["name"] == "lasagna":
            r["AP"] = original_ap*.9
    b["focus"]["lasagna"]["AP"] = original_ap*.9
    b["official_apr"] -= original_ap*.1/5
    result = ops.analyze(a, b)
    lasagna = next(r for r in result["per_class"] if r["name"] == "lasagna")
    assert lasagna["delta_AP50"] == lasagna["delta_AP75"] == 0
    assert lasagna["delta_AP"] < 0
    assert sum(lasagna["ap_partition_points"].values()) == pytest.approx(lasagna["delta_AP"])


def test_missing_intermediate_iou_or_tampered_curve_rejected():
    a, b = reports()
    del b["focus"]["keg"]["iou_curves"]["0.65"]
    with pytest.raises(ValueError, match="Missing B keg IoU=0.65"):
        ops.analyze(a, b)
    a, b = reports()
    b["focus"]["bass_horn"]["iou_curves"]["0.95"] = curve([], 2)
    with pytest.raises(ValueError, match="all-IoU AP"):
        ops.analyze(a, b)


def test_representatives_cover_gain_decline_unchanged_and_gt_strata():
    result = ops.analyze(*reports())
    examples = result["representatives"]
    assert {e["outcome"] for e in examples} == {"gain", "decline", "unchanged"}
    assert {e["gt_range"] for e in examples} == {"1", "2-4", "5-9", "10-19"}
    assert sum(s["apr_contribution"] for s in result["gt_strata"]) == pytest.approx(result["delta_apr"])
    assert sum(s["classes"] for s in result["gt_strata"]) == 5


@pytest.fixture
def trial(tmp_path):
    path = tmp_path/"trial"
    path.mkdir()
    annotations = tmp_path/"annotations.json"
    cli.save_json(annotations, {"mock": "full validation"})
    classification = tmp_path/"classification.json"
    cli.save_json(classification, {"inputs": {"sources": {"annotations": cli.file_identity(annotations)}}})
    m = {"updates": 500, "classification_audit": cli.file_identity(classification)}
    m["fingerprint"] = cli.digest(m)
    cli.save_json(path/"manifest.json", m)
    a, b = reports()
    evals = {}
    for arm, report in (("A", a), ("B", b)):
        pred = path/f"predictions_{arm}.json"
        cli.save_json(pred, [{"mock": arm}])
        evals[arm] = {"metrics": {"APr": report["official_apr"]}, "predictions": cli.file_identity(pred)}
    cli.save_json(path/"summary.json", {
        "complete": True, "updates_per_arm": 500, "manifest_fingerprint": m["fingerprint"],
        "evaluations": evals, "delta_B_minus_A": {"APr": b["official_apr"]-a["official_apr"]}})
    return cli.parse_args(["--summary", str(path/"summary.json"), "--output-dir", str(tmp_path/"review")])


def test_missing_or_changed_inputs_abort_before_cpu_evaluation(trial, monkeypatch):
    monkeypatch.setattr(cli, "run_cpu", lambda *a: pytest.fail("evaluation launched"))
    path = Path(trial.summary).parent/"predictions_B.json"
    path.write_text("changed")
    with pytest.raises(ValueError, match="B all-class predictions changed"):
        cli.run(trial)
    assert not Path(trial.output_dir).exists()


def test_incomplete_trial_and_bad_delta_are_rejected(trial):
    path = Path(trial.summary)
    report = cli.load_json(path)
    report["complete"] = False
    cli.save_json(path, report)
    with pytest.raises(ValueError, match="completed"):
        cli.inputs_from_trial(trial)
    report["complete"] = True
    report["delta_B_minus_A"]["APr"] = 10
    cli.save_json(path, report)
    with pytest.raises(ValueError, match="arithmetic"):
        cli.inputs_from_trial(trial)


def test_pipeline_only_cpu_reports_then_cached_resume(trial, monkeypatch):
    calls = []
    original = {p: p.read_bytes() for p in Path(trial.summary).parent.iterdir()}
    def run_cpu(command, log):
        calls.append(command)
        assert command[2].endswith("report_lvis_rare_pr.py")
        assert "--all-iou-curves" in command and "--all-curves" in command
        arm = log.stem
        report = reports()[arm == "B"]
        pred = Path(command[command.index("--predictions")+1])
        report.update(predictions=str(pred), prediction_bytes=pred.stat().st_size,
                      annotations=command[command.index("--annotations")+1])
        cli.save_json(command[-1], report)
        log.write_text("CPU mock")
    monkeypatch.setattr(cli, "run_cpu", run_cpu)
    result = cli.run(trial)
    assert len(calls) == 2 and result["training_updates"] == 0 and not result["gpu_inference"]
    trial.resume = True
    assert cli.run(trial) == result
    assert len(calls) == 2
    for path, value in original.items():
        assert path.read_bytes() == value


def test_prepare_only_and_output_protection(trial, monkeypatch):
    monkeypatch.setattr(cli, "run_cpu", lambda *a: pytest.fail("evaluation launched"))
    trial.prepare_only = True
    result = cli.run(trial)
    assert result["training_updates"] == 0
    assert not (Path(trial.output_dir)/"A_report.json").exists()
    with pytest.raises(ValueError, match="NEW directory"):
        cli.run(trial)
    trial.output_dir = str(Path(trial.summary).parent/"inside")
    with pytest.raises(ValueError, match="separate"):
        cli.run(trial)


def test_help_and_ops_do_not_need_torch_or_any_site_packages():
    script = Path(cli.__file__)
    result = subprocess.run([sys.executable, "-S", str(script), "--help"], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "CPU-only" in result.stdout


def test_cpu_subprocess_hides_gpus_and_failure_propagates(tmp_path, monkeypatch):
    seen = {}
    class Process:
        stdout = iter(["official evaluator failed\n"])
        def __enter__(self):
            return self
        def __exit__(self, *a):
            pass
        def wait(self):
            return 2
    def popen(command, **kwargs):
        seen.update(kwargs)
        return Process()
    monkeypatch.setattr(cli.subprocess, "Popen", popen)
    with pytest.raises(subprocess.CalledProcessError):
        cli.run_cpu(["cpu_report"], tmp_path/"cpu.log")
    assert seen["env"]["CUDA_VISIBLE_DEVICES"] == ""
    assert (tmp_path/"cpu.log").read_text() == "official evaluator failed\n"


def test_cpu_report_requires_expected_apr(trial, tmp_path):
    inputs = cli.inputs_from_trial(trial)
    a = reports()[0]
    a.update(predictions=inputs["predictions"]["A"]["path"], prediction_bytes=inputs["predictions"]["A"]["bytes"],
             annotations=inputs["annotations"]["path"], official_apr=20.)
    path = tmp_path/"bad.json"
    cli.save_json(path, a)
    with pytest.raises(ValueError, match="certified"):
        cli.verify_report(path, inputs, "A")
