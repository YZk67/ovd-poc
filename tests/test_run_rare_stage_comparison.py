import subprocess
import json
from pathlib import Path

import pytest

from tools.run_rare_stage_comparison import parse_args, run


@pytest.fixture
def args(tmp_path):
    for name in ("old", "new", "annotations"):
        (tmp_path / (name + ".json")).write_text("[]")
    return parse_args([
        "--old-predictions", str(tmp_path / "old.json"),
        "--new-predictions", str(tmp_path / "new.json"),
        "--annotations", str(tmp_path / "annotations.json"),
        "--expected-old-apr", "42.8843", "--expected-new-apr", "42.3031",
        "--output-dir", str(tmp_path / "output"),
    ])


def test_missing_earlier_predictions_prevents_all_work(args, tmp_path, monkeypatch):
    (tmp_path / "old.json").unlink()
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: pytest.fail("Must not start evaluation"))
    with pytest.raises(FileNotFoundError, match="NO evaluation or GPU inference started"):
        run(args)
    assert not (tmp_path / "output").exists()


def test_saved_inputs_cannot_be_overwritten_or_identical(args, tmp_path):
    args.new_predictions = args.old_predictions
    with pytest.raises(ValueError, match="different saved files"):
        run(args)
    args.new_predictions = str(tmp_path / "new.json")
    (tmp_path / "output").mkdir()
    (tmp_path / "output/new_report.json").write_text("preserve")
    with pytest.raises(ValueError, match="Refusing to overwrite"):
        run(args)
    assert (tmp_path / "output/new_report.json").read_text() == "preserve"


def test_pipeline_uses_same_python_cpu_all_curves_and_expected_apr(args, monkeypatch):
    import sys

    calls = []
    monkeypatch.setattr(subprocess, "run", lambda command, **kw: calls.append((command, kw)))
    run(args)
    assert len(calls) == 3
    for command, kw in calls:
        assert command[:2] == [sys.executable, "-u"]
        assert kw["check"] and kw["env"]["CUDA_VISIBLE_DEVICES"] == ""
    for (command, _), apr in zip(calls[:2], ("42.8843", "42.3031")):
        assert "--all-curves" in command
        assert command[command.index("--expected-apr") + 1] == apr
    command = calls[-1][0]
    assert command[command.index("--top-declines") + 1] == "20"
    assert "--fill-missing-curves" not in command  # No redundant evaluator pass.


def test_failed_apr_check_or_evaluation_aborts_later_stages(args, monkeypatch):
    calls = []

    def fail(command, **kwargs):
        calls.append(command)
        raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(subprocess, "run", fail)
    with pytest.raises(subprocess.CalledProcessError):
        run(args)
    assert len(calls) == 1


@pytest.mark.parametrize("field,value", [("expected_old_apr", float("nan")),
                                        ("expected_new_apr", 101), ("top_declines", 0)])
def test_invalid_options_fail_before_creating_output(args, field, value, tmp_path):
    setattr(args, field, value)
    with pytest.raises(ValueError):
        run(args)
    assert not (tmp_path / "output").exists()


def test_full_iou_pipeline_explicit_labels_strict_expected_and_no_gpu(args, monkeypatch):
    args.full_iou = True
    args.old_label, args.new_label = "8ep", "12ep"
    args.expected_new_apr = 42.4229
    calls = []
    monkeypatch.setattr(subprocess, "run", lambda cmd, **kw: calls.append((cmd, kw)))
    output = run(args)
    assert len(calls) == 4 and output["full_pr"].name == "report.json"
    for cmd, kw in calls:
        assert kw["env"]["CUDA_VISIBLE_DEVICES"] == "" and kw["check"]
        assert "train_net.py" not in " ".join(cmd)
    for cmd, _ in calls[:2]:
        assert "--all-iou-curves" in cmd
        assert cmd[cmd.index("--apr-tolerance")+1] == "0.0002"
    final = calls[-1][0]
    assert final[final.index("--new-label")+1] == "12ep"
    assert final[final.index("--expected-new-apr")+1] == "42.4229"
    manifest = json.loads((Path(args.output_dir)/"inputs.json").read_text())
    assert len(manifest["inputs"]) == 3 and all(v["sha256"] for v in manifest["inputs"].values())
    assert manifest["stage_labels"] == {"old": "8ep", "new": "12ep"}
    assert (Path(args.output_dir)/"COMPLETE.json").is_file()


def reusable_reports(args, tmp_path):
    from test_decoder_aux_postmortem import reports
    for side, report in zip(("old", "new"), reports()):
        prediction = Path(getattr(args, side+"_predictions"))
        report.update(predictions=str(prediction), prediction_bytes=prediction.stat().st_size,
                      annotations=args.annotations)
        path = tmp_path/(side+"_reusable.json")
        path.write_text(json.dumps(report))
        setattr(args, side+"_report", str(path))
        setattr(args, "expected_"+side+"_apr", report["official_apr"])
    args.full_iou = True


def test_complete_reports_reused_without_evaluator(args, tmp_path, monkeypatch):
    reusable_reports(args, tmp_path)
    snapshots = {p: p.read_bytes() for p in tmp_path.glob("*.json")}
    calls = []
    monkeypatch.setattr(subprocess, "run", lambda cmd, **kw: calls.append(cmd))
    run(args)
    assert len(calls) == 2
    assert all("report_lvis_rare_pr.py" not in " ".join(cmd) for cmd in calls)
    for p, value in snapshots.items():
        assert p.read_bytes() == value


@pytest.mark.parametrize("mutation", ["missing_iou", "source_path", "source_size", "annotations", "wrong_apr"])
def test_invalid_reuse_fails_before_any_cpu_pass(args, tmp_path, monkeypatch, mutation):
    reusable_reports(args, tmp_path)
    path = Path(args.new_report)
    report = json.loads(path.read_text())
    if mutation == "missing_iou":
        del report["focus"]["keg"]["iou_curves"]["0.65"]
    elif mutation == "source_path":
        report["predictions"] = args.old_predictions
    elif mutation == "source_size":
        report["prediction_bytes"] += 1
    elif mutation == "annotations":
        report["annotations"] = args.old_predictions
    else:
        report["official_apr"] += .001
    path.write_text(json.dumps(report))
    # Missing old report is intentional: must validate the supplied new report
    # before starting the old evaluator, not spend one pass then fail.
    args.old_report = None
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: pytest.fail("Must not evaluate"))
    with pytest.raises(ValueError):
        run(args)
    assert not Path(args.output_dir).exists()


def test_changed_source_or_failed_child_never_marks_complete(args, monkeypatch):
    def mutate(cmd, **kw):
        Path(args.new_predictions).write_text("changed during pipeline")
    monkeypatch.setattr(subprocess, "run", mutate)
    with pytest.raises(ValueError, match="Source changed"):
        run(args)
    assert not (Path(args.output_dir)/"COMPLETE.json").exists()


def test_preexisting_completion_marker_not_overwritten(args):
    output = Path(args.output_dir)
    output.mkdir()
    marker = output/"COMPLETE.json"
    marker.write_text("preserve")
    with pytest.raises(ValueError, match="Refusing to overwrite"):
        run(args)
    assert marker.read_text() == "preserve"
