from copy import deepcopy
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from tools import evaluate_formula_b_log5 as runner
from tools.compare_rare_pr_reports import file_identity, load_json, save_json
from tools.diagnose_rare_fp_regions import fingerprint


METRICS_A = dict(zip(runner.METRICS, (
    41.0177, 53.4954, 43.3802, 29.9986, 51.2753, 57.9643, 42.1607, 38.6691, 43.1366,
)))
METRICS_B = dict(zip(runner.METRICS, (
    38.5362, 49.9244, 40.8816, 26.9322, 47.5498, 55.8181, 42.6877, 35.1792, 40.4582,
)))


def write_result(directory, metrics):
    (directory / "console.log").write_text("Evaluation completed\n")
    (directory / "log.txt").write_text(
        "copypaste: " + ",".join(f"{metrics[key]:.4f}" for key in runner.METRICS)
    )
    (directory / "lvis_instances_results.json").write_text('[{"image_id": 1}]')
    return {
        "metrics": metrics,
        "predictions": file_identity(directory / "lvis_instances_results.json"),
    }


@pytest.fixture
def screen(tmp_path):
    directory = tmp_path / "screen"
    directory.mkdir()
    config = tmp_path / "config.py"
    config.write_text("# fixture config\n")
    manifest = {
        "output_dir": str(directory),
        "config": file_identity(config),
        "updates": 500,
        "num_gpus": 4,
        "seed": 42,
        "torch_version": str(torch.__version__),
        "code": {},
        "arms": {"A": "calibrated", "B": "calibrated_plus_logK", "C": "legacy"},
        "tpa_geometry_digest": "fixed-geometry",
    }
    manifest["fingerprint"] = fingerprint(manifest)
    save_json(directory / "manifest.json", manifest)
    report = {
        "complete": True,
        "all_tpa_geometry_unchanged": True,
        "manifest_fingerprint": manifest["fingerprint"],
        "updates_per_arm": 500,
        "training": {},
        "evaluations": {},
    }
    for arm, metrics in (("A", METRICS_A), ("B", METRICS_B)):
        (directory / arm).mkdir()
        checkpoint = directory / arm / "model_final.pth"
        checkpoint.write_bytes(b"fixture checkpoint " + arm.encode())
        stage = {
            "checkpoint": file_identity(checkpoint),
            "tpa_geometry_unchanged": True,
            "receipts": [{
                "complete": True, "rank": rank, "arm": arm, "updates": 500,
                "aggregation": manifest["arms"][arm],
                "manifest_fingerprint": manifest["fingerprint"],
                "tpa_geometry_digest": manifest["tpa_geometry_digest"],
            } for rank in range(4)],
        }
        report["training"][arm] = stage
        evaluation = directory / f"eval_{arm}"
        evaluation.mkdir()
        result = write_result(evaluation, deepcopy(metrics))
        report["evaluations"][arm] = result
        save_json(evaluation / "verified_result.json", {
            "checkpoint": stage["checkpoint"],
            "manifest_fingerprint": manifest["fingerprint"], "result": result,
        })
        save_json(evaluation / "command.json", runner.evaluation_command(config, checkpoint, evaluation, 4))
    save_json(directory / "summary.json", report)
    return runner.parse_args(["--screen-dir", str(directory)])


def test_command_changes_only_eval_bias():
    original = runner.evaluation_command("config.py", "B.pth", "new-output", 4)
    changed = runner.log5_command("config.py", "B.pth", "new-output", 4)
    differences = [(a, b) for a, b in zip(original, changed) if a != b]
    assert len(original) == len(changed)
    assert differences == [(runner.BIAS_KEY + "0.0", runner.BIAS_KEY + repr(runner.LOG5))]
    assert "--eval-only" in changed and "--resume" not in changed
    assert "model.classifier.tpa_eval_legacy_logsumexp=False" in changed


def test_dry_run_validates_without_writes_or_gpu(screen, monkeypatch):
    screen.dry_run = True
    monkeypatch.setattr(runner, "run_evaluation", lambda *args: pytest.fail("GPU started"))
    result = runner.run(screen)
    assert result["training_updates"] == 0
    assert not (Path(screen.screen_dir) / "eval_B_plus_log5").exists()


def test_one_eval_reports_bias_effect_separately_and_preserves_inputs(screen, monkeypatch):
    root = Path(screen.screen_dir)
    protected = [root / "B/model_final.pth", root / "summary.json", root / "eval_B/log.txt"]
    before = [file_identity(path) for path in protected]
    calls = []

    def evaluate(command, output):
        calls.append(command)
        assert command[0] == sys.executable
        assert f'train.init_checkpoint="{root / "B/model_final.pth"}"' in command
        write_result(output, {**METRICS_B, "AP": 41.2, "APr": 42.7})

    monkeypatch.setattr(runner, "run_evaluation", evaluate)
    result = runner.run(screen)
    assert len(calls) == 1
    assert result["delta_B_plus_log5_minus_B_calibrated"]["AP"] == pytest.approx(2.6638)
    assert result["delta_B_plus_log5_minus_A_calibrated"]["AP"] == pytest.approx(.1823)
    assert result["checkpoint_unchanged"] and result["training_updates"] == 0
    assert before == [file_identity(path) for path in protected]
    assert load_json(root / "eval_B_plus_log5/summary.json")["complete"]
    with pytest.raises(ValueError, match="NEW"):
        runner.run(screen)


@pytest.mark.parametrize("failure", ["checkpoint", "log", "command", "freeze", "receipt"])
def test_stale_inputs_fail_before_gpu(screen, monkeypatch, failure):
    root = Path(screen.screen_dir)
    if failure == "checkpoint":
        (root / "B/model_final.pth").write_bytes(b"changed")
    elif failure == "log":
        (root / "eval_B/log.txt").write_text("copypaste: " + ",".join(["1.0000"] * 9))
    elif failure == "command":
        path = root / "eval_B/command.json"
        command = load_json(path)
        command.append("model.beta=0.5")
        save_json(path, command)
    elif failure == "freeze":
        path = root / "summary.json"
        report = load_json(path)
        report["training"]["B"]["receipts"][0]["tpa_geometry_digest"] = "changed"
        save_json(path, report)
    else:
        path = root / "eval_B/verified_result.json"
        receipt = load_json(path)
        receipt["result"]["metrics"]["APr"] += 1
        save_json(path, receipt)
    monkeypatch.setattr(runner, "run_evaluation", lambda *args: pytest.fail("GPU started"))
    with pytest.raises(ValueError):
        runner.run(screen)
    assert not (root / "eval_B_plus_log5").exists()


@pytest.mark.parametrize("failure", ["nonzero", "swallowed", "missing_predictions", "checkpoint_changed"])
def test_failed_eval_has_no_success_summary(screen, monkeypatch, failure):
    def evaluate(command, output):
        if failure == "nonzero":
            raise subprocess.CalledProcessError(1, command)
        write_result(output, METRICS_B)
        if failure == "swallowed":
            (output / "console.log").write_text("Skipping evaluation")
        elif failure == "missing_predictions":
            (output / "lvis_instances_results.json").unlink()
        else:
            (Path(screen.screen_dir) / "B/model_final.pth").write_bytes(b"changed")
    monkeypatch.setattr(runner, "run_evaluation", evaluate)
    with pytest.raises((RuntimeError, ValueError, subprocess.CalledProcessError)):
        runner.run(screen)
    assert not (Path(screen.screen_dir) / "eval_B_plus_log5/summary.json").exists()
