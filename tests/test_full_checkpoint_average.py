from copy import deepcopy
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from lami_dino.checkpoint_init import load_trusted_torch_file
from tools import evaluate_full_checkpoint_average as runner
from tools.compare_rare_pr_reports import file_identity, load_json, save_json


def model_state(offset, step):
    state = {
        "transformer.encoder.layers.0.weight": torch.full((2, 2), offset),
        "transformer.decoder.layers.0.weight": torch.full((2, 2), offset + 2),
        "class_embed.5.linear.weight": torch.full((2, 2), offset + 4),
        "bbox_embed.5.weight": torch.full((2,), offset + 6),
        # Frozen CLIP trunk must be bit-identical across the trajectory.
        "backbone.stages.0.weight": torch.full((2,), 8.),
        "class_embed.0.tpa.prototype_queries": torch.full((5, 256), offset + 10),
        "class_embed.0.tpa.key_proj.weight": torch.full((2, 2), offset + 11),
        "class_embed.0.tpa.key_proj.bias": torch.full((2,), offset + 12),
        "class_embed.0.tpa.value_proj.weight": torch.full((2, 2), offset + 13),
        "class_embed.0.tpa.value_proj.bias": torch.full((2,), offset + 14),
        "class_embed.0.tpa.prototype_mode_strength": torch.tensor(0.),
        "class_embed.0.tpa.slot_prior_strength": torch.tensor(.2),
        "class_embed.0.tpa._step": torch.tensor(step, dtype=torch.int64),
    }
    for key, value in list(state.items()):
        if key.startswith(("class_embed.", "bbox_embed.")):
            state["transformer.decoder." + key] = value.clone()
    return state


def write_eval(directory, metrics=None):
    metrics = metrics or runner.BASELINE
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "log.txt").write_text(
        "copypaste: " + ",".join(f"{metrics[key]:.4f}" for key in runner.METRICS)
    )
    (directory / "console.log").write_text("Evaluation completed\n")
    (directory / "lvis_instances_results.json").write_text('[{"image_id": 1}]')


def test_exact_joint_average_and_inputs_unchanged():
    early, late = model_state(1., 10), model_state(3., 20)
    before = deepcopy((early, late))
    averaged, info = runner.average_state(early, late)
    for key in late:
        expected = runner.mean_tensor(early[key], late[key]) if late[key].is_floating_point() else late[key]
        assert torch.equal(averaged[key], expected)
        assert torch.equal(early[key], before[0][key])
        assert torch.equal(late[key], before[1][key])
    assert "class_embed.0.tpa._step" in info["late_nonfloating_counter_keys"]
    assert set(info["floating_raw_keys"]) == {k for k, v in late.items() if v.is_floating_point()}
    assert not any(key.startswith("optimizer") for key in averaged)
    runner.verify_average({"model": averaged, "full_checkpoint_average": {}}, early, late)


def test_half_average_uses_stable_work_dtype():
    early = torch.tensor([1., 2.], dtype=torch.float16)
    late = torch.tensor([3., 4.], dtype=torch.float16)
    value = runner.mean_tensor(early, late)
    assert value.dtype == torch.float16
    assert torch.equal(value, torch.tensor([2., 3.], dtype=torch.float16))


@pytest.mark.parametrize("failure", [
    "keys", "shape", "dtype", "nonfinite", "alias", "integer", "frozen", "protocol", "unchanged",
])
def test_invalid_endpoints_fail_closed(failure):
    early, late = model_state(1., 10), model_state(3., 20)
    if failure == "keys":
        early.pop("bbox_embed.5.weight")
    elif failure == "shape":
        early["transformer.encoder.layers.0.weight"] = torch.ones(3)
    elif failure == "dtype":
        early["transformer.encoder.layers.0.weight"] = torch.ones(2, 2).double()
    elif failure == "nonfinite":
        early["transformer.encoder.layers.0.weight"][0, 0] = float("nan")
    elif failure == "alias":
        early["transformer.decoder.class_embed.0.tpa.key_proj.weight"] += 1
    elif failure == "integer":
        early["transformer.decoder.layers.0.counter"] = torch.tensor(1)
        late["transformer.decoder.layers.0.counter"] = torch.tensor(2)
    elif failure == "frozen":
        late["backbone.stages.0.weight"] += 1
    elif failure == "protocol":
        for key in list(late):
            if key.endswith("tpa.slot_prior_strength"):
                late[key] = torch.tensor(.3)
    else:
        early = deepcopy(late)
    with pytest.raises(ValueError):
        runner.average_state(early, late)


def test_saved_verification_rejects_wrong_average_and_training_state():
    early, late = model_state(1., 10), model_state(3., 20)
    averaged, _ = runner.average_state(early, late)
    bad = deepcopy(averaged)
    bad["transformer.encoder.layers.0.weight"][0, 0] += 1
    with pytest.raises(ValueError, match="verification failed"):
        runner.verify_average({"model": bad, "full_checkpoint_average": {}}, early, late)
    with pytest.raises(ValueError, match="weights-only"):
        runner.verify_average({"model": averaged, "full_checkpoint_average": {}, "iteration": 1},
                              early, late)


@pytest.fixture
def setup_run(tmp_path, monkeypatch):
    trajectory = tmp_path / "trajectory"
    trajectory.mkdir()
    early_path = trajectory / "model_0056799.pth"
    late_path = trajectory / "model_final.pth"
    torch.save({"model": model_state(1., 10), "iteration": runner.EARLY_ITERATION,
                "optimizer": {"must": "not copy"}}, early_path)
    torch.save({"model": model_state(3., 20), "iteration": runner.LATE_ITERATION,
                "scheduler": {"must": "not copy"}}, late_path)
    write_eval(trajectory)
    config = tmp_path / "no_radius.py"
    config.write_text("locked config")
    audit_path = tmp_path / "audit.json"
    save_json(audit_path, {"complete": True})
    report = {"inputs": {"sources": {
        "old_checkpoint": file_identity(early_path), "config_file": file_identity(config),
    }}}
    monkeypatch.setattr(runner, "validate_audit", lambda _: (report, {}))
    args = runner.parse_args(["--audit-report", str(audit_path),
                              "--output-dir", str(tmp_path / "average_eval"),
                              "--cpu-threads", "1"])
    return args, trajectory


def test_prepare_only_is_weights_only_and_never_starts_gpu(setup_run, monkeypatch):
    args, trajectory = setup_run
    args.prepare_only = True
    monkeypatch.setattr(runner, "run_evaluation", lambda *x: pytest.fail("GPU must not start"))
    manifest = runner.run(args)
    saved = load_trusted_torch_file(manifest["average_checkpoint"]["path"])
    assert set(saved) == {"model", "full_checkpoint_average"}
    assert saved["full_checkpoint_average"]["ratio_early"] == .5
    assert not (Path(args.output_dir) / "summary.json").exists()
    assert file_identity(trajectory / "model_final.pth")["sha256"] == manifest["sources"]["late_checkpoint"]["sha256"]
    with pytest.raises(ValueError, match="NEW output"):
        runner.run(args)


def test_one_formal_eval_and_precommitted_gate(setup_run, monkeypatch):
    args, _ = setup_run
    calls = []

    def fake_eval(command, output):
        calls.append(command)
        assert command[0] == sys.executable
        assert "--eval-only" in command and "--resume" not in command and "--ddebug" not in command
        assert "model.classifier.tpa_prototype_mode_strength=0.0" in command
        assert "model.beta=0.3" in command and "model.novel_scale=3.0" in command
        write_eval(output, {**runner.BASELINE, "AP": 44.5, "APr": 43.0})

    monkeypatch.setattr(runner, "run_evaluation", fake_eval)
    result = runner.run(args)
    assert len(calls) == 1
    assert result["passes_all_thresholds"] is True
    assert result["delta_average_minus_native12"]["AP"] == pytest.approx(-.1979)
    assert result["delta_average_minus_native12"]["APr"] == pytest.approx(.5771)
    assert load_json(Path(args.output_dir) / "summary.json")["complete"]


@pytest.mark.parametrize("failure", ["wrong_metrics", "missing_predictions"])
def test_baseline_must_be_exact_and_complete(tmp_path, failure):
    write_eval(tmp_path)
    if failure == "wrong_metrics":
        (tmp_path / "log.txt").write_text("copypaste: " + ",".join(["1.0000"] * 9))
    else:
        (tmp_path / "lvis_instances_results.json").unlink()
    with pytest.raises((ValueError, FileNotFoundError)):
        runner.validate_baseline(tmp_path)


@pytest.mark.parametrize("failure", ["swallowed", "no_metrics", "no_predictions"])
def test_failed_eval_never_emits_success(tmp_path, failure):
    write_eval(tmp_path)
    if failure == "swallowed":
        (tmp_path / "console.log").write_text("Skipping evaluation because evaluator failed")
    elif failure == "no_metrics":
        (tmp_path / "log.txt").write_text("No bbox metrics")
    else:
        (tmp_path / "lvis_instances_results.json").unlink()
    with pytest.raises((ValueError, RuntimeError)):
        runner.collect_results(tmp_path)


def test_help_without_detectron2():
    result = subprocess.run([sys.executable, str(Path(runner.__file__).resolve()), "--help"],
                            cwd="/tmp", capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "--audit-report" in result.stdout
