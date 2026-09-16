from copy import deepcopy
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from lami_dino.checkpoint_init import load_trusted_torch_file
from tools import evaluate_decoder_rollback as runner
from tools.compare_rare_pr_reports import file_identity, load_json, save_json
from tools.diagnose_rare_fp_regions import fingerprint
from tools.query_path_update_ops import canonical_name, key_group


def model_state(offset=0.):
    state = {
        **{f"transformer.decoder.layers.{i}.weight": torch.full((2, 2), offset + i)
           for i in range(6)},
        "transformer.decoder.norm.weight": torch.full((2,), offset),
        "transformer.decoder.ref_point_head.layers.0.weight": torch.full((2, 2), offset),
        "transformer.encoder.layers.0.weight": torch.full((2,), offset),
        "class_embed.5.linear.weight": torch.full((2, 2), offset),
        "class_embed.5.cls_bias": torch.tensor(offset),
        "bbox_embed.5.weight": torch.full((2,), offset),
        "backbone.stages.0.weight": torch.full((2,), offset),
        "thead.weight": torch.full((2,), offset),
        "class_embed.0.tpa.prototype_queries": torch.full((5, 256), offset),
        "class_embed.0.tpa.key_proj.weight": torch.full((2, 2), offset),
        "class_embed.0.tpa.key_proj.bias": torch.full((2,), offset),
        "class_embed.0.tpa.value_proj.weight": torch.full((2, 2), offset),
        "class_embed.0.tpa.value_proj.bias": torch.full((2,), offset),
        "class_embed.0.tpa.prototype_mode_strength": torch.tensor(0.),
        "class_embed.0.tpa.slot_prior_strength": torch.tensor(.2),
        "class_embed.0.tpa._step": torch.tensor(int(offset)),
    }
    # Real checkpoints serialize classifier, bbox and shared TPA aliases.
    for k, v in list(state.items()):
        if k.startswith(("class_embed.", "bbox_embed.")):
            state["transformer.decoder." + k] = v.clone()
    return state


def core_keys(state):
    return sorted(k for k in state if key_group(canonical_name(k)) == "decoder_core")


def test_exact_core_only_and_inputs_unchanged():
    old, new = model_state(1.), model_state(2.)
    before = deepcopy((old, new))
    hybrid, info = runner.hybrid_state(old, new, core_keys(new))
    for k in new:
        assert torch.equal(hybrid[k], old[k] if k in core_keys(new) else new[k])
        assert torch.equal(old[k], before[0][k])
        assert torch.equal(new[k], before[1][k])
    assert len(info["swapped_canonical_keys"]) == 8
    assert "bbox_embed.5.weight" in info["preserved_raw_keys"]
    assert "transformer.decoder.class_embed.5.linear.weight" in info["preserved_raw_keys"]
    runner.verify_hybrid({"model": hybrid, "decoder_rollback": {}}, old, new, core_keys(new))


@pytest.mark.parametrize("failure", ["alias", "nonfinite", "keys", "shape", "dtype", "audit_keys", "unchanged"])
def test_invalid_endpoints_fail(failure):
    old, new = model_state(1.), model_state(2.)
    keys = core_keys(new)
    if failure == "alias":
        old["transformer.decoder.bbox_embed.5.weight"] += 1
    elif failure == "nonfinite":
        new[keys[0]][0, 0] = float("nan")
    elif failure == "keys":
        old.pop(keys[0])
    elif failure == "shape":
        old[keys[0]] = torch.ones(3)
    elif failure == "dtype":
        old[keys[0]] = old[keys[0]].double()
    elif failure == "audit_keys":
        keys = keys[:-1]
    else:
        old = deepcopy(new)
    with pytest.raises(ValueError):
        runner.hybrid_state(old, new, keys)


def test_saved_verification_rejects_other_swaps_and_training_state():
    old, new = model_state(1.), model_state(2.)
    state, _ = runner.hybrid_state(old, new, core_keys(new))
    saved = {"model": deepcopy(state), "decoder_rollback": {}}
    saved["model"]["backbone.stages.0.weight"] = old["backbone.stages.0.weight"]
    with pytest.raises(ValueError, match="verification failed"):
        runner.verify_hybrid(saved, old, new, core_keys(new))
    with pytest.raises(ValueError, match="weights-only"):
        runner.verify_hybrid({"model": state, "decoder_rollback": {}, "optimizer": {}},
                             old, new, core_keys(new))


@pytest.mark.parametrize("change", ["iteration", "radius", "kp", "prior", "prefix_collision"])
def test_endpoint_protocol_checks(change):
    state = model_state(1.)
    checkpoint = {"model": state, "iteration": 70999}
    if change == "iteration":
        checkpoint["iteration"] = 85199
    elif change == "prefix_collision":
        state["module.thead.weight"] = state["thead.weight"]
    else:
        suffix, value = {
            "radius": ("prototype_mode_strength", torch.tensor(1.5)),
            "kp": ("prototype_queries", torch.ones(1, 256)),
            "prior": ("slot_prior_strength", torch.tensor(0.)),
        }[change]
        for k in list(state):
            if k.endswith("tpa." + suffix):
                state[k] = value
    with pytest.raises(ValueError):
        runner.endpoint_state(checkpoint, 70999)


def write_results(output, metrics=None):
    metrics = metrics or runner.BASELINE
    (output / "log.txt").write_text("copypaste: " + ",".join(f"{metrics[k]:.4f}" for k in runner.METRICS))
    (output / "console.log").write_text("Evaluation completed\n")
    (output / "lvis_instances_results.json").write_text('[{"image_id": 1}]')


@pytest.fixture
def inputs(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "model_code_hash", lambda _: "model_hash")
    sources = {}
    for side, iteration in (("old", 56799), ("new", 70999)):
        path = tmp_path / (side + ".pth")
        torch.save({"model": model_state(1. if side == "old" else 2.), "iteration": iteration,
                    "optimizer": {"a": 1}, "trainer": {"a": 2}, "scheduler": {"a": 3}}, path)
        sources[side + "_checkpoint"] = file_identity(path)
    for name in ("annotations", "prompt_bank", "config_file"):
        path = tmp_path / name
        path.write_text("locked " + name)
        sources[name] = file_identity(path)
    baseline = tmp_path / "baseline"
    baseline.mkdir()
    write_results(baseline)
    sources["new_predictions"] = file_identity(baseline / "lvis_instances_results.json")
    stage = tmp_path / "stage/report.json"
    save_json(stage, {"complete": True})
    protocol = {**runner.PROTOCOL, "code_sha256": "model_hash", "asset_sha256": {},
                "old_sha256": sources["old_checkpoint"]["sha256"],
                "new_sha256": sources["new_checkpoint"]["sha256"]}
    save_json(stage.parent / "pairing_cache/manifest.json", {
        "inputs": protocol, "fingerprint": fingerprint(protocol),
    })
    report = {"complete": True,
              "inputs": {"sources": sources, "stage_report": file_identity(stage),
                         "parent_fingerprint": fingerprint(protocol), "variants": ["decoder_core"],
                         "model_code_sha256": "model_hash", "audit_code": {}},
              "inventory": {"decoder_core": {"keys": core_keys(model_state())}}}
    audit = tmp_path / "audit.json"
    save_json(audit, report)
    return runner.parse_args(["--audit-report", str(audit), "--output-dir", str(tmp_path / "eval"),
                              "--cpu-threads", "1"])


def test_prepare_weights_only_roundtrip_no_gpu(inputs, monkeypatch):
    inputs.prepare_only = True
    monkeypatch.setattr(runner, "run_evaluation", lambda *a: pytest.fail("GPU must not start"))
    manifest = runner.run(inputs)
    saved = load_trusted_torch_file(manifest["hybrid_checkpoint"]["path"])
    assert set(saved) == {"model", "decoder_rollback"}
    assert saved["decoder_rollback"]["eval_only"] is True
    assert not (Path(inputs.output_dir) / "summary.json").exists()
    for side in ("old", "new"):
        expected = manifest["provenance"]["sources"][side]
        assert file_identity(expected["path"])["sha256"] == expected["sha256"]
    with pytest.raises(ValueError, match="NEW output"):
        runner.run(inputs)


def test_one_formal_evaluation_exact_protocol_and_summary(inputs, monkeypatch):
    calls = []

    def fake_eval(command, output):
        calls.append(command)
        assert command[0] == sys.executable
        assert "--eval-only" in command and "--resume" not in command and "--ddebug" not in command
        assert "model.classifier.tpa_prototype_mode_strength=0.0" in command
        assert "model.tpa_eval_mode_scale=1.0" in command
        assert "model.inference_query_class_topk=0" in command
        assert "dataloader.test.dataset.names=lvis_v1_val" in command
        assert "model.beta=0.3" in command and "model.novel_scale=3.0" in command
        write_results(output, {**runner.BASELINE, "AP": 41.0, "APr": 43.0})

    monkeypatch.setattr(runner, "run_evaluation", fake_eval)
    result = runner.run(inputs)
    assert len(calls) == 1
    assert result["delta_hybrid_minus_native10"]["APr"] == pytest.approx(.6969)
    assert result["delta_hybrid_minus_native10"]["AP"] == pytest.approx(-.7624)
    assert load_json(Path(inputs.output_dir) / "summary.json")["complete"]


@pytest.mark.parametrize("failure", ["incomplete_audit", "checkpoint_changed", "model_changed", "baseline_changed"])
def test_preflight_failure_does_not_start_gpu_or_write_outputs(inputs, monkeypatch, failure):
    audit = load_json(inputs.audit_report)
    sources = audit["inputs"]["sources"]
    if failure == "incomplete_audit":
        audit["complete"] = False
        save_json(inputs.audit_report, audit)
    elif failure == "checkpoint_changed":
        Path(sources["new_checkpoint"]["path"]).write_bytes(b"changed")
    elif failure == "model_changed":
        monkeypatch.setattr(runner, "model_code_hash", lambda _: "different")
    else:
        baseline = Path(sources["new_predictions"]["path"]).parent / "log.txt"
        baseline.write_text("copypaste: " + ",".join(["1.0000"] * 9))
    monkeypatch.setattr(runner, "run_evaluation", lambda *a: pytest.fail("GPU must not start"))
    with pytest.raises(ValueError):
        runner.run(inputs)
    assert not Path(inputs.output_dir).exists()


@pytest.mark.parametrize("failure", ["swallowed", "no_metrics", "no_predictions", "nonzero"])
def test_failed_eval_does_not_emit_success(inputs, monkeypatch, failure):
    def fake_eval(command, output):
        if failure == "nonzero":
            raise subprocess.CalledProcessError(1, command)
        write_results(output)
        if failure == "swallowed":
            (output / "console.log").write_text("Skipping evaluation due to failure")
        elif failure == "no_metrics":
            (output / "log.txt").write_text("No bbox result")
        else:
            (output / "lvis_instances_results.json").unlink()
    monkeypatch.setattr(runner, "run_evaluation", fake_eval)
    with pytest.raises((ValueError, RuntimeError, subprocess.CalledProcessError)):
        runner.run(inputs)
    assert not (Path(inputs.output_dir) / "summary.json").exists()


def test_console_is_streamed_and_retained(tmp_path):
    runner.run_evaluation([sys.executable, "-u", "-c", "print('mock LVIS progress')"], tmp_path)
    assert "mock LVIS progress" in (tmp_path / "console.log").read_text()


def test_help_without_detectron2():
    script = Path(runner.__file__).resolve()
    result = subprocess.run([sys.executable, str(script), "--help"], cwd="/tmp", capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "--audit-report" in result.stdout
