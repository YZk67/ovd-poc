"""CPU-only guards for no-radius 4 -> 8ep / continuous 4 -> 12ep resume."""

import ast
import copy
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from tools.prepare_no_radius_8ep_resume import prepare, validate_checkpoint


def checkpoint():
    state = {}
    for head in (0, 1):
        prefix = f"transformer.decoder.class_embed.{head}.tpa."
        state[prefix + "prototype_queries"] = torch.zeros(5, 256)
        state[prefix + "prototype_mode_strength"] = torch.tensor(0.0)
        state[prefix + "slot_prior_strength"] = torch.tensor(0.2)
    return {
        "iteration": 28399, "model": state,
        "trainer": {
            "iteration": 28399, "lr_scheduler_max_iter": 85200,
            "gradient_accumulation_steps": 2,
            "optimizer": {"state": {0: {"step": 28400, "exp_avg": torch.ones(1)}},
                          "param_groups": [{"lr": 1e-4}, {"lr": 1e-3}]},
            "hooks": {"LRScheduler": {"last_epoch": 28400, "base_lrs": [1e-4, 1e-3]}},
        },
    }


def make_run(tmp_path):
    run = tmp_path / "no_radius_4ep"
    run.mkdir()
    torch.save(checkpoint(), run / "model_final.pth")
    (run / "last_checkpoint").write_text("model_final.pth")
    for name in ("config.yaml", "log.txt", "metrics.json", "lvis_instances_results.json"):
        (run / name).write_text("4ep evidence\n")
    return run


@pytest.mark.parametrize("target_epochs,stop", [(8, 56800), (12, 85200)])
def test_checkpoint_preflight_preserves_state_and_12ep_lr_horizon(target_epochs, stop):
    ckpt = checkpoint()
    report = validate_checkpoint(ckpt, target_epochs=target_epochs)
    assert report["resume_iteration"] == 28400
    assert report["stop_iteration"] == stop
    assert report["lr_scheduler_max_iter"] == 85200
    assert report["gradient_accumulation_steps"] == 2
    assert report["optimizer_lrs"] == [1e-4, 1e-3]
    assert ckpt["trainer"]["hooks"]["LRScheduler"]["last_epoch"] == 28400
    assert torch.equal(ckpt["trainer"]["optimizer"]["state"][0]["exp_avg"], torch.ones(1))


@pytest.mark.parametrize("target_epochs", [4, 10, 16])
def test_unsupported_continuation_target_is_rejected(target_epochs):
    with pytest.raises(ValueError, match="target_epochs must be 8 or 12"):
        validate_checkpoint(checkpoint(), target_epochs=target_epochs)


@pytest.mark.parametrize("key,value", [
    ("iteration", 14199), ("iteration", 28400),
    ("lr_scheduler_max_iter", 28400), ("lr_scheduler_max_iter", 56800),
    ("lr_scheduler_max_iter", None), ("gradient_accumulation_steps", 1),
])
def test_wrong_trainer_protocol_is_rejected(key, value):
    ckpt = checkpoint()
    ckpt["trainer"][key] = value
    with pytest.raises(ValueError, match=f"trainer.{key}"):
        validate_checkpoint(ckpt)


@pytest.mark.parametrize("iteration", [14199, 42599, 56799, 85199, None])
def test_preparation_only_accepts_completed_four_epoch_stage(iteration):
    ckpt = checkpoint()
    ckpt["iteration"] = iteration
    with pytest.raises(ValueError, match="completed 4ep"):
        validate_checkpoint(ckpt)


@pytest.mark.parametrize("missing", ["trainer", "optimizer", "hooks"])
def test_partial_training_state_is_rejected(missing):
    ckpt = checkpoint()
    if missing == "trainer":
        del ckpt[missing]
    else:
        del ckpt["trainer"][missing]
    with pytest.raises(ValueError):
        validate_checkpoint(ckpt)


@pytest.mark.parametrize("field,value", [("state", {}), ("param_groups", [])])
def test_empty_optimizer_is_rejected(field, value):
    ckpt = checkpoint()
    ckpt["trainer"]["optimizer"][field] = value
    with pytest.raises(ValueError, match="optimizer state"):
        validate_checkpoint(ckpt)


def test_reset_scheduler_and_decayed_optimizer_lr_are_rejected():
    ckpt = checkpoint()
    ckpt["trainer"]["hooks"]["LRScheduler"]["last_epoch"] = 0
    with pytest.raises(ValueError, match="last_epoch"):
        validate_checkpoint(ckpt)
    ckpt = checkpoint()
    ckpt["trainer"]["optimizer"]["param_groups"][0]["lr"] = 1e-5
    with pytest.raises(ValueError, match="high-LR plateau"):
        validate_checkpoint(ckpt)
    ckpt = checkpoint()
    ckpt["trainer"]["hooks"]["LRScheduler"]["base_lrs"] = []
    with pytest.raises(ValueError, match="base_lrs"):
        validate_checkpoint(ckpt)


@pytest.mark.parametrize("name,value", [
    ("prototype_queries", torch.zeros(1, 256)),
    ("prototype_mode_strength", torch.tensor(1.5)),
    ("prototype_mode_strength", torch.tensor(float("nan"))),
    ("slot_prior_strength", torch.tensor(0.0)),
])
def test_all_tpa_heads_must_match_no_radius_control(name, value):
    ckpt = checkpoint()
    ckpt["model"]["transformer.decoder.class_embed.1.tpa." + name] = value
    with pytest.raises(ValueError):
        validate_checkpoint(ckpt)


def test_missing_prototypes_or_radius_buffer_is_rejected():
    ckpt = checkpoint()
    ckpt["model"] = {}
    with pytest.raises(ValueError, match="No TPA"):
        validate_checkpoint(ckpt)
    ckpt = checkpoint()
    del ckpt["model"]["transformer.decoder.class_embed.0.tpa.prototype_mode_strength"]
    with pytest.raises(ValueError, match="prototype_mode_strength"):
        validate_checkpoint(ckpt)


def test_check_only_does_not_write_and_snapshot_is_independent_and_idempotent(tmp_path):
    run = make_run(tmp_path)
    originals = {p.name: p.read_bytes() for p in run.iterdir()}
    prepare(run)
    assert set(p.name for p in run.iterdir()) == set(originals)
    prepare(run, snapshot=True)
    archive = run / "four_ep_snapshot"
    manifest = json.loads((archive / "manifest.json").read_text())
    assert set(manifest["files"]) == set(originals)
    for name, value in originals.items():
        assert (run / name).read_bytes() == (archive / name).read_bytes() == value
        assert (run / name).stat().st_ino != (archive / name).stat().st_ino
    archive_stat = (archive / "model_final.pth").stat()
    prepare(run, snapshot=True)
    assert (archive / "model_final.pth").stat().st_ino == archive_stat.st_ino
    assert (archive / "model_final.pth").stat().st_mtime_ns == archive_stat.st_mtime_ns
    # Simulate training overwriting predictions; snapshot must stay untouched.
    (run / "lvis_instances_results.json").write_text("8ep predictions\n")
    assert (archive / "lvis_instances_results.json").read_bytes() == originals["lvis_instances_results.json"]
    with pytest.raises(ValueError, match="Snapshot/source mismatch"):
        prepare(run, snapshot=True)


def test_missing_marker_and_wrong_run_pointer_fail_before_writes(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    with pytest.raises(ValueError, match="Missing last_checkpoint"):
        prepare(run, snapshot=True)
    outside = tmp_path / "outside.pth"
    torch.save(checkpoint(), outside)
    (run / "last_checkpoint").write_text(str(outside))
    with pytest.raises(ValueError, match="directly in this run"):
        prepare(run, snapshot=True)
    assert not (run / "four_ep_snapshot").exists()


def test_twelve_epoch_target_reuses_identical_four_epoch_snapshot_without_writing(tmp_path):
    run = make_run(tmp_path)
    prepare(run, snapshot=True)
    archive = run / "four_ep_snapshot"
    before = {p.name: p.read_bytes() for p in archive.iterdir()}
    report = prepare(run, snapshot=True, target_epochs=12)
    assert report["stop_iteration"] == 85200
    assert {p.name: p.read_bytes() for p in archive.iterdir()} == before
    # Ignoring the future stop must NOT accept a changed source identity.
    manifest = json.loads((archive / "manifest.json").read_text())
    manifest["resume"]["gradient_accumulation_steps"] = 1
    (archive / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="different source"):
        prepare(run, snapshot=True, target_epochs=12)


def test_new_snapshot_records_twelve_epoch_plan(tmp_path):
    run = make_run(tmp_path)
    prepare(run, snapshot=True, target_epochs=12)
    manifest = json.loads((run / "four_ep_snapshot/manifest.json").read_text())
    assert manifest["resume"]["stop_iteration"] == manifest["resume"]["lr_scheduler_max_iter"] == 85200


def test_snapshot_requires_evidence_and_free_space(tmp_path, monkeypatch):
    from tools import prepare_no_radius_8ep_resume as module

    run = make_run(tmp_path)
    (run / "log.txt").unlink()
    with pytest.raises(ValueError, match="Missing 4ep evidence"):
        prepare(run, snapshot=True)
    (run / "log.txt").write_text("restored log")
    monkeypatch.setattr(module.shutil, "disk_usage", lambda _: SimpleNamespace(free=0))
    with pytest.raises(ValueError, match="insufficient space"):
        prepare(run, snapshot=True)
    assert not (run / "four_ep_snapshot").exists()


@pytest.mark.parametrize("target_epochs,stop", [(8, 56800), (12, 85200)])
def test_continuation_config_changes_only_stopping_point(monkeypatch, target_epochs, stop):
    module_name = "_resume_test_package.dino_convnext_large_4scale_4ep_lvis_no_radius"
    base = ModuleType(module_name)
    base.train = SimpleNamespace(max_iter=28400, lr_scheduler_max_iter=85200,
                                 output_dir="original_4ep_run", gradient_accumulation_steps=2,
                                 eval_period=28400, checkpointer=SimpleNamespace(period=14200))
    base.model = SimpleNamespace(classifier=SimpleNamespace(tpa_prototype_mode_strength=0.0))
    base.dataloader = SimpleNamespace(evaluator=SimpleNamespace(output_dir="original_4ep_run"))
    base.optimizer = SimpleNamespace(lr=1e-4)
    base.lr_multiplier = object()
    base.iterations_per_epoch = 7100
    before = copy.deepcopy(vars(base.train))
    monkeypatch.setitem(sys.modules, module_name, base)
    path = Path(__file__).resolve().parents[1] / f"lami_dino/configs/dino_convnext_large_4scale_{target_epochs}ep_lvis_no_radius.py"
    source = path.read_text()
    namespace = {"__package__": "_resume_test_package", "__name__": "_resume_test_package.continuation"}
    exec(compile(source, str(path), "exec"), namespace)
    assert vars(base.train) == dict(before, max_iter=stop)
    assert 56800 % base.train.checkpointer.period == 0
    assert 56800 % base.train.eval_period == 0
    assert base.model.classifier.tpa_prototype_mode_strength == 0.0
    for name in ("dataloader", "model", "optimizer", "lr_multiplier", "train"):
        assert namespace[name] is getattr(base, name)
    statements = ast.parse(source).body
    assert [ast.unparse(s.targets[0]) for s in statements if isinstance(s, ast.Assign)] == ["train.max_iter"]
    assert all(isinstance(s, (ast.Expr, ast.ImportFrom, ast.Assign)) for s in statements)


def test_actual_eval_hook_keeps_eight_epoch_eval_inside_twelve_epoch_run():
    # Exercise real hook control flow without importing Detectron2/CUDA. This
    # checks timing, not RNG/data-stream equivalence across a process restart.
    path = Path(__file__).resolve().parents[1] / "detectron2/detectron2/engine/hooks.py"
    node = next(n for n in ast.parse(path.read_text()).body
                if isinstance(n, ast.ClassDef) and n.name == "EvalHook")
    namespace = {"HookBase": object}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    calls = []
    hook = namespace["EvalHook"](28400, lambda: calls.append(hook.trainer.iter))
    hook.trainer = SimpleNamespace(iter=56799, max_iter=85200)
    hook._do_eval = hook._func  # Avoid evaluator side effects; test scheduling only.
    hook.after_step()
    assert calls == [56799] and callable(hook._func)
    hook.trainer.iter = 56800
    hook.after_step()
    assert calls == [56799]
    hook.trainer.iter = 85199
    hook.after_step()
    assert calls == [56799]
    hook.trainer.iter = 85200
    hook.after_train()
    assert calls == [56799, 85200]
    assert not hasattr(hook, "_func")
