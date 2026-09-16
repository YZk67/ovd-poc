from contextlib import nullcontext
from copy import deepcopy
from pathlib import Path
import subprocess
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from tools import audit_decoder_classification_sources as detail
from tools import audit_decoder_loss_sources as base
from tools.compare_rare_pr_reports import file_identity, save_json
from tools.decoder_loss_audit_ops import (
    CLASSIFICATION_PARTS, DETAIL_GROUPS, GROUPS, average_gradients,
    classification_reconstruction, component_gradients, directional_effects,
    gradient_summary, isolated_rng, split_classification_losses, summarize_probes,
)
from tools.diagnose_rare_fp_regions import fingerprint


def losses(p):
    return {"loss_class": p.sum(), "loss_class_0": 2*p.sum(), "loss_class_4": -p.sum(),
            "loss_class_dn": -3*p.sum(), "loss_class_dn_0": p.sum(),
            "loss_class_enc": torch.tensor(1.), "loss_bbox": 4*p.sum(),
            "loss_giou": -2*p.sum(), "loss_rpsa": torch.tensor(0.), "loss_apr": torch.tensor(1.)}


def test_exact_branch_partition_and_independent_total_gradient():
    p = torch.nn.Parameter(torch.tensor([1., 2.]))
    before = p.detach().clone()
    groups = split_classification_losses(losses(p))
    assert tuple(groups) == DETAIL_GROUPS
    assert groups["class_aux"] == ["loss_class_0", "loss_class_4"]
    assert groups["class_dn"] == ["loss_class_dn", "loss_class_dn_0"]
    children = [k for g in CLASSIFICATION_PARTS for k in groups[g]]
    assert len(children) == len(set(children)) and sorted(children) == sorted(groups["classification"])
    g, paths, _, _ = component_gradients(losses(p), (p,), splitter=split_classification_losses)
    assert g["class_encoder"] is None and paths["class_encoder"] == []
    for group, expected in (("class_final", 1.), ("class_aux", 1.), ("class_dn", -2.), ("classification", 0.)):
        torch.testing.assert_close(g[group], torch.full_like(p, expected))
    assert g["classification"] is not None  # Connected cancellation, not an unused parameter.
    assert classification_reconstruction(g)["relative_l2_error"] == 0
    assert torch.equal(before, p) and p.grad is None


@pytest.mark.parametrize("change", ["unknown", "missing", "encoder_connected"])
def test_fail_closed_unknown_branch_missing_or_encoder_path(change):
    p = torch.nn.Parameter(torch.ones(2))
    data = losses(p)
    if change == "unknown":
        data["loss_class_future"] = p.sum()
    elif change == "missing":
        data.pop("loss_class")
    else:
        data["loss_class_enc"] = p.sum()*0  # Even a connected ZERO must be reported, not silently excluded.
    with pytest.raises(ValueError):
        component_gradients(data, (p,), splitter=split_classification_losses)
    assert p.grad is None


def test_reconstruction_rejects_bad_sum_and_raw_not_unit_adds():
    gradients = {g: None for g in DETAIL_GROUPS}
    gradients.update(class_final=torch.tensor([1., 0.]), class_aux=torch.tensor([0., 2.]),
                     class_dn=torch.tensor([1., 0.]), classification=torch.tensor([2., 2.]))
    assert classification_reconstruction(gradients)["max_abs_error"] == 0
    averaged = average_gradients([gradients, gradients])
    assert tuple(averaged) == DETAIL_GROUPS
    effects = directional_effects(torch.tensor([3., 4.]), averaged)
    assert effects["classification"]["raw_descent_derivative"] == sum(
        effects[g]["raw_descent_derivative"] for g in CLASSIFICATION_PARTS)
    assert effects["classification"]["unit_descent_derivative"] != sum(
        effects[g]["unit_descent_derivative"] or 0 for g in CLASSIFICATION_PARTS)
    averaged["classification"] += 1
    with pytest.raises(ValueError, match="reconstruct"):
        classification_reconstruction(averaged)


def fixture_capture():
    p = torch.nn.Parameter(torch.ones(2))
    g, paths, keys, values = component_gradients(losses(p), (p,), splitter=split_classification_losses)
    micro = {"mapped_inputs": [{"image_id": 1, "image": "pixels"}], "fedloss_category_indices": [0],
             "connections": paths, "loss_keys": keys, "weighted_losses": values}
    summary = gradient_summary(g, None)
    window = {"train_annotations": {"sha256": "train"}, "microbatches": [micro], "summary": summary,
              "classification_reconstruction": classification_reconstruction(g),
              "parent_gradient_replay": {k: {"connected": g[k] is not None, "relative_l2_error": 0.}
                                         for k in GROUPS}}
    probes = [{"panel": "focus", "category": "cat", "image_id": 1, "kind": kind,
               "count": 1, "expected_count": 1, "matches": [{"query_id": i, "iou": 1.}],
               "effects": [directional_effects(torch.tensor([float(i+1), 0.]), g)]}
              for i, kind in enumerate(("tp", "fp"))]
    current = {"side": "old", "iteration": 56799, "parameters": ["core"], "weights_unchanged": True,
               "training_updates": 0, "optimizer_created": False, "windows": [window],
               "probes": probes, "control_coverage": []}
    parent = deepcopy(current)
    for w in parent["windows"]:
        w["summary"] = {k: v for k, v in w["summary"].items() if k in GROUPS}
        for row in w["microbatches"]:
            row["loss_keys"] = {k: v for k, v in row["loss_keys"].items() if k in GROUPS}
    for row in parent["probes"]:
        row["effects"] = [{k: v for k, v in e.items() if k in GROUPS} for e in row["effects"]]
    return current, parent


def test_verify_parent_replay_and_report_without_updates(tmp_path):
    current, parent = fixture_capture()
    ctx = SimpleNamespace(reference=parent, output=tmp_path, signature="s", identity={}, parent={"scope": []})
    save_json(tmp_path / "old_capture.json", current)
    result = detail.analyze(ctx, current)
    assert result["complete"] and result["parent_replay"]["verified"]
    assert result["training_updates"] == 0 and not result["optimizer_created"]
    assert {r["loss_group"] for r in result["local_margin_derivatives"]} == set(DETAIL_GROUPS)
    assert all(r["valid_windows"] == 0 for r in result["window_signs"] if r["source"] == "class_encoder")


@pytest.mark.parametrize("change", ["image", "fedloss", "loss", "norm", "probe", "region", "coverage", "windows"])
def test_parent_replay_rejects_drift(change):
    current, parent = fixture_capture()
    ctx = SimpleNamespace(reference=parent)
    micro = current["windows"][0]["microbatches"][0]
    if change == "image":
        micro["mapped_inputs"][0]["image"] = "other"
    elif change == "fedloss":
        micro["fedloss_category_indices"] = [1]
    elif change == "loss":
        micro["weighted_losses"]["loss_class"] += 1
    elif change == "norm":
        current["windows"][0]["summary"]["classification"]["norm"] += 1
    elif change == "probe":
        current["probes"][0]["effects"][0]["bbox_l1"]["raw_descent_derivative"] += 1
    elif change == "region":
        current["probes"][0]["matches"][0]["query_id"] = 3
    elif change == "coverage":
        current["control_coverage"] = [{"unexpected": True}]
    else:
        current["windows"] = []
    with pytest.raises(ValueError):
        detail.verify_replay(ctx, current)


def test_training_runner_split_replays_same_inputs_and_cache(tmp_path, monkeypatch):
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.core = torch.nn.Parameter(torch.ones(2))
            self.register_buffer("novel_idx", torch.tensor([False]))
            self.calls = 0

        def filter_content_info(self, data):
            return torch.tensor([0]), data

        def forward(self, data):
            self.calls += 1
            self.filter_content_info(data)
            return losses(self.core)

    batches = [[{"image_id": 1, "image": torch.zeros(3, 2, 2), "instances": SimpleNamespace(
        gt_classes=torch.tensor([0]), gt_boxes=SimpleNamespace(tensor=torch.zeros(1, 4)))}]]
    config, data, events = (ModuleType(n) for n in (
        "detectron2.config", "detectron2.data", "detectron2.utils.events"))
    config.instantiate = lambda _: deepcopy(batches)
    annotation = tmp_path / "train.json"
    save_json(annotation, {})
    data.MetadataCatalog = SimpleNamespace(get=lambda _: SimpleNamespace(json_file=annotation))
    events.EventStorage = lambda **_: nullcontext()
    for module in (config, data, events):
        monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(base, "isolated_rng", lambda seed, device: isolated_rng(seed))
    monkeypatch.setattr(torch.optim.Optimizer, "__init__", lambda *a, **k: pytest.fail("No optimizer"))
    cfg = SimpleNamespace(dataloader=SimpleNamespace(train=SimpleNamespace(
        dataset=SimpleNamespace(names="lvis_v1_train_norare"))))
    args = SimpleNamespace(seed=1, device="cuda:0", windows=1, microbatches=1, batch_size=1)
    model = Model()
    parent = base.training_windows(model, (model.core,), cfg, args,
        SimpleNamespace(output=tmp_path / "parent", signature="parent"), "old", 56799, None)
    ctx = SimpleNamespace(output=tmp_path / "split", signature="split", training_reference=parent,
                          loss_splitter=split_classification_losses, verify_micro=detail.verify_micro,
                          verify_window=lambda value, w: detail.verify_window(value, parent[w]))
    split = base.training_windows(model, (model.core,), cfg, args, ctx, "old", 56799, None)
    assert tuple(split[0]["gradients"]) == DETAIL_GROUPS and model.calls == 2
    assert split[0]["classification_reconstruction"]["relative_l2_error"] == 0
    assert split[0]["parent_gradient_replay"]["bbox_l1"]["relative_l2_error"] == 0
    assert torch.equal(model.core, torch.ones(2)) and model.core.grad is None
    base.training_windows(model, (model.core,), cfg, args, ctx, "old", 56799, None)
    assert model.calls == 2
    batches[0][0]["image"] += 1
    with pytest.raises(ValueError, match="augmentation differ"):
        base.training_windows(model, (model.core,), cfg, args, ctx, "old", 56799, None)


def test_run_old_only_and_analysis_never_gpu_fallback(tmp_path, monkeypatch):
    args = detail.parse_args(["--parent-audit", "parent", "--output-dir", str(tmp_path)])
    ctx = SimpleNamespace(output=tmp_path, signature="split", identity={})
    monkeypatch.setattr(detail, "prepare", lambda _: ctx)
    calls = []
    def capture(args, ctx, side, delta):
        calls.append((side, delta))
        return {"side": side}
    monkeypatch.setattr(base, "capture_side", capture)
    monkeypatch.setattr(detail, "analyze", lambda ctx, data: data)
    assert detail.run(args) == {"side": "old"} and calls == [("old", None)]
    args.analyze_only = True
    with pytest.raises(ValueError, match="never falls back"):
        detail.run(args)
    assert len(calls) == 1
    save_json(tmp_path / "old_capture.json", {"fingerprint": "split", "cached": True})
    assert detail.run(args)["cached"] and len(calls) == 1


def test_cli_from_outside_repo():
    result = subprocess.run([sys.executable, detail.__file__, "--help"], cwd="/tmp", capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "--parent-audit" in result.stdout and "--analyze-only" in result.stdout


def parent_files(tmp_path, monkeypatch):
    directory = tmp_path / "parent"
    annotation = tmp_path / "val.json"
    save_json(annotation, {"images": [], "annotations": [], "categories": []})
    sources = {"annotations": file_identity(annotation)}
    query = {"inputs": {"sources": sources, "stage_report": {"path": str(tmp_path / "stage/report.json")}},
             "interventions": {"old": {"native": {"regions": [{"image_id": 1, "category": "cat"}]}}}}
    query_path = tmp_path / "query.json"
    save_json(query_path, query)
    code_paths = ("tools/audit_decoder_loss_sources.py", "tools/decoder_loss_audit_ops.py", "tools/train_net.py")
    inputs = {"query_audit": file_identity(query_path), "sources": sources,
              "windows": 1, "microbatches": 1, "batch_size": 1, "seed": 1729, "device": "cuda:0",
              "focus_images": [1], "controls": [
                  {"image_id": 2, "panel": "control_rare"}, {"image_id": 3, "panel": "control_base"}],
              "torch_version": str(torch.__version__),
              "code": {k: file_identity(detail.ROOT / k)["sha256"] for k in code_paths}}
    signature = fingerprint(inputs)
    _, capture = fixture_capture()
    identities = {}
    for side in ("old", "new"):
        value = {**deepcopy(capture), "side": side, "iteration": 56799 if side == "old" else 70999,
                 "fingerprint": signature}
        save_json(directory / f"{side}_capture.json", value)
        identities[side] = file_identity(directory / f"{side}_capture.json")
    for image_id in (2, 3):
        save_json(directory / "new" / f"validation_{image_id}.json",
                  {"fingerprint": signature, "side": "new", "image_id": image_id, "probes": [], "anchor_regions": []})
    gradient_path = directory / "old/training_window_0.pt"
    gradient_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"fingerprint": signature}, gradient_path)
    parent = {"complete": True, "paired_training_inputs_verified": True, "training_updates": 0,
              "optimizer_created": False, "inputs": inputs, "fingerprint": signature,
              "captures": identities, "scope": []}
    path = directory / "report.json"
    save_json(path, parent)
    monkeypatch.setattr(detail, "validate_audit", lambda _: (query, None))
    return path, parent


def test_prepare_locks_parent_selection_budget_and_fresh_output(tmp_path, monkeypatch):
    path, parent = parent_files(tmp_path, monkeypatch)
    args = detail.parse_args(["--parent-audit", str(path), "--output-dir", str(tmp_path / "split"), "--prepare-only"])
    ctx = detail.prepare(args)
    assert ctx.identity["training_image_exposures"] == 1
    assert ctx.identity["iteration"] == 56799 and ctx.controls == parent["inputs"]["controls"]
    assert ctx.control_reference_dir == path.parent and ctx.loss_splitter is split_classification_losses
    assert args.seed == 1729 and args.microbatches == 1
    assert detail.prepare(args).signature == ctx.signature
    args.output_dir = str(path.parent / "child")
    with pytest.raises(ValueError, match="separate output"):
        detail.prepare(args)


@pytest.mark.parametrize("change", ["incomplete", "fingerprint", "code", "capture", "control"])
def test_parent_identity_drift_fails_before_gpu(tmp_path, monkeypatch, change):
    path, parent = parent_files(tmp_path, monkeypatch)
    if change == "incomplete":
        parent["complete"] = False
    elif change == "fingerprint":
        parent["fingerprint"] = "bad"
    elif change == "code":
        parent["inputs"]["code"]["tools/train_net.py"] = "bad"
        parent["fingerprint"] = fingerprint(parent["inputs"])
    elif change == "capture":
        save_json(path.parent / "old_capture.json", {"changed": True})
    else:
        cached = path.parent / "new/validation_2.json"
        save_json(cached, {"fingerprint": parent["fingerprint"], "side": "new", "image_id": 2,
                          "probes": [{"changed": True}], "anchor_regions": []})
    save_json(path, parent)
    with pytest.raises(ValueError):
        detail.validate_parent(path)


def test_full_vector_replay_catches_same_norm_changed_direction():
    prior = {"side": "old", "window": 0, "train_annotations": {},
             "gradients": {k: torch.tensor([1., 0.]) if k in GROUPS[:3] else None for k in GROUPS}}
    current = deepcopy(prior)
    assert detail.verify_window(current, prior)["classification"]["relative_l2_error"] == 0
    current["gradients"]["classification"] = torch.tensor([0., 1.])
    with pytest.raises(ValueError, match="full gradient vector"):
        detail.verify_window(current, prior)
