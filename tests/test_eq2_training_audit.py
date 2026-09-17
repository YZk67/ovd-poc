import ast
from copy import deepcopy
import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Tuple

import pytest
import torch
import torch.nn.functional as F

from lami_dino.prototype_ops import calibrated_logmeanexp_similarity, legacy_uncalibrated_logsumexp_similarity
from tools import audit_eq2_training_stages as runner
from tools.eq2_training_audit_ops import (
    VARIANTS, assemble_records, assignment_change, assignment_signature, audit_branch,
    capture_head_inputs, compare_stages, detached_matcher, focal_cells, formula,
    gradient_probe, targets_for,
)

ROOT = Path(__file__).resolve().parents[1]
SETTINGS = dict(cost_class=2., cost_bbox=5., cost_giou=2., alpha=.25, gamma=2., cost_class_type="focal_loss_cost")


def test_formula_parity_duplicate_bias_and_k1():
    torch.manual_seed(4)
    x, p = F.normalize(torch.randn(2, 4, 7), dim=-1), F.normalize(torch.randn(3, 5, 7), dim=-1)
    sim = torch.einsum("bqd,ckd->bqck", x, p)
    torch.testing.assert_close(formula(sim, "calibrated", 50., .07),
                               calibrated_logmeanexp_similarity(x, p, temperature=.07, logit_scale=50.))
    torch.testing.assert_close(formula(sim, "legacy", 50., .07),
                               legacy_uncalibrated_logsumexp_similarity(x, p, logit_scale=50.))
    assert not torch.allclose(formula(sim, "legacy", 50., .07), formula(sim, "calibrated_plus_logK", 50., .07))
    duplicate = sim[..., :1].expand_as(sim)
    torch.testing.assert_close(formula(duplicate, "legacy", 50., .07),
                               formula(duplicate, "calibrated", 50., .07)+math.log(5))
    for variant in VARIANTS:
        torch.testing.assert_close(formula(sim[..., :1], variant, 50., .07), sim[..., 0]*50)


def native_symbols(path, names, env):
    """Execute actual source definitions without importing the CUDA/D2 package."""
    tree = ast.parse((ROOT / path).read_text())
    tree.body = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
    exec(compile(tree, str(path), "exec"), env)
    return env


def test_focal_value_and_gradients_match_actual_native_source():
    native = native_symbols("detrex/modeling/criterion/criterion.py", {"sigmoid_focal_loss"}, {"torch": torch, "F": F})
    z = torch.randn(2, 5, 4, dtype=torch.double, requires_grad=True)
    labels = torch.zeros_like(z)
    labels[0, 2, 1] = labels[1, 0, 3] = 1
    a = native["sigmoid_focal_loss"](z, labels, 2., .25, 2.)*5*2
    b = focal_cells(z, labels, .25, 2., 2., 2.).sum()
    torch.testing.assert_close(a, b)
    torch.testing.assert_close(torch.autograd.grad(a, z, retain_graph=True)[0], torch.autograd.grad(b, z)[0])


def test_matcher_matches_native_source_with_empty_image():
    from scipy.optimize import linear_sum_assignment
    env = {"torch": torch, "Tuple": Tuple, "box_area": lambda b: (b[:, 2]-b[:, 0])*(b[:, 3]-b[:, 1])}
    native_symbols("detrex/layers/box_ops.py", {"box_cxcywh_to_xyxy", "box_iou", "generalized_box_iou"}, env)
    env.update(nn=torch.nn, linear_sum_assignment=linear_sum_assignment)
    native_symbols("detrex/modeling/matcher/matcher.py", {"HungarianMatcher"}, env)
    torch.manual_seed(7)
    outputs = {"pred_logits": torch.randn(2, 9, 4), "pred_boxes": torch.rand(2, 9, 4)}
    targets = [{"labels": torch.tensor([0, 3, 1]), "boxes": torch.rand(3, 4)},
               {"labels": torch.empty(0, dtype=torch.long), "boxes": torch.empty(0, 4)}]
    a = env["HungarianMatcher"](**SETTINGS)(outputs, targets)
    b = detached_matcher(SETTINGS)(outputs, targets)
    assert assignment_signature(a) == assignment_signature(b)


def record_fixture(empty=False, dn=False):
    # Same boxes: calibrated favors two similar modes; legacy favors one peak.
    x = torch.tensor([[[1., 0., 0.], [0., 1., 0.]]])
    p = torch.tensor([[[.1, .12, math.sqrt(1-.1**2-.12**2)], [.1, -.2, math.sqrt(1-.1**2-.2**2)]]])
    z = formula(torch.einsum("bqd,ckd->bqck", x, p), "calibrated", 50., .07)
    settings = {**SETTINGS, "cost_bbox": 0., "cost_giou": 0.}
    boxes = torch.tensor([[[.5, .5, .2, .2]]]).expand(1, 2, 4).clone()
    targets = [{"labels": torch.empty(0, dtype=torch.long) if empty else torch.tensor([0]),
                "boxes": torch.empty(0, 4) if empty else boxes[0, :1]}]
    indices = detached_matcher(settings)({"pred_logits": z, "pred_boxes": boxes}, targets)
    labels = targets_for(z, targets, indices)
    record = dict(features=x, prototypes=p, logits=z, boxes=boxes, targets=targets, indices=indices,
                  scale=50., tau=.07, bias=0., name="loss_class", group="final", hungarian=not dn,
                  weight=2., normalizer=1., alpha=.25, gamma=2.)
    record["weighted_native_loss"] = float(focal_cells(z, labels, .25, 2., 1., 2.).sum())
    return record, settings


def test_fixed_assignment_vs_rematched_changes_only_target_and_formula():
    record, settings = record_fixture()
    saved = deepcopy(record)
    report = audit_branch(record, detached_matcher(settings))
    assert report["variants"]["calibrated"]["matching"]["changed"] == 0
    legacy = report["variants"]["legacy"]
    assert legacy["matching"]["changed_fraction"] == 1.
    assert legacy["fixed_assignment"]["total"]["weighted_loss"] != legacy["rematched"]["total"]["weighted_loss"]
    for k in ("features", "prototypes", "boxes", "logits"):
        assert torch.equal(record[k], saved[k]) and record[k].grad is None
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize("empty,dn", [(True, False), (False, True)])
def test_empty_positive_group_and_dn_not_rematched(empty, dn):
    record, settings = record_fixture(empty, dn)
    def fail_match(*_):
        raise AssertionError("DN is not Hungarian matched")
    report = audit_branch(record, fail_match if dn else detached_matcher(settings))
    for v in report["variants"].values():
        assert v["matching"]["changed"] == 0
        if empty:
            pos = v["fixed_assignment"]["positive"]
            assert pos["cells"] == 0 and pos["mean_abs_logit_gradient"] is None
            assert not any(pos["category_has_nonzero_slot_gradient"])
            assert pos["prototype_output_gradient_l2"] == 0


def test_normalization_gradcheck_and_positive_negative_closure():
    torch.manual_seed(12)
    x = torch.randn(1, 3, 4, dtype=torch.double, requires_grad=True)
    p = torch.randn(2, 2, 4, dtype=torch.double, requires_grad=True)
    labels = torch.zeros(1, 3, 2, dtype=torch.double)
    labels[0, 1, 0] = 1
    options = dict(alpha=.25, gamma=2., normalizer=1., weight=2.)
    def get_z(x, p):
        sim = torch.einsum("bqd,ckd->bqck", F.normalize(x, dim=-1), F.normalize(p, dim=-1))
        return formula(sim, "calibrated", 5., .07)
    fn = lambda x, p: focal_cells(get_z(x, p), labels, **options).sum()
    assert torch.autograd.gradcheck(fn, (x, p))
    report, vectors = gradient_probe(x, p, get_z(x, p), labels, options)
    gx, gp = torch.autograd.grad(fn(x, p), (x, p))
    torch.testing.assert_close(gx, vectors["total"][0])
    torch.testing.assert_close(gp, vectors["total"][1])
    assert report["total"]["weighted_loss"] == pytest.approx(float(fn(x, p).detach()))
    assert x.grad is None and p.grad is None


@pytest.mark.parametrize("field", ["logits", "weighted_native_loss", "indices", "features"])
def test_native_replay_fails_closed(field):
    record, settings = record_fixture()
    if field == "indices":
        record[field] = [(torch.tensor([1]), torch.tensor([0]))]
    elif field == "features":
        record[field][0, 0, 0] = float("nan")
    else:
        record[field] += 1
    with pytest.raises(ValueError):
        audit_branch(record, detached_matcher(settings))


class FakeHead(torch.nn.Module):
    def __init__(self, prototypes):
        super().__init__()
        self._external_prototypes = prototypes
        self.norm_weight, self.use_bias = True, False
        self.norm_temperature, self.tpa_cls_tau = 5., .07

    def _compute_tpa_logits(self, x, *, content_inds, additional_class):
        sim = torch.einsum("bqd,ckd->bqck", F.normalize(x, dim=-1), F.normalize(self._external_prototypes, dim=-1))
        return formula(sim, "calibrated", 5., .07)


class FakeModel:
    def __init__(self):
        torch.manual_seed(31)
        self.num_queries = 3
        self.transformer = SimpleNamespace(decoder=SimpleNamespace(num_layers=2))
        p = torch.randn(2, 2, 4)
        self.class_embed = [FakeHead(p) for _ in range(3)]
        self.x = [torch.randn(1, 5, 4) for _ in range(2)] + [torch.randn(1, 3, 4)]
        self.criterion = SimpleNamespace(alpha=.25, gamma=2., weight_dict={
            "loss_class": 2., "loss_class_0": 2., "loss_class_enc": 2., "loss_class_dn": 2., "loss_class_dn_0": 2.})
        def loss_labels(outputs, targets, indices, num_boxes):
            z = outputs["pred_logits"]
            return {"loss_class": focal_cells(z, targets_for(z, targets, indices), .25, 2., num_boxes, 1.).sum()}
        self.criterion.loss_labels = loss_labels

    def filter_content_info(self, data):
        return torch.tensor([0, 1]), data

    def forward(self):
        ids, _ = self.filter_content_info(None)
        kwargs = dict(content_inds=ids, additional_class=None)
        # Large spatial-token call is not a criterion input.
        self.class_embed[2]._compute_tpa_logits(torch.ones(1, 7, 4), **kwargs)
        zs = [h._compute_tpa_logits(x, **kwargs) for h, x in zip(self.class_embed, self.x)]
        losses = {}
        specs = [("loss_class", 1, False), ("loss_class_0", 0, False), ("loss_class_enc", 2, False),
                 ("loss_class_dn", 1, True), ("loss_class_dn_0", 0, True)]
        for name, layer, dn in specs:
            z = zs[layer] if layer == 2 else (zs[layer][:, :2] if dn else zs[layer][:, 2:])
            indices = [(torch.tensor([0, 1]) if dn else torch.tensor([0]), torch.tensor([0, 0]) if dn else torch.tensor([0]))]
            t = [{"labels": torch.tensor([1]), "boxes": torch.ones(1, 4)}]
            loss = self.criterion.loss_labels({"pred_logits": z, "pred_boxes": torch.ones(1, z.shape[1], 4)}, t, indices, 2 if dn else 1)
            losses[name] = loss["loss_class"]*2
        return losses


def test_native_capture_hooks_restore_and_dn_slices_reconstruct():
    model = FakeModel()
    baseline = model.forward()
    original = model.class_embed[0]._compute_tpa_logits
    with capture_head_inputs(model) as captured:
        losses = model.forward()
        records = assemble_records(model, captured, losses)
    assert model.class_embed[0]._compute_tpa_logits == original
    for name in losses:
        torch.testing.assert_close(losses[name], baseline[name])
    assert [r["group"] for r in records] == ["final", "auxiliary", "encoder", "denoising", "denoising"]
    assert [r["features"].shape[1] for r in records] == [3, 3, 3, 2, 2]
    for r in records:
        sim = torch.einsum("bqd,ckd->bqck", F.normalize(r["features"], dim=-1), F.normalize(r["prototypes"], dim=-1))
        torch.testing.assert_close(formula(sim, "calibrated", r["scale"], r["tau"]), r["logits"])
    with pytest.raises(RuntimeError), capture_head_inputs(model):
        raise RuntimeError("stop")
    assert model.class_embed[0]._compute_tpa_logits == original


def metadata():
    return dict(mapped_inputs=[{"image_id": 1}], category_indices=[0, 1], sampled_rare_count=0,
                branch_layout=[{"name": "loss_class"}], rng_after_forward=["cpu", "cuda"])


@pytest.mark.parametrize("key", list(metadata()))
def test_pairing_rejects_changed_images_fedloss_layout_rng(key):
    a = metadata()
    b = deepcopy(a)
    b[key] = "different"
    with pytest.raises(ValueError, match="not paired"):
        runner.paired_metadata(a, b)


def test_summary_separates_formula_effect_from_stage_effect():
    record, settings = record_fixture()
    branch = audit_branch(record, detached_matcher(settings))
    stages = {s: [{"branches": [deepcopy(branch)]}] for s in runner.STAGES}
    report = compare_stages(stages)
    for variant in VARIANTS[1:]:
        for value in report["formula_stage_interactions"]["all"][variant].values():
            if value is not None:
                assert value["stage_difference_of_formula_effect"] == 0
    assert report["stage_summaries"]["8ep"]["denoising"] is None


def args_for(tmp_path, extra=()):
    return runner.parse_args(["--checkpoint-8ep", str(tmp_path/"8.pth"), "--checkpoint-12ep", str(tmp_path/"12.pth"),
                              "--output-dir", str(tmp_path/"out"), *extra])


def test_budget_and_analyze_only_no_gpu_fallback(tmp_path, monkeypatch):
    args = args_for(tmp_path, ["--analyze-only"])
    assert runner.check_budget(args) == 16
    args.batches = 9
    with pytest.raises(ValueError, match="Budget"):
        runner.check_budget(args)
    args.batches = 4
    monkeypatch.setattr(runner, "prepare", lambda a: (tmp_path, {"fingerprint": "test"}))
    monkeypatch.setattr(runner, "load_capture", lambda *a: None)
    monkeypatch.setattr(runner, "capture_stage", lambda *a: pytest.fail("Must not capture"))
    with pytest.raises(ValueError, match="will not run GPU"):
        runner.run(args)


def test_capture_receipt_rejects_modified_cache(tmp_path):
    payload = tmp_path/"b.pt"
    torch.save({"x": torch.ones(2)}, payload)
    receipt = dict(fingerprint="s", side="8ep", iteration=56799, weights_unchanged=True,
                   assets=[], train_annotations=runner.file_identity(payload),
                   batches=[{"cache": runner.file_identity(payload)}])
    runner.save_json(tmp_path/"8ep_capture.json", receipt)
    assert runner.load_capture(tmp_path, "8ep", "s") == receipt
    torch.save({"x": torch.zeros(2)}, payload)
    with pytest.raises(ValueError, match="Changed input/cache"):
        runner.load_capture(tmp_path, "8ep", "s")


def test_cpu_cache_end_to_end_without_constructing_model(tmp_path, monkeypatch):
    args = args_for(tmp_path, ["--analyze-only", "--batches", "1"])
    manifest = {"fingerprint": "test", "inputs": {}}
    annotation = tmp_path/"train.json"
    annotation.write_text("{}")
    record, settings = record_fixture()
    for side, iteration in runner.STAGES.items():
        row = {**metadata(), "batch": 0}
        path = tmp_path/side/"batch_0.pt"
        runner.save_gradients(path, dict(fingerprint="test", side=side, metadata=row, records=[record]))
        receipt = dict(fingerprint="test", side=side, iteration=iteration, weights_unchanged=True,
                       assets=[], train_annotations=runner.file_identity(annotation), matcher=settings,
                       classifier=["synthetic"], native_training_protocol={},
                       batches=[{**row, "cache": runner.file_identity(path)}])
        runner.save_json(tmp_path/(side+"_capture.json"), receipt)
    monkeypatch.setattr(runner, "prepare", lambda a: (tmp_path, manifest))
    monkeypatch.setattr(runner, "capture_stage", lambda *a: pytest.fail("No model allowed"))
    result = runner.run(args)
    assert result["complete"] and result["training_updates"] == 0
    assert result["analysis_device"] == "cpu"
    assert set(result["stages"]) == {"8ep", "12ep"}
    saved = json.loads((tmp_path/"report.json").read_text())
    assert saved["stages"]["8ep"][0]["branches"][0]["native_replay"]["matching_identical"]


def test_prepare_locks_sources_and_never_overwrites_unrelated_output(tmp_path, monkeypatch):
    config = tmp_path/"config.py"
    config.write_text("# config")
    for side in ("8", "12"):
        torch.save({"model": {}}, tmp_path/(side+".pth"))
    args = args_for(tmp_path, ["--config-file", str(config)])
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    (tmp_path/"tools").mkdir()
    # Minimal source identities for this test, not a model-forward simulation.
    for name in ("audit_eq2_training_stages.py", "eq2_training_audit_ops.py", "audit_tpa_gradients.py",
                 "capture_tpa_gradients.py", "decoder_loss_audit_ops.py", "evaluate_decoder_rollback.py",
                 "tpa_geometry_audit_ops.py", "compare_rare_pr_reports.py"):
        (tmp_path/"tools"/name).write_text("# test")
    seen = []
    monkeypatch.setattr(runner, "endpoint_state", lambda ckpt, it: seen.append(it))
    output, manifest = runner.prepare(args)
    assert seen == [56799, 85199]
    assert runner.prepare(args)[1] == manifest
    args.batches = 2
    with pytest.raises(ValueError, match="changed"):
        runner.prepare(args)
    args.output_dir = str(tmp_path/"unrelated")
    other = Path(args.output_dir)
    other.mkdir()
    (other/"keep.txt").write_text("user data")
    with pytest.raises(ValueError, match="Nonempty"):
        runner.prepare(args)
    assert (other/"keep.txt").read_text() == "user data"
