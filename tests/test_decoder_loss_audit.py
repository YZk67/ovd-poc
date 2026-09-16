from contextlib import nullcontext
from copy import deepcopy
import json
from pathlib import Path
import random
import subprocess
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import torch

from tools import audit_decoder_loss_sources as runner
from tools.compare_rare_pr_reports import file_identity, save_json
from tools.decoder_loss_audit_ops import (
    GROUPS, average_gradients, component_gradients, directional_effects, gradient_summary,
    isolated_rng, official_control_regions, select_controls, split_losses, summarize_probes,
)


def losses(p, q):
    return {"loss_class": p.sum(), "loss_class_0": 2*p.sum(), "loss_class_dn_4": 3*p.sum(),
            "loss_class_enc": q.sum(), "loss_bbox": 4*p.sum(), "loss_bbox_dn": -p.sum(),
            "loss_giou_0": -2*p.sum(), "loss_rpsa": q.square().sum(), "loss_apr": q.sum(),
            "diagnostic": torch.tensor(float("nan"))}


def test_component_gradient_paths_aux_dn_enc_and_no_updates():
    p, q, unused = (torch.nn.Parameter(torch.tensor([1., 2.])) for _ in range(3))
    before = p.detach().clone()
    g, connections, keys, _ = component_gradients(losses(p, q), (p, unused))
    torch.testing.assert_close(g["classification"], torch.tensor([6., 6., 0., 0.]))
    torch.testing.assert_close(g["bbox_l1"], torch.tensor([3., 3., 0., 0.]))
    torch.testing.assert_close(g["bbox_giou"], torch.tensor([-2., -2., 0., 0.]))
    assert g["rpsa"] is None and g["apr"] is None
    assert connections["classification"] == [0] and connections["apr"] == []
    assert "loss_class_enc" in keys["classification"]
    assert all(v.grad is None for v in (p, q, unused)) and torch.equal(before, p)


def test_connected_zero_is_not_disconnected():
    p, q = torch.nn.Parameter(torch.ones(2)), torch.tensor([1.])
    value = losses(p, q)
    value["loss_rpsa"] = p.sum() * 0
    g, paths, _, _ = component_gradients(value, (p,))
    assert paths["rpsa"] == [0] and paths["apr"] == []
    assert g["rpsa"] is not None and g["rpsa"].norm() == 0


@pytest.mark.parametrize("kind", ["unknown", "missing", "nonfinite", "nonscalar"])
def test_loss_grouping_fails_closed(kind):
    value = losses(torch.ones(2), torch.ones(2))
    if kind == "unknown":
        value["loss_new"] = torch.tensor(1.)
    elif kind == "missing":
        value.pop("loss_apr")
    else:
        value["loss_class"] = torch.tensor(float("nan")) if kind == "nonfinite" else torch.ones(3)
    with pytest.raises(ValueError):
        split_losses(value)


def test_microbatch_average_counts_disconnected_micro_and_direction_sign():
    a = {k: None for k in GROUPS}
    b = {**a, "classification": torch.tensor([4., 0.])}
    g = average_gradients([a, b])
    torch.testing.assert_close(g["classification"], torch.tensor([2., 0.]))
    effects = directional_effects(torch.tensor([3., 8.]), g)
    assert effects["classification"]["raw_descent_derivative"] == -6
    assert effects["classification"]["unit_descent_derivative"] == -3
    assert effects["apr"]["unit_descent_derivative"] is None
    summary = gradient_summary(g, torch.tensor([-1., 0.]))
    assert summary["classification"]["descent_cosine_with_8_to_10ep_delta"] == 1


def test_reverse_gradient_dot_matches_central_difference():
    p = torch.tensor([.3, -.2], dtype=torch.float64, requires_grad=True)
    observable = lambda x: torch.nn.functional.logsigmoid(x).dot(torch.tensor([.7, -.7], dtype=x.dtype))
    h, = torch.autograd.grad(observable(p), (p,))
    g = torch.tensor([.8, .1], dtype=p.dtype)
    measured = directional_effects(h, {"classification": g})["classification"]["raw_descent_derivative"]
    eps = 1e-5
    finite = float((observable(p.detach()-eps*g) - observable(p.detach()+eps*g))/(2*eps))
    assert measured == pytest.approx(finite, abs=1e-9)


def test_rng_isolation_restores_torch_numpy_python_even_on_exception():
    random.seed(8)
    np.random.seed(8)
    torch.manual_seed(8)
    state = (random.getstate(), np.random.get_state(), torch.random.get_rng_state())
    with pytest.raises(RuntimeError):
        with isolated_rng(100):
            random.random(), np.random.rand(), torch.rand(2)
            raise RuntimeError("test")
    assert random.getstate() == state[0]
    assert np.array_equal(np.random.get_state()[1], state[1][1])
    assert torch.equal(torch.random.get_rng_state(), state[2])


def dataset():
    return {"images": [{"id": i, "height": 100, "width": 100, "neg_category_ids": [],
                        "not_exhaustive_category_ids": []} for i in range(1, 13)],
            "categories": [{"id": i, "name": f"cat{i}", "frequency": "r" if i <= 6 else "c"}
                           for i in range(1, 13)],
            "annotations": [{"id": i, "image_id": i, "category_id": i, "bbox": [0, 0, 10, 10],
                             "area": 100} for i in range(1, 13)]}


def test_controls_are_seeded_annotation_only_exclude_focus_nonexhaustive():
    ds = dataset()
    ds["images"][1]["not_exhaustive_category_ids"] = [2]
    a = select_controls(ds, {"cat1"}, {1}, per_split=3, seed=7)
    assert a == select_controls(ds, {"cat1"}, {1}, per_split=3, seed=7)
    assert len(a) == 6 and len({r["image_id"] for r in a}) == 6
    assert all(r["category_id"] not in (1, 2) for r in a)
    assert sum(r["panel"] == "control_rare" for r in a) == 3
    with pytest.raises(ValueError, match="Insufficient"):
        select_controls(ds, set(), set(range(1, 13)), per_split=1)


def test_official_controls_tp_fp_and_global_top300():
    pytest.importorskip("lvis")
    ds = dataset()
    control = {"category_id": 1, "category": "cat1", "frequency": "r", "panel": "control_rare", "image_id": 1}
    boxes = torch.tensor([[0., 0., 10., 10.], [50., 50., 60., 60.]])
    scores = torch.tensor([[.9, .1], [.8, .2]])
    rows = official_control_regions(ds, control, boxes, scores, [1, 7])
    assert [(r["query_id"], r["kind"]) for r in rows] == [(0, "tp"), (1, "fp")]
    # 300 higher scoring other-class pairs must evict the target; no rare-only cap.
    boxes = boxes[0].repeat(301, 1)
    scores = torch.zeros(301, 2)
    scores[:300, 1] = .9
    scores[300, 0] = .8
    assert official_control_regions(ds, control, boxes, scores, [1, 7]) == []


def probe(kind, count, expected, raw):
    return {"panel": "focus", "category": "a", "kind": kind, "count": count, "expected_count": expected,
            "effects": [{g: {"raw_descent_derivative": raw, "unit_descent_derivative": raw/2}
                          for g in GROUPS}] if count else []}


def test_summary_region_weighting_not_image_weighting_and_missing_is_na():
    rows = [probe("tp", 2, 2, 6.), probe("tp", 1, 1, 3.), probe("fp", 1, 1, 5.)]
    result = summarize_probes(rows, 1)
    assert result[0]["raw_descent_derivative"]["tp_minus_fp"] == -2
    assert result[0]["unit_descent_derivative"]["tp_minus_fp"] == -1
    rows.append(probe("fp", 0, 1, 0))
    assert summarize_probes(rows, 1)[0]["unit_descent_derivative"]["tp_minus_fp"] is None


@pytest.mark.parametrize("option,value", [("windows", 5), ("microbatches", 17), ("batch_size", 5),
                                         ("controls_per_split", 5), ("windows", 0), ("cpu_threads", 0)])
def test_budgets(option, value):
    args = runner.parse_args(["--query-audit", "q", "--output-dir", "o"])
    assert runner.check_budget(args) == 256
    setattr(args, option, value)
    with pytest.raises(ValueError):
        runner.check_budget(args)


class Toy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.core = torch.nn.Parameter(torch.ones(2))
        self.register_buffer("novel_idx", torch.tensor([False, True]))
        self.calls = 0

    def filter_content_info(self, data):
        return torch.tensor([0]), data

    def forward(self, data):
        self.calls += 1
        self.filter_content_info(data)
        return losses(self.core * data[0]["image_id"], torch.ones(2))


def test_training_loop_averages_caches_pairs_and_never_updates(tmp_path, monkeypatch):
    batches = [[{"image_id": i, "image": torch.zeros(3, 2, 2), "instances": SimpleNamespace(
        gt_classes=torch.tensor([0]), gt_boxes=SimpleNamespace(tensor=torch.zeros(1, 4)))}] for i in (1, 3)]
    config = ModuleType("detectron2.config")
    config.instantiate = lambda _: deepcopy(batches)
    data = ModuleType("detectron2.data")
    annotations = tmp_path / "train.json"
    annotations.write_text("train_only")
    data.MetadataCatalog = SimpleNamespace(get=lambda _: SimpleNamespace(json_file=annotations))
    events = ModuleType("detectron2.utils.events")
    events.EventStorage = lambda **_: nullcontext()
    for module in (config, data, events):
        monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(runner, "isolated_rng", lambda seed, device: isolated_rng(seed))
    monkeypatch.setattr(torch.optim.Optimizer, "__init__", lambda *a, **k: pytest.fail("No optimizer"))
    cfg = SimpleNamespace(dataloader=SimpleNamespace(train=SimpleNamespace(
        dataset=SimpleNamespace(names="lvis_v1_train_norare"), total_batch_size=1, num_workers=0)))
    args = SimpleNamespace(seed=1, device="cuda:0", windows=1, microbatches=2, batch_size=1)
    ctx = SimpleNamespace(output=tmp_path / "output", signature="locked")
    model = Toy()
    original = model.core.detach().clone()
    windows = runner.training_windows(model, (model.core,), cfg, args, ctx, "old", 56799, torch.ones(2))
    assert model.calls == 2 and torch.equal(model.core, original) and model.core.grad is None
    torch.testing.assert_close(windows[0]["gradients"]["classification"], torch.tensor([12., 12.]))
    assert windows[0]["gradients"]["apr"] is None
    runner.training_windows(model, (model.core,), cfg, args, ctx, "old", 56799, torch.ones(2))
    assert model.calls == 2  # Cached window incurs no model forward/backward.
    batches[0][0]["image"] += 1
    with pytest.raises(ValueError, match="augmentation changed"):
        runner.training_windows(model, (model.core,), cfg, args, ctx, "old", 56799, torch.ones(2))


def test_eval_hook_keeps_graph_and_restores_methods():
    class Classifier:
        def _compute_tpa_logits(self, x, **_):
            return x * 2
    classifier = Classifier()
    model = SimpleNamespace(class_embed=[classifier], transformer=SimpleNamespace(decoder=SimpleNamespace(num_layers=1)),
                            inference=lambda *a, **k: [])
    old = classifier._compute_tpa_logits, model.inference
    with pytest.raises(RuntimeError):
        with runner.capture_eval_graph(model) as capture:
            x = torch.ones(2, requires_grad=True)
            classifier._compute_tpa_logits(x, content_inds=None, additional_class=None)
            h, = torch.autograd.grad(capture["logits"].sum(), (x,))
            assert torch.equal(h, torch.full((2,), 2.))
            raise RuntimeError("stop")
    assert (classifier._compute_tpa_logits, model.inference) == old


def test_analyze_only_never_falls_back_to_gpu(tmp_path, monkeypatch):
    args = runner.parse_args(["--query-audit", "q", "--output-dir", str(tmp_path), "--analyze-only"])
    monkeypatch.setattr(runner, "prepare", lambda _: SimpleNamespace(output=tmp_path))
    monkeypatch.setattr(runner, "capture_side", lambda *a: pytest.fail("No GPU fallback"))
    with pytest.raises(ValueError, match="never falls back"):
        runner.run(args)


def test_cache_identity_rejected(tmp_path):
    path = tmp_path / "cache.json"
    save_json(path, {"fingerprint": "wrong"})
    with pytest.raises(ValueError, match="Stale"):
        runner.read_locked(path, "expected")


def test_cli_without_detectron2_or_cuda():
    result = subprocess.run([sys.executable, runner.__file__, "--help"], cwd="/tmp", capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "--analyze-only" in result.stdout


@pytest.mark.parametrize("parent_control", [False, True])
def test_real_validation_loop_first_order_graph_and_cached_resume(tmp_path, monkeypatch, parent_control):
    class Classifier:
        def _compute_tpa_logits(self, x, **_):
            return x

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.core = torch.nn.Parameter(torch.tensor([.1, .2]))
            self.class_embed = [Classifier()]
            self.transformer = SimpleNamespace(decoder=SimpleNamespace(num_layers=1))
            self.calls = 0

        def inference(self, *a, **k):
            return []

        def forward(self, data):
            self.calls += 1
            features = torch.stack([self.core, self.core*2])[None]
            logits = self.class_embed[0]._compute_tpa_logits(features, content_inds=None, additional_class=None)
            boxes = torch.tensor([[[.1, .1, .1, .1], [.6, .6, .1, .1]]])
            return self.inference(logits, boxes, [(100, 100)])

    annotation = tmp_path / "annotation.json"
    save_json(annotation, dataset())
    cfg = SimpleNamespace(dataloader=SimpleNamespace(test=SimpleNamespace(
        dataset=SimpleNamespace(names="lvis_v1_val"), mapper=SimpleNamespace(is_train=False,
        augmentation_with_crop=None, augmentation=[SimpleNamespace(_target_="ResizeShortestEdge", short_edge_length=[32])]))))
    config, data = ModuleType("detectron2.config"), ModuleType("detectron2.data")
    config.instantiate = lambda _: lambda r: {**r, "image": torch.zeros(3, 32, 32), "width": 100, "height": 100}
    data.get_detection_dataset_dicts = lambda **_: [{"image_id": 1}]
    data.MetadataCatalog = SimpleNamespace(get=lambda _: SimpleNamespace(json_file=str(annotation)))
    for m in (config, data):
        monkeypatch.setitem(sys.modules, m.__name__, m)
    model = Model()
    before = model.core.detach().clone()
    rows = [{"image_id": 1, "query_id": i, "panel": "focus", "category": "cat1", "kind": kind}
            for i, kind in enumerate(("tp", "fp"))]
    ctx = SimpleNamespace(output=tmp_path / "output", stage=tmp_path / "stage", signature="locked",
        sources={"annotations": file_identity(annotation)}, dataset=dataset(), controls=[],
        identity={"focus_images": [1]}, records={"new": rows}, report={"inputs": {"parent_fingerprint": "parent"}})
    path = ctx.stage / "pairing_cache/new/1.pt"
    path.parent.mkdir(parents=True)
    torch.save({"fingerprint": "parent", "features": torch.stack([before, before*2]),
                "query_boxes": torch.tensor([[5., 5., 15., 15.], [55., 55., 65., 65.]])}, path)
    side = "new"
    if parent_control:
        side = "old"
        ctx.control_reference_dir = tmp_path / "previous_audit"
        ctx.control_reference_signature = "previous"
        ctx.controls = [{"image_id": 1, "panel": "control_rare", "category": "cat1", "category_id": 1}]
        anchors = [{**row, "panel": "control_rare", "box_xyxy": box} for row, box in zip(
            rows, ([5., 5., 15., 15.], [55., 55., 65., 65.]))]
        save_json(ctx.control_reference_dir / "new/validation_1.json",
                  {"fingerprint": "previous", "anchor_regions": anchors})
    bank = {"category_ids": list(range(1, 13)), "novel_mask": torch.tensor([True]*6+[False]*6)}
    windows = [{"gradients": {g: torch.tensor([1., 0.]) if g == "classification" else None for g in GROUPS}}]
    probes, controls = runner.validation_probes(model, (model.core,), cfg, None, ctx, side, windows, bank)
    assert bool(controls) == parent_control and len(probes) == 2 and model.calls == 1
    if parent_control:
        assert controls[0]["matched_counts"] == {"tp": 1, "fp": 1}
    htp = -.7 * float(torch.sigmoid(torch.tensor(-.1)))
    hfp = -1.4 * float(torch.sigmoid(torch.tensor(-.2)))
    summaries = summarize_probes(probes, 1)
    classified = next(r for r in summaries if r["loss_group"] == "classification")
    assert classified["raw_descent_derivative"]["tp_minus_fp"] == pytest.approx(htp-hfp)
    assert torch.equal(before, model.core) and model.core.grad is None
    replay, _ = runner.validation_probes(model, (model.core,), cfg, None, ctx, side, windows, bank)
    assert replay == probes and model.calls == 1


def test_analysis_checks_pairing_and_marks_missing_controls(tmp_path):
    source = {"mapped_inputs": [{"image": "a"}], "fedloss_category_indices": [0], "loss_keys": {"class": ["loss_class"]}}
    window = {"train_annotations": {"sha256": "train"}, "microbatches": [source], "summary": {}}
    captures = {}
    for side in ("old", "new"):
        captures[side] = {"side": side, "windows": [deepcopy(window)], "probes": [probe("tp", 1, 1, 2), probe("fp", 1, 1, 3)],
                          "control_coverage": [], "weights_unchanged": True, "optimizer_created": False, "training_updates": 0}
        save_json(tmp_path / f"{side}_capture.json", captures[side])
    ctx = SimpleNamespace(output=tmp_path, signature="locked", identity={})
    result = runner.analyze(ctx, captures)
    assert result["paired_training_inputs_verified"]
    assert not any(result["fully_paired_control_classes"].values())
    assert result["training_updates"] == 0
    captures["new"]["windows"][0]["microbatches"][0]["fedloss_category_indices"] = [1]
    with pytest.raises(ValueError, match="paired mapped"):
        runner.analyze(ctx, captures)
