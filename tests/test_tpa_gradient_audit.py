import json
from types import SimpleNamespace

import pytest
import torch

from tools.audit_tpa_geometry import run as run_geometry
from tools.audit_tpa_gradients import run
from tools.capture_tpa_gradients import collect_windows, state_versions
from tools.tpa_geometry_audit_ops import WEIGHTS, extract_shared_tpa
from tools.tpa_gradient_audit_ops import (
    direction_jvp, gradient_directions, loss_gradients, text_observables,
)
from test_tpa_geometry_audit import checkpoint_for, fixture as geometry_fixture, make_tpa


def test_grouped_gradients_equal_native_total_without_touching_parameters():
    p = torch.nn.Parameter(torch.tensor([2., -3.]))
    losses = {"loss_class": p.square().sum() * .5, "loss_bbox": p.sum() * 2,
              "loss_rpsa": p[0] * .05, "loss_apr": p[1].square() * .1,
              "diagnostic_not_a_loss": p.sum() * 1000}
    expected = torch.autograd.grad(sum(v for k, v in losses.items() if k.startswith("loss")), p, retain_graph=True)[0]
    before = p.detach().clone()
    gradients, values, unused = loss_gradients(losses, (p,))
    torch.testing.assert_close(sum(gradients.values()), expected)
    assert "diagnostic_not_a_loss" not in values
    assert all(not names for names in unused.values())
    assert p.grad is None and torch.equal(p, before)


def test_constant_rpsa_is_an_explicit_zero_direction():
    p = torch.nn.Parameter(torch.ones(2))
    gradients, _, unused = loss_gradients({"loss_class": p.sum(), "loss_apr": p.square().sum(),
                                          "loss_rpsa": torch.tensor(0.)}, (p,))
    assert torch.equal(gradients["rpsa"], torch.zeros(2))
    assert unused["rpsa"] == [0]


def test_disconnected_apr_and_nonfinite_losses_fail():
    p = torch.nn.Parameter(torch.ones(2))
    with pytest.raises(ValueError, match="disconnected"):
        loss_gradients({"loss_class": p.sum(), "loss_apr": torch.tensor(1.)}, (p,))
    with pytest.raises(ValueError, match="finite"):
        loss_gradients({"loss_class": p.sum() * float("nan"), "loss_apr": p.sum()}, (p,))


def test_projection_is_after_gradient_averaging_not_per_microbatch():
    a = torch.tensor([1., 0.])
    first = {"detector": torch.tensor([-2., 0.]), "rpsa": torch.zeros(2), "apr": a}
    second = {"detector": torch.tensor([2., 0.]), "rpsa": torch.zeros(2), "apr": a}
    averaged = {k: (first[k] + second[k]) / 2 for k in first}
    directions, stats = gradient_directions(averaged, .5)
    routed_first = gradient_directions(first, .5)[0]["routed_total"]
    routed_second = gradient_directions(second, .5)[0]["routed_total"]
    assert not torch.equal(directions["routed_total"], (routed_first + routed_second) / 2)
    torch.testing.assert_close(directions["routed_total"], a)
    assert stats["routing"]["conflict_projected"] == 0


def test_conflicting_projection_preserves_additive_components_and_apr_descent():
    gradients = {"detector": torch.tensor([-2., 3.]), "rpsa": torch.tensor([0., .5]), "apr": torch.tensor([1., 0.])}
    directions, stats = gradient_directions(gradients, .5)
    torch.testing.assert_close(directions["routed_total"], torch.tensor([1., 3.5]))
    torch.testing.assert_close(directions["routed_total"], directions["unprojected_total"] + directions["projection_added"])
    assert stats["routing"]["conflict_projected"] == 1
    assert stats["directions"]["routed_total"]["loss_derivative_per_unit_descent"]["apr"] < 0
    assert stats["directions"]["task"]["loss_derivative_per_unit_descent"]["apr"] > 0


def jvp_fixture():
    tpa = make_tpa()
    state, _ = extract_shared_tpa(checkpoint_for(tpa))
    prompts = torch.randn(3, 8, 4)
    bank = {"tpa_tau": .07, "temperature": .07, "logit_scale": 5., "cls_bias": -3.}
    features, indices = torch.randn(2, 4), torch.tensor([0, 2])
    gradient = torch.randn(sum(state[n].numel() for n in WEIGHTS))
    return state, prompts, bank, features, indices, gradient


def test_text_jvp_matches_central_finite_difference():
    state, prompts, bank, features, indices, gradient = jvp_fixture()
    before = {n: t.clone() for n, t in state.items()}
    result = direction_jvp(state, prompts, bank, features, indices, gradient)
    weights = tuple(state[n].double() for n in WEIGHTS)
    buffers = {n: state[n].double() for n in ("slot_prior_strength", "prototype_mode_strength")}
    flat = -gradient.double() / gradient.double().norm()
    offset, direction = 0, []
    for weight in weights:
        direction.append(flat[offset:offset + weight.numel()].view_as(weight))
        offset += weight.numel()
    def evaluate(sign):
        perturbed = tuple(w + sign * 1e-5 * d for w, d in zip(weights, direction))
        return text_observables(perturbed, buffers, prompts.double(), .07, features.double(), indices, .07, 5., -3.)
    plus, minus = evaluate(1), evaluate(-1)
    for column, key in ((1, "scalar_derivatives"), (2, "query_logit_derivatives")):
        torch.testing.assert_close(torch.tensor(result[key], dtype=torch.float64),
                                   (plus[column] - minus[column]) / 2e-5, rtol=1e-5, atol=1e-7)
    assert all(torch.equal(state[n], t) for n, t in before.items())


def test_parameter_layout_key_only_direction_cannot_move_value_center():
    state, prompts, bank, features, indices, gradient = jvp_fixture()
    gradient.zero_()
    gradient[:state["prototype_queries"].numel()] = 1.
    result = direction_jvp(state, prompts, bank, features, indices, gradient)
    assert all(row[0] == 0 for row in result["center_norm_derivatives"])
    assert all(row[0] == 0 for row in result["center_angular_speed_deg"])


def test_raw_jvps_are_additive_but_unit_jvps_need_norm_rescaling():
    state, prompts, bank, features, indices, first = jvp_fixture()
    second = torch.randn_like(first)
    results = [direction_jvp(state, prompts, bank, features, indices, g)
               for g in (first, second, first + second)]
    raw = [torch.tensor(r["query_logit_derivatives"], dtype=torch.float64) * g.double().norm()
           for r, g in zip(results, (first, second, first + second))]
    torch.testing.assert_close(raw[0] + raw[1], raw[2], rtol=1e-6, atol=1e-7)


def test_zero_direction_and_live_state_mutation_guard():
    state, prompts, bank, features, indices, gradient = jvp_fixture()
    result = direction_jvp(state, prompts, bank, features, indices, gradient * 0)
    assert not torch.tensor(result["query_logit_derivatives"]).any()
    model = make_tpa()
    previous = state_versions(model)
    with torch.no_grad():
        model.value_proj.weight.add_(1)
    assert state_versions(model) != previous


def pipeline_fixture(tmp_path):
    geo_args, _, _, _, _ = geometry_fixture(tmp_path)
    geometry = run_geometry(geo_args)
    config = tmp_path / "config.py"
    config.write_text("# synthetic config; heavy detector capture is mocked\n")
    args = SimpleNamespace(geometry_json=geo_args.output, config_file=str(config), checkpoint=None,
                           output=str(tmp_path / "gradient_report.json"), gradient_cache=None,
                           reuse_gradients=False, windows=1, microbatches=2, batch_size=2,
                           seed=42, device="cuda:0", cpu_threads=1)
    state, _ = extract_shared_tpa(torch.load(geometry["checkpoint"]["path"], weights_only=True))
    size = sum(state[n].numel() for n in WEIGHTS)
    gradients = {"detector": torch.randn(size), "rpsa": torch.zeros(size), "apr": torch.randn(size) * .1}
    captured = {"windows": [{"window": 0, "gradients": gradients, "mean_losses": {"loss_apr": .1}, "microbatches": []}],
                "clip_max_norm": .5, "weights_unchanged": True, "optimizer_created": False}
    return args, captured, geometry


def test_pipeline_saves_gradients_then_cpu_reuse_without_capture(tmp_path, monkeypatch):
    args, captured, _ = pipeline_fixture(tmp_path)
    before = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    monkeypatch.setattr("tools.audit_tpa_gradients.capture", lambda *a: captured)
    result = run(args)
    assert result["complete"] and len(result["windows"]) == 1
    def forbidden(*a):
        raise AssertionError("No repeated GPU/image work on reuse")
    monkeypatch.setattr("tools.audit_tpa_gradients.capture", forbidden)
    args.reuse_gradients = True
    assert run(args) == result
    assert json.loads(open(args.output).read()) == result
    for path, content in before.items():
        assert path.read_bytes() == content


def test_missing_query_cache_fails_before_gpu_capture(tmp_path, monkeypatch):
    args, _, geometry = pipeline_fixture(tmp_path)
    directory = geometry["regions"][0]["cache_directory"]
    from pathlib import Path
    (Path(directory) / "new" / "1.pt").unlink()
    def forbidden(*a):
        raise AssertionError("Preflight must precede GPU capture")
    monkeypatch.setattr("tools.audit_tpa_gradients.capture", forbidden)
    with pytest.raises(FileNotFoundError):
        run(args)


def test_old_side_gradient_audit_reads_old_query_cache(tmp_path, monkeypatch):
    args, captured, geometry = pipeline_fixture(tmp_path)
    from pathlib import Path
    directory = Path(geometry["regions"][0]["cache_directory"])
    (directory / "old").mkdir()
    for path in (directory / "new").glob("*.pt"):
        saved = torch.load(path, weights_only=True)
        saved["label"] = "old"
        torch.save(saved, directory / "old" / path.name)
        path.unlink()  # ensure an accidental hard-coded new/ path cannot work
    geometry["side"] = "old"
    Path(args.geometry_json).write_text(json.dumps(geometry))
    monkeypatch.setattr("tools.audit_tpa_gradients.capture", lambda *a: captured)
    assert run(args)["complete"]


def test_changed_capture_identity_refuses_reuse_and_existing_cache_requires_explicit_flag(tmp_path, monkeypatch):
    args, captured, _ = pipeline_fixture(tmp_path)
    monkeypatch.setattr("tools.audit_tpa_gradients.capture", lambda *a: captured)
    run(args)
    with pytest.raises(ValueError, match="already exists"):
        run(args)
    args.reuse_gradients = True
    args.seed += 1
    with pytest.raises(ValueError, match="provenance differs"):
        run(args)


def test_output_protection_and_hard_image_budget(tmp_path):
    args, _, _ = pipeline_fixture(tmp_path)
    args.output = args.geometry_json
    with pytest.raises(ValueError, match="must not overwrite"):
        run(args)
    args.windows = 100
    with pytest.raises(ValueError, match="budget"):
        run(args)


class TinyNativeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.tpa = make_tpa().train()
        self.tpa.dropout.p = 0.
        self.register_buffer("prompts", torch.randn(2, 8, 4))
        self.register_buffer("novel_idx", torch.tensor([False, True]))
        self.calls = 0

    def filter_content_info(self, data):
        return torch.tensor([0]), data

    def forward(self, data):
        self.calls += 1
        self.filter_content_info(data)
        prototypes, apr = self.tpa(self.prompts, with_loss=True, advance_step=False)
        return {"loss_class": prototypes.square().mean() * data[0]["image_id"],
                "loss_apr": apr, "loss_rpsa": prototypes.sum() * 0}


def test_real_capture_loop_averages_losses_without_updates_or_extra_forwards(monkeypatch):
    model = TinyNativeModel()
    before = {k: v.clone() for k, v in model.state_dict().items()}
    args = SimpleNamespace(windows=2, microbatches=2, batch_size=1)
    data = [[{"image_id": i, "image": torch.zeros(3, 2, 2),
              "instances": SimpleNamespace(gt_classes=torch.tensor([0]))}] for i in (1, 3, 1, 3)]
    original = model.filter_content_info
    def forbidden(*a, **kw):
        raise AssertionError("No optimizer may be constructed")
    monkeypatch.setattr(torch.optim.Optimizer, "__init__", forbidden)
    result = collect_windows(model, model.tpa, iter(data), args)
    assert model.calls == 4
    assert model.filter_content_info == original
    assert all(p.grad is None for p in model.parameters())
    assert all(torch.equal(model.state_dict()[k], v) for k, v in before.items())
    for key in ("detector", "apr", "rpsa"):
        torch.testing.assert_close(result[0]["gradients"][key], result[1]["gradients"][key])
    losses = result[0]["microbatches"]
    assert losses[0]["mapped_inputs"][0]["image_sha256"]
    assert losses[0]["mapped_inputs"][0]["gt_classes_sha256"]
    assert result[0]["mean_losses"]["loss_class"] == pytest.approx(
        (losses[0]["losses"]["loss_class"] + losses[1]["losses"]["loss_class"]) / 2)


def test_capture_loop_rejects_rare_targets_before_forward_and_restores_hook():
    model = TinyNativeModel()
    args = SimpleNamespace(windows=1, microbatches=1, batch_size=1)
    data = [[{"image_id": 1, "image": torch.zeros(3, 2, 2),
              "instances": SimpleNamespace(gt_classes=torch.tensor([1]))}]]
    original = model.filter_content_info
    with pytest.raises(ValueError, match="rare/invalid GT"):
        collect_windows(model, model.tpa, iter(data), args)
    assert model.calls == 0
    assert model.filter_content_info == original
