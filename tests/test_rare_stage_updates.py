from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from lami_dino.checkpoint_init import load_trusted_torch_file
from tools import audit_rare_stage_updates as runner
from tools.rare_stage_update_ops import (
    endpoint_effects, flat_delta, grouped_margins, local_margin_audit,
    probe_pairing, select_ranking_classes, selected_logits,
)
from tools.tpa_geometry_audit_ops import WEIGHTS, extract_shared_tpa, reconstruct_tpa
from tools.tpa_gradient_audit_ops import gradient_directions, text_observables
from test_tpa_geometry_audit import checkpoint_for, make_tpa, fixture as geometry_fixture


def comparison_fixture():
    rows = []
    for name, delta, delta50, old_tp, new_tp in (
        ("chocolate_mousse", -65, 0, 1, 1), ("lasagna", -59, -50, 1, 1),
        ("recall_drop", -90, -40, 2, 1), ("keg", -34, -25, 2, 2),
        ("bass_horn", -26, -22, 2, 2), ("gain", 10, 10, 1, 1),
    ):
        rows.append({"category_id": len(rows), "name": name, "delta_AP": delta,
                     "delta_AP50": delta50, "curves": {"0.50": {
                         "old": {"num_gt": 3, "true_positives": old_tp},
                         "new": {"num_gt": 3, "true_positives": new_tp}}}})
    return {"complete": True, "old_apr": 42.8843, "new_apr": 42.3031, "per_class": rows}


def test_selects_ranking_not_recall_or_only_higher_iou_regressions():
    report = comparison_fixture()
    assert select_ranking_classes(report) == ["lasagna", "keg", "bass_horn"]
    assert select_ranking_classes(report, 1) == ["lasagna"]
    report["complete"] = False
    with pytest.raises(ValueError, match="complete"):
        select_ranking_classes(report)


def panel_fixture():
    torch.manual_seed(4)
    state, _ = extract_shared_tpa(checkpoint_for(make_tpa(0.)))
    prompts = torch.randn(2, 8, 4)
    new = {k: v.clone() for k, v in state.items()}
    new["value_proj.weight"] += torch.randn_like(new["value_proj.weight"]) * .1
    states = {"old": state, "new": new}
    banks = {s: {"category_ids": [11, 22], "temperature": .07, "tpa_tau": .07,
                  "logit_scale": 5., "cls_bias": -3., "prompt_sha256": "fixture",
                  "prototypes": reconstruct_tpa(prompts, v, .07)["after"]}
             for s, v in states.items()}
    features = {"old": torch.randn(4, 4), "new": torch.randn(4, 4)}
    indices = torch.tensor([0, 0, 1, 1])
    records = [{"category": name, "kind": kind}
               for name in ("lasagna", "keg") for kind in ("tp", "fp")]
    clips = {s: torch.tensor([-1., -2., -3., -4.], dtype=torch.float64) for s in states}
    return states, prompts, banks, features, indices, records, clips


def test_endpoint_decomposition_closes_and_classifier_swaps_preserve_query_bias():
    _, _, banks, features, indices, records, clips = panel_fixture()
    banks["new"]["cls_bias"] = -2.
    clips["new"] += torch.tensor([.1, -.1, .2, -.2])
    result = endpoint_effects(features, indices, banks, records, clips, .3)
    assert result["closure_max_abs_error"] < 1e-12
    for category, row in result["margin_change"]["total"].items():
        assert row["mean_tp_minus_fp"] == pytest.approx(sum(
            result["margin_change"][key][category]["mean_tp_minus_fp"]
            for key in ("terminal_tpa_bank", "query_and_bias_path", "clip_roi_path")))
        assert row["mean_tp_minus_fp"] == pytest.approx(
            result["native_margin"]["new"][category]["mean_tp_minus_fp"]
            - result["native_margin"]["old"][category]["mean_tp_minus_fp"])
    banks["new"]["prototypes"] = banks["old"]["prototypes"]
    result = endpoint_effects(features, indices, banks, records, clips, .3)
    assert result["per_region_effects"]["terminal_tpa_bank"] == [0.] * 4


def test_unchanged_endpoint_zero_and_pure_bank_change_has_no_query_component():
    _, _, banks, features, indices, records, clips = panel_fixture()
    features["new"] = features["old"]
    result = endpoint_effects(features, indices, banks, records, clips, .3)
    assert result["per_region_effects"]["query_and_bias_path"] == [0.] * 4
    banks["new"] = banks["old"]
    result = endpoint_effects(features, indices, banks, records, clips, .3)
    assert result["per_region_effects"]["total"] == [0.] * 4
    with pytest.raises(ValueError, match="finite"):
        grouped_margins(torch.tensor([float("nan")]), records[:1])
    assert grouped_margins(torch.ones(1), records[:1])["lasagna"]["mean_tp_minus_fp"] is None


def test_endpoint_rejects_changed_protocol_and_nonfinite_features():
    _, _, banks, features, indices, records, clips = panel_fixture()
    banks["new"]["temperature"] = .03
    with pytest.raises(ValueError, match="protocol differs"):
        endpoint_effects(features, indices, banks, records, clips, .3)
    banks["new"]["temperature"] = .07
    features["new"][0, 0] = float("inf")
    with pytest.raises(ValueError, match="Nonfinite"):
        endpoint_effects(features, indices, banks, records, clips, .3)


def capture_fixture(state, seed=100):
    torch.manual_seed(seed)
    size = sum(state[k].numel() for k in WEIGHTS)
    apr = torch.randn(size, dtype=torch.float64)
    gradients = {"apr": apr, "detector": -2 * apr, "rpsa": .1 * torch.randn(size, dtype=torch.float64)}
    return {"windows": [{"window": 0, "gradients": gradients, "mean_losses": {"loss_apr": .1},
                         "microbatches": []}], "clip_max_norm": .5,
            "weights_unchanged": True, "optimizer_created": False}


def test_local_fused_margin_derivative_matches_raw_clipped_finite_difference():
    states, prompts, banks, features, indices, records, _ = panel_fixture()
    state, bank, x = states["new"], banks["new"], features["new"]
    logits = selected_logits(x, bank["prototypes"], indices, bank)
    records = [{**r, "native_cache_logit": float(z)} for r, z in zip(records, logits)]
    captured = capture_fixture(state)
    before = {k: v.clone() for k, v in state.items()}
    delta = flat_delta(states["old"], state)
    result = local_margin_audit(captured, state, prompts, bank, x, indices, records, .3, delta)[0]
    assert result["routing"]["conflict_projected"] == 1
    assert result["additive_closure_max_abs_error"] < 1e-7
    directions, _ = gradient_directions(captured["windows"][0]["gradients"], .5)
    # Finite difference of ACTUAL raw common-clipped direction, not a unit JVP.
    for direction in ("routed_total", "projection_added", "apr", "detector"):
        descent = -directions[direction].double() * result["common_clip_coefficient"]
        weights, offset = [], 0
        for key in WEIGHTS:
            n = state[key].numel()
            weights.append(descent[offset:offset + n].view_as(state[key]))
            offset += n
        def evaluate(sign):
            shifted = tuple(state[k].double() + sign * 1e-5 * d for k, d in zip(WEIGHTS, weights))
            buffers = {k: state[k].double() for k in ("slot_prior_strength", "prototype_mode_strength")}
            z = text_observables(shifted, buffers, prompts.double(), .07, x.double(), indices,
                                 .07, 5., -3.)[2][:, -1]
            return .7 * F.logsigmoid(z)
        numerical = grouped_margins((evaluate(1) - evaluate(-1)) / 2e-5, records)
        for name, row in numerical.items():
            assert result["directions"][direction]["margin_derivative"][name]["mean_tp_minus_fp"] == pytest.approx(
                row["mean_tp_minus_fp"], abs=1e-7, rel=1e-5)
    assert all(torch.equal(state[k], v) for k, v in before.items())


def test_pairing_checks_mapped_pixels_gt_and_fedloss_not_just_seed():
    micro = {"image_ids": [7], "global_gt_classes_before_remap": [[1]],
             "fedloss_category_indices": [1, 2], "mapped_inputs": [
                 {"image_sha256": "pixels", "gt_classes_sha256": "classes", "gt_boxes_sha256": "boxes"}]}
    old = {"windows": [{"microbatches": [micro]}]}
    new = deepcopy(old)
    assert probe_pairing(old, new)["all_verified_equal"]
    new["windows"][0]["microbatches"][0]["mapped_inputs"][0]["image_sha256"] = "different"
    assert not probe_pairing(old, new)["all_verified_equal"]
    del new["windows"][0]["microbatches"][0]["mapped_inputs"]
    assert probe_pairing(old, new)["unverifiable"] == 1
    new["windows"][0]["microbatches"] = []
    with pytest.raises(ValueError, match="counts differ"):
        probe_pairing(old, new)


def args_fixture(tmp_path):
    args = runner.parse_args(["--comparison", str(tmp_path / "comparison.json"),
        "--old-checkpoint", str(tmp_path / "old.pth"), "--new-checkpoint", str(tmp_path / "new.pth"),
        "--old-predictions", str(tmp_path / "old.json"), "--new-predictions", str(tmp_path / "new.json"),
        "--output-dir", str(tmp_path / "audit"), "--annotations", str(tmp_path / "ann.json"),
        "--prompt-bank", str(tmp_path / "prompts.npy"), "--config-file", str(tmp_path / "config.py")])
    for key in ("old_predictions", "new_predictions", "annotations", "prompt_bank", "config_file"):
        Path(getattr(args, key)).write_text("fixture")
    Path(args.comparison).write_text(json.dumps(comparison_fixture()))
    # Only the TPA state is needed during CPU preflight. Never load this into DINO.
    state = {"prototype_queries": torch.randn(5, 256), "key_proj.weight": torch.randn(256, 4),
             "key_proj.bias": torch.randn(256), "value_proj.weight": torch.randn(4, 4),
             "value_proj.bias": torch.randn(4), "slot_prior_strength": torch.tensor(.2),
             "prototype_mode_strength": torch.tensor(0.)}
    for side, iteration in (("old", 56799), ("new", 70999)):
        torch.save({"iteration": iteration, "model": {"classifier.tpa." + k: v for k, v in state.items()}},
                   getattr(args, side + "_checkpoint"))
    return args


def test_preflight_checks_checkpoint_iteration_radius_apr_and_total_budget(tmp_path):
    args = args_fixture(tmp_path)
    names, identities, states = runner.preflight(args)
    assert names == ["lasagna", "keg", "bass_horn"] and len(identities) == 8
    assert states["new"]["prototype_queries"].shape == (5, 256)
    args.windows = 3
    with pytest.raises(ValueError, match="128"):
        runner.preflight(args)
    args.windows = 2
    checkpoint = load_trusted_torch_file(args.old_checkpoint)
    checkpoint["iteration"] = 85199
    torch.save(checkpoint, args.old_checkpoint)
    with pytest.raises(ValueError, match="iteration 56799"):
        runner.preflight(args)
    checkpoint["iteration"] = 56799
    checkpoint["model"]["classifier.tpa.prototype_mode_strength"] = torch.tensor(1.5)
    torch.save(checkpoint, args.old_checkpoint)
    with pytest.raises(ValueError, match="no-radius"):
        runner.preflight(args)


def test_prepare_only_cannot_start_gpu_and_changed_apr_or_missing_files_fail(tmp_path, monkeypatch):
    args = args_fixture(tmp_path)
    args.prepare_only = True
    monkeypatch.setattr(runner, "prepare", lambda *a: ({}, [1]))
    def forbidden(*a, **kw):
        raise AssertionError("No GPU allowed")
    monkeypatch.setattr(runner, "dump_and_pair", forbidden)
    monkeypatch.setattr(runner, "capture_endpoints", forbidden)
    runner.run(args)
    Path(args.comparison).write_text(json.dumps({**comparison_fixture(), "old_apr": 45.2037}))
    with pytest.raises(ValueError, match="old_apr"):
        runner.run(args)
    Path(args.old_predictions).unlink()
    with pytest.raises(FileNotFoundError, match="Missing inputs"):
        runner.run(args)


def test_prepare_reuses_exact_fp_details_and_enforces_image_budget_before_gpu(tmp_path, monkeypatch):
    args = args_fixture(tmp_path)
    args.max_images = 1
    names, identities, _ = runner.preflight(args)
    def fake_inspect(a):
        details = {"sources": {k: identities[k] for k in
                               ("comparison", "old_predictions", "new_predictions", "annotations")}}
        Path(a.output).write_text(json.dumps(details))
    monkeypatch.setattr("tools.inspect_rare_pre_tp_fps.run", fake_inspect)
    monkeypatch.setattr("tools.rare_region_pairing_ops.collect_regions", lambda *a: [{"image_id": 1}])
    out = Path(args.output_dir)
    assert runner.prepare(args, names, identities, out)[1] == [1]
    def forbidden(*a):
        raise AssertionError("No replay when identities agree")
    monkeypatch.setattr("tools.inspect_rare_pre_tp_fps.run", forbidden)
    assert runner.prepare(args, names, identities, out)[1] == [1]
    monkeypatch.setattr("tools.rare_region_pairing_ops.collect_regions",
                        lambda *a: [{"image_id": 1}, {"image_id": 2}])
    with pytest.raises(ValueError, match="No GPU work started"):
        runner.prepare(args, names, identities, out)
    args.seed += 1
    with pytest.raises(ValueError):
        runner.prepare(args, names, identities, out)


def test_paired_panel_uses_geometric_counterpart_not_same_query_index(tmp_path):
    _, regions, cache, bank, _ = geometry_fixture(tmp_path)
    state, _ = extract_shared_tpa(load_trusted_torch_file(regions["checkpoints"]["new"]["path"]))
    prompts = torch.from_numpy(np.load(tmp_path / "prompts.npy"))
    (cache / "old").mkdir()
    torch.save({**bank, "label": "old"}, cache / "old" / "bank.pt")
    for image in (1, 2):
        sample = load_trusted_torch_file(cache / "new" / f"{image}.pt")
        # Swap query rows in old cache. The spatial counterpart is now index 1.
        sample = {**sample, "label": "old"}
        for key in ("features", "roi_features", "query_boxes"):
            sample[key] = sample[key].flip(0)
        torch.save(sample, cache / "old" / f"{image}.pt")
        regions["regions"][image - 1]["other_best_score_box"]["query_id"] = 1
    _, features, _, records, _, excluded = runner.load_panel(regions, cache, {"old": state, "new": state}, prompts)
    assert not excluded
    assert [r["query_id"] for r in records["old"]] == [1, 1]
    assert [r["query_id"] for r in records["new"]] == [0, 0]
    torch.testing.assert_close(features["old"], features["new"])


@pytest.mark.parametrize("resume_cpu", [False, True])
def test_mocked_gpu_pipeline_produces_real_cpu_derivatives_and_no_optimizer(tmp_path, monkeypatch, resume_cpu):
    args = args_fixture(tmp_path)
    states, prompts, banks, features, indices, records, clips = panel_fixture()
    np.save(args.prompt_bank, prompts.numpy())
    identities = runner.preflight(args)[1]
    paired = {side: [{**r, "native_cache_logit": float(z)} for r, z in zip(records,
                     selected_logits(features[side], banks[side]["prototypes"], indices, banks[side]))]
              for side in states}
    captures = {side: capture_fixture(states[side]) for side in states}
    monkeypatch.setattr(runner, "preflight", lambda a: (["lasagna", "keg"], identities, states))
    monkeypatch.setattr(runner, "prepare", lambda *a: ({}, [1, 2]))
    monkeypatch.setattr(runner, "dump_and_pair", lambda *a: {})
    monkeypatch.setattr(runner, "load_panel", lambda *a: (banks, features, indices, paired, clips, []))
    monkeypatch.setattr(runner, "capture_endpoints", lambda *a: captures)
    def forbidden(*a, **kw):
        raise AssertionError("Audit must not construct an optimizer")
    monkeypatch.setattr(torch.optim.Optimizer, "__init__", forbidden)
    if resume_cpu:
        args.resume_cpu = True
        monkeypatch.setattr(runner, "load_cpu_resume", lambda *a:
                            (["lasagna", "keg"], identities, states, {}, captures, [1, 2]))
        for name in ("preflight", "prepare", "dump_and_pair", "capture_endpoints"):
            monkeypatch.setattr(runner, name, forbidden)
    before = {s: {k: v.clone() for k, v in t.items()} for s, t in states.items()}
    report = runner.run(args)
    assert report["complete"] and not report["optimizer_created"]
    assert set(report["local_gradient_audit"]) == {"old", "new"}
    assert report["gradient_probe_image_exposures"] == 128
    assert json.loads((Path(args.output_dir) / "report.json").read_text()) == report
    assert all(torch.equal(states[s][k], v) for s, t in before.items() for k, v in t.items())


def test_gradient_caches_cover_both_endpoints_and_refuse_changed_probe_inputs(tmp_path, monkeypatch):
    args = args_fixture(tmp_path)
    _, _, states = runner.preflight(args)
    output = Path(args.output_dir)
    output.mkdir()
    calls = []
    annotation = tmp_path / "train.json"
    annotation.write_text("train_norare fixture")
    def geometry(a):
        g = {"side": a.side}
        Path(a.output).write_text(json.dumps(g))
        return g
    def identity(a, g):
        return {"side": g["side"], "seed": a.seed}
    def capture(a, g, state):
        calls.append(g["side"])
        return {**capture_fixture(state), "train_annotations": runner.file_identity(annotation)}
    monkeypatch.setattr("tools.audit_tpa_geometry.run", geometry)
    monkeypatch.setattr("tools.audit_tpa_gradients.capture_identity", identity)
    monkeypatch.setattr("tools.capture_tpa_gradients.capture", capture)
    result = runner.capture_endpoints(args, output, states)
    assert calls == ["old", "new"] and set(result) == {"old", "new"}
    runner.capture_endpoints(args, output, states)
    assert calls == ["old", "new"]  # no extra native training forward
    annotation.write_text("changed annotations")
    with pytest.raises(ValueError, match="annotations changed"):
        runner.capture_endpoints(args, output, states)
    args.seed += 1
    with pytest.raises(ValueError, match="cache identity changed"):
        runner.capture_endpoints(args, output, states)


def cpu_resume_fixture(tmp_path):
    from tools.audit_tpa_gradients import capture_identity, save_gradients
    from tools.diagnose_detector_tpa_pairing import locked_manifest
    from tools.diagnose_rare_fp_regions import fingerprint
    args = args_fixture(tmp_path)
    names, identities, states = runner.preflight(args)
    output = Path(args.output_dir)
    locked_manifest(output / "manifest.json", {"schema_version": 1, "sources": identities,
        "categories": names, "max_images": 32, "seed": 42, "windows": 2, "microbatches": 8, "batch_size": 4})
    pair_fp = locked_manifest(output / "pairing_cache/manifest.json", {
        "old_sha256": identities["old_checkpoint"]["sha256"],
        "new_sha256": identities["new_checkpoint"]["sha256"]})
    (output / "fp_details.json").write_text("{}")
    regions = {"complete": True, "parent_fingerprint": pair_fp, "selected_image_ids": [1],
               "annotations": identities["annotations"],
               "source_details": runner.file_identity(output / "fp_details.json"),
               "checkpoints": {s: identities[s + "_checkpoint"] for s in ("old", "new")}}
    (output / "regions.json").write_text(json.dumps(regions))
    train = tmp_path / "train.json"
    train.write_text("train annotations")
    for side, iteration in (("old", 56799), ("new", 70999)):
        (output / "pairing_cache" / side).mkdir()
        torch.save({}, output / "pairing_cache" / side / "bank.pt")
        geometry_path = output / f"{side}_geometry.json"
        geometry = {"checkpoint": identities[side + "_checkpoint"], "side": side,
                    "source_report": runner.file_identity(output / "regions.json"),
                    "prompt_bank": identities["prompt_bank"]}
        geometry_path.write_text(json.dumps(geometry))
        probe_args = SimpleNamespace(config_file=args.config_file, geometry_json=str(geometry_path),
            device="cuda:0", windows=2, microbatches=8, batch_size=4, seed=42)
        identity = capture_identity(probe_args, geometry)
        saved = {"inputs": identity, "fingerprint": fingerprint(identity),
                 "capture": {**capture_fixture(states[side]), "iteration": iteration,
                             "train_annotations": runner.file_identity(train)}}
        save_gradients(output / f"{side}_gradients.pt", saved)
    return runner.parse_args(["--resume-cpu", "--output-dir", str(output)]), output


def test_cpu_resume_loads_locked_metadata_without_native_or_geometry_regeneration(tmp_path, monkeypatch):
    args, output = cpu_resume_fixture(tmp_path)
    def forbidden(*a, **kw):
        raise AssertionError("CPU resume must not capture native images/gradients or rewrite geometry")
    monkeypatch.setattr("tools.audit_tpa_geometry.run", forbidden)
    monkeypatch.setattr("tools.capture_tpa_gradients.capture", forbidden)
    monkeypatch.setattr("tools.diagnose_detector_tpa_pairing.dump_checkpoint", forbidden)
    before = {p: p.read_bytes() for p in output.rglob("*") if p.is_file()}
    names, identities, states, regions, captures, images = runner.load_cpu_resume(args, output)
    assert names == ["lasagna", "keg", "bass_horn"] and images == [1]
    assert captures["old"]["iteration"] == 56799 and captures["new"]["iteration"] == 70999
    assert args.prompt_bank == identities["prompt_bank"]["path"]
    assert all(p.read_bytes() == content for p, content in before.items())


@pytest.mark.parametrize("changed", ["missing", "checkpoint", "gradient", "geometry", "annotations"])
def test_cpu_resume_fails_closed_instead_of_starting_gpu(tmp_path, changed):
    args, output = cpu_resume_fixture(tmp_path)
    if changed == "missing":
        (output / "old_gradients.pt").unlink()
        error = FileNotFoundError
    else:
        error = ValueError
        if changed == "checkpoint":
            (tmp_path / "old.pth").write_bytes(b"different")
        elif changed == "gradient":
            path = output / "new_gradients.pt"
            saved = load_trusted_torch_file(path)
            saved["inputs"]["seed"] += 1
            torch.save(saved, path)
        elif changed == "geometry":
            path = output / "old_geometry.json"
            geometry = json.loads(path.read_text())
            geometry["source_report"]["sha256"] = "different"
            path.write_text(json.dumps(geometry))
        else:
            (tmp_path / "train.json").write_text("changed train annotations")
    with pytest.raises(error):
        runner.load_cpu_resume(args, output)
