from copy import deepcopy
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch
from torch import nn

from lami_dino.checkpoint_init import load_trusted_torch_file
from tools import audit_query_path_updates as runner
from tools.compare_rare_pr_reports import file_identity, save_json
from tools.diagnose_rare_fp_regions import fingerprint
from tools.query_path_update_ops import (
    GROUPS, canonical_name, canonical_state, inventory, key_group, match_regions,
    region_effects, swap_group,
)
from tools.rare_stage_update_ops import selected_logits


@pytest.mark.parametrize("key,group", [
    ("backbone.norm1.weight", "visual_adapter"), ("neck.convs.0.weight", "visual_adapter"),
    ("backbone.stages.1.weight", "frozen_clip"), ("thead.weight", "frozen_clip"),
    ("transformer.encoder.layers.0.weight", "encoder_memory"),
    ("transformer.level_embeds", "encoder_memory"),
    ("transformer.enc_output_norm.weight", "encoder_memory"),
    ("class_embed.6.linear.weight", "proposal_head"), ("bbox_embed.6.weight", "proposal_head"),
    ("content_layer.weight", "query_content"),
    ("transformer.decoder.layers.0.weight", "decoder_core"),
    ("transformer.decoder.ref_point_head.weight", "decoder_core"),
    ("bbox_embed.5.weight", "decoder_box"),
    ("class_embed.5.linear.weight", "final_projection"),
    ("class_embed.5.cls_bias", "final_bias"),
    ("class_embed.0.linear.weight", "auxiliary_classifier"),
    ("class_embed.0.tpa.value_proj.weight", "upstream_tpa"),
    ("class_embed.0.tpa._step", "training_only"),
    ("class_embed.0.tpa.prototype_mode_strength", "fixed_protocol"),
    ("transformer.rpsa.weight", "training_only"), ("freq_weight", "training_only"),
])
def test_disjoint_group_partition(key, group):
    assert key_group(key) == group


def test_aliases_verified_and_counted_once():
    name = "module.transformer.decoder.class_embed.6.tpa.key_proj.weight"
    assert canonical_name(name) == "class_embed.0.tpa.key_proj.weight"
    old = {name: torch.ones(2), "class_embed.0.tpa.key_proj.weight": torch.ones(2)}
    new = {k: v + .5 for k, v in old.items()}
    result = inventory(old, new)["upstream_tpa"]
    assert result["numel"] == 2
    assert result["delta_l2"] == pytest.approx(2 ** -.5)
    new[name][0] = 4
    with pytest.raises(ValueError, match="aliases"):
        inventory(old, new)


@pytest.mark.parametrize("key", ["unexpected.weight", "backbone.stages.0.weight", "class_embed.0.tpa.prototype_mode_strength"])
def test_unknown_frozen_or_protocol_changes_fail_closed(key):
    with pytest.raises(ValueError, match="Unsupported changed"):
        inventory({key: torch.zeros(1)}, {key: torch.ones(1)})


def test_nonfinite_and_shape_changes_fail_closed():
    with pytest.raises(ValueError, match="Nonfinite"):
        canonical_state({"x": torch.tensor(float("nan"))})
    with pytest.raises(ValueError, match="shape/dtype"):
        inventory({"content_layer.weight": torch.zeros(1)}, {"content_layer.weight": torch.ones(2)})
    with pytest.raises(ValueError, match="keys differ"):
        inventory({"a": torch.zeros(1)}, {"b": torch.zeros(1)})


class Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.class_embed = nn.ModuleList([nn.Module() for _ in range(7)])
        shared = nn.Linear(2, 2)
        for m in self.class_embed:
            m.tpa = shared
            m.linear = nn.Linear(2, 2)
            m._cached_eval = torch.ones(1)
            m._external_prototypes = torch.ones(1)
        self.transformer = nn.Module()
        self.transformer.decoder = nn.Module()
        self.transformer.decoder.class_embed = self.class_embed
        self.content_layer = nn.Linear(2, 2)


def test_shared_swap_restores_every_alias_even_on_error_and_clears_cache():
    m, donor = Toy(), Toy().state_dict()
    before = deepcopy(m.state_dict())
    with pytest.raises(RuntimeError, match="abort"):
        with swap_group(m, donor, "upstream_tpa"):
            for k, v in m.state_dict().items():
                assert torch.equal(v, donor[k] if ".tpa." in k else before[k])
            assert all(ce._cached_eval is None for ce in m.class_embed)
            raise RuntimeError("abort")
    assert all(torch.equal(v, before[k]) for k, v in m.state_dict().items())
    assert all(ce._external_prototypes is None for ce in m.class_embed)


def test_joint_swap_does_not_touch_auxiliary_or_terminal_bias():
    m, donor = Toy(), Toy().state_dict()
    before = deepcopy(m.state_dict())
    with swap_group(m, donor, "all_query"):
        for k, v in m.state_dict().items():
            expected = donor if key_group(canonical_name(k)) in GROUPS else before
            assert torch.equal(v, expected[k])
    assert all(torch.equal(v, before[k]) for k, v in m.state_dict().items())


def rows():
    return [{"category": "rare", "kind": kind, "query_id": q, "image_id": 1, "box_xyxy": box}
            for q, kind, box in ((0, "tp", [0., 0., 10., 10.]), (1, "fp", [20., 20., 30., 30.]))]


def test_matching_permutation_not_query_index_and_no_score_dependence():
    boxes = torch.tensor([[20., 20., 30., 30.], [0., 0., 10., 10.]])
    matched = match_regions(rows(), boxes)
    assert [m["query_id"] for m in matched] == [1, 0]
    assert [m["iou"] for m in matched] == [1., 1.]
    duplicated = rows() + [rows()[0]]
    assert match_regions(duplicated, boxes)[0] == match_regions(duplicated, boxes)[2]
    assert match_regions(rows(), boxes[:1])[0] is None
    assert match_regions(rows(), boxes[:0]) == [None, None]


def test_distinct_queries_never_silently_reuse_hybrid_query():
    records = rows()
    records[1]["box_xyxy"] = records[0]["box_xyxy"]
    matches = match_regions(records, torch.tensor([[0., 0., 10., 10.]]))
    assert matches == [None, None]  # Indistinguishable anchors cannot identify a counterpart.


def test_equal_iou_candidate_ties_are_excluded_not_arbitrarily_scored():
    matches = match_regions(rows()[:1], torch.tensor([[0., 0., 10., 10.], [0., 0., 10., 10.]]))
    assert matches == [None]


def test_maximum_cardinality_precedes_iou_greed():
    records = rows()
    records[1]["box_xyxy"] = [0., 0., 8., 10.]
    boxes = torch.tensor([[0., 0., 10., 10.], [3., 0., 13., 10.]])
    matches = match_regions(records, boxes)
    # Greedy anchor0->candidate0 would strand anchor1; Hungarian keeps both.
    assert [m["query_id"] for m in matches] == [1, 0]


def samples():
    native = {"features": torch.tensor([[1., 0.], [0., 1.]]),
              "query_boxes": torch.tensor([r["box_xyxy"] for r in rows()])}
    bank = {"prototypes": torch.tensor([[[1., 0.], [0.8, .2]]]),
            "temperature": .07, "logit_scale": 5., "cls_bias": -2.}
    return native, bank


def test_fixed_readout_change_matches_direct_log_score_calculation():
    native, bank = samples()
    hybrid = {"features": torch.tensor([[0.2, 1.], [0.9, .1]]),
              "query_boxes": native["query_boxes"].flip(0)}
    result = region_effects(rows(), native, hybrid, bank, torch.tensor([0, 0]))
    z0 = selected_logits(native["features"], bank["prototypes"], torch.tensor([0, 0]), bank)
    z1 = selected_logits(hybrid["features"].flip(0), bank["prototypes"], torch.tensor([0, 0]), bank)
    delta = .7 * (torch.nn.functional.logsigmoid(z1) - torch.nn.functional.logsigmoid(z0))
    assert result["margin_change"]["rare"]["full_panel_margin_change"] == pytest.approx(float(delta[0] - delta[1]))
    assert [r["delta_fused_log_score"] for r in result["regions"]] == pytest.approx(delta.tolist())


def test_missing_fp_is_not_an_improvement():
    native, bank = samples()
    hybrid = {k: v[:1] for k, v in native.items()}
    result = region_effects(rows(), native, hybrid, bank, torch.tensor([0, 0]))
    assert result["matched"] == 1
    assert result["margin_change"]["rare"]["full_panel_margin_change"] is None
    summary = runner.summarize(result["regions"])["by_class"]["rare"]
    assert summary["full_panel_margin_change"] is None
    assert summary["matched_regions"] == 1


def test_unchanged_is_zero_and_nonfinite_fails():
    native, bank = samples()
    r = region_effects(rows(), native, native, bank, torch.tensor([0, 0]))
    assert r["margin_change"]["rare"]["full_panel_margin_change"] == 0.
    hybrid = deepcopy(native)
    hybrid["features"][0, 0] = float("nan")
    with pytest.raises(ValueError, match="Nonfinite"):
        region_effects(rows(), native, hybrid, bank, torch.tensor([0, 0]))


def test_summary_is_global_not_average_of_per_image_means():
    records = rows() + [{**rows()[0], "image_id": 2, "query_id": 2}]
    records = [{**r, "match": {"query_id": r["query_id"], "iou": .9}, "delta_fused_log_score": v}
               for r, v in zip(records, [1., 2., 5.])]
    summary = runner.summarize(records)["by_class"]["rare"]
    assert summary["full_panel_margin_change"] == 1.


def stage_fixture(tmp_path, monkeypatch):
    from tools import diagnose_rare_fp_regions as pairing
    from tools.diagnose_detector_tpa_pairing import locked_manifest

    stage, output = tmp_path / "stage", tmp_path / "query"
    stage.mkdir()
    old = {"class_embed.0.tpa.prototype_queries": torch.ones(5, 256),
           "class_embed.0.tpa.key_proj.weight": torch.ones(256, 2),
           "class_embed.0.tpa.key_proj.bias": torch.zeros(256),
           "class_embed.0.tpa.value_proj.weight": torch.eye(2),
           "class_embed.0.tpa.value_proj.bias": torch.zeros(2),
           "class_embed.0.tpa.slot_prior_strength": torch.tensor(.2),
           "class_embed.0.tpa.prototype_mode_strength": torch.tensor(0.),
           "content_layer.weight": torch.eye(2)}
    new = deepcopy(old)
    new["content_layer.weight"] += .1
    sources = {}
    for side, state, iteration in (("old", old, 56799), ("new", new, 70999)):
        path = tmp_path / (side + ".pth")
        torch.save({"model": state, "iteration": iteration}, path)
        sources[side + "_checkpoint"] = file_identity(path)
    for name in ("annotations", "config_file", "prompt_bank"):
        path = tmp_path / (name + ".json")
        save_json(path, {"categories": [{"id": 11, "name": "rare", "frequency": "r"}]})
        sources[name] = file_identity(path)
    parent_inputs = {"code_sha256": "model", "asset_sha256": {}, "alpha": 0., "beta": .3,
                     "novel_scale": 3., "tpa_tau": .004375, "cls_tau": .07, "max_dets": 300,
                     "old_sha256": sources["old_checkpoint"]["sha256"],
                     "new_sha256": sources["new_checkpoint"]["sha256"]}
    parent_fp = locked_manifest(stage / "pairing_cache/manifest.json", parent_inputs)
    locked_manifest(stage / "manifest.json", {"sources": sources})
    monkeypatch.setattr(pairing, "model_code_hash", lambda _: "model")
    native, bank = samples()
    bank.update(category_ids=[11], tpa_tau=.004375, prompt_sha256="prompt", vlm_temperature=100.,
                vlm_text=torch.ones(1, 2), novel_mask=torch.ones(1, dtype=torch.bool), fingerprint=parent_fp)
    paired = {}
    for side in ("old", "new"):
        current = deepcopy(native)
        state = old if side == "old" else new
        current["features"] = torch.nn.functional.linear(native["features"], state["content_layer.weight"])
        records = rows()
        logits = selected_logits(current["features"], bank["prototypes"], torch.tensor([0, 0]), bank)
        for row, logit in zip(records, logits):
            row["native_cache_logit"] = float(logit)
        paired[side] = records
        directory = stage / "pairing_cache" / side
        directory.mkdir()
        torch.save(dict(bank, label=side), directory / "bank.pt")
        torch.save(dict(current, image_id=1, fingerprint=parent_fp, label=side,
                        native_replay_check={"logit_max_abs_error": 0., "score_max_abs_error": 0.}), directory / "1.pt")
    report = {"complete": True, "sources": sources, "paired_regions": paired,
              "excluded_regions": []}
    save_json(stage / "report.json", report)
    args = runner.parse_args(["--stage-dir", str(stage), "--output-dir", str(output), "--device", "cpu"])
    return args


def fill_forward_cache(ctx):
    from tools.diagnose_detector_tpa_pairing import save_tensor_file
    for side in ("old", "new"):
        for variant in ctx.variants:
            native = ctx.samples[side, 1]
            save_tensor_file(runner.cache_path(ctx, side, variant, 1), {
                "fingerprint": ctx.fingerprint, "side": side, "variant": variant, "image_id": 1,
                "features": native["features"], "query_boxes": native["query_boxes"],
                "mapped_input_sha256": "pixels"})


def test_preflight_budget_sources_and_no_gpu_prepare(tmp_path, monkeypatch):
    args = stage_fixture(tmp_path, monkeypatch)
    ctx = runner.prepare(args)
    assert ctx.variants == ["native", "query_content", "all_query"]
    assert ctx.max_forwards == 6
    args.prepare_only = True
    monkeypatch.setattr(runner, "capture_forwards", lambda *_: pytest.fail("GPU forbidden"))
    runner.run(args)
    assert (Path(args.output_dir) / "preflight.json").is_file()
    args.max_forwards = 1
    with pytest.raises(ValueError, match="budget"):
        runner.prepare(args)


def test_cpu_resume_never_falls_back_and_completed_cache_reuses(tmp_path, monkeypatch):
    args = stage_fixture(tmp_path, monkeypatch)
    args.analyze_only = True
    monkeypatch.setattr(runner, "capture_forwards", lambda *_: pytest.fail("GPU forbidden"))
    with pytest.raises(FileNotFoundError, match="NEVER"):
        runner.run(args)
    ctx = runner.prepare(args)
    fill_forward_cache(ctx)
    report = runner.run(args)
    assert report["complete"]
    assert report["training_image_exposures"] == 0
    assert not report["optimizer_created"]
    assert all(x["by_class"]["rare"]["full_panel_margin_change"] == 0.
               for variants in report["interventions"].values() for x in variants.values())
    args.analyze_only = False
    assert runner.run(args)["complete"]  # Normal restart also never recaptures completed forwards.


def test_changed_source_and_protected_output_fail_before_gpu(tmp_path, monkeypatch):
    args = stage_fixture(tmp_path, monkeypatch)
    args.output_dir = args.stage_dir + "/overwrite"
    with pytest.raises(ValueError, match="separate sibling"):
        runner.prepare(args)
    args.output_dir = str(tmp_path / "query")
    path = tmp_path / "old.pth"
    torch.save({}, path)
    with pytest.raises(ValueError, match="input changed"):
        runner.prepare(args)


def test_corrupted_cache_and_mixed_pixels_rejected(tmp_path, monkeypatch):
    args = stage_fixture(tmp_path, monkeypatch)
    ctx = runner.prepare(args)
    fill_forward_cache(ctx)
    path = runner.cache_path(ctx, "old", "query_content", 1)
    value = load_trusted_torch_file(path)
    value["mapped_input_sha256"] = "wrong_pixels"
    torch.save(value, path)
    with pytest.raises(ValueError, match="pixels changed"):
        runner.analyze(ctx, .5)
    value["fingerprint"] = "wrong"
    torch.save(value, path)
    with pytest.raises(ValueError, match="identity mismatch"):
        runner.read_forward(ctx, "old", "query_content", 1)


def test_complete_forward_orchestration_on_cpu_toy_model(tmp_path, monkeypatch):
    """Exercise the real runner/swap/cache/matching; no D2, CUDA or training."""
    from tools import diagnose_tpa_usage as usage

    args = stage_fixture(tmp_path, monkeypatch)
    ctx = runner.prepare(args)
    checkpoint_digests = {s: file_identity(ctx.source["sources"][s + "_checkpoint"]["path"])["sha256"]
                          for s in ("old", "new")}
    calls = []

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            tpa = nn.Module()
            tpa.prototype_queries = nn.Parameter(torch.ones(5, 256))
            tpa.key_proj = nn.Linear(2, 256)
            tpa.value_proj = nn.Linear(2, 2)
            tpa.register_buffer("slot_prior_strength", torch.tensor(.2))
            tpa.register_buffer("prototype_mode_strength", torch.tensor(0.))
            head = nn.Module()
            head.tpa = tpa
            head.use_tpa = head.norm_weight = True
            self.class_embed = nn.ModuleList([head])
            self.content_layer = nn.Linear(2, 2, bias=False)
            self.transformer = nn.Module()
            self.transformer.decoder = nn.Module()
            self.transformer.decoder.num_layers = 6
            self.score_ensemble = True

        def forward(self, inputs):
            assert not self.training and not torch.is_grad_enabled()
            assert all(not p.requires_grad for p in self.parameters())
            calls.append(1)
            features = self.content_layer(torch.eye(2))
            xyxy = ctx.samples["old", 1]["query_boxes"] / 100
            boxes = torch.cat(((xyxy[:, :2] + xyxy[:, 2:]) / 2, xyxy[:, 2:] - xyxy[:, :2]), -1)
            self.capture.update(projected_features=features[None], query_boxes=boxes[None],
                                prototypes=ctx.banks["old"]["prototypes"])
            return []

    model = Model()
    aug = SimpleNamespace(_target_="ResizeShortestEdge", short_edge_length=[32])
    mapper_cfg = SimpleNamespace(augmentation=[aug], augmentation_with_crop=None, is_train=False)
    cfg = SimpleNamespace(model="model", dataloader=SimpleNamespace(
        test=SimpleNamespace(mapper=mapper_cfg, dataset=SimpleNamespace(names="fixture"))))
    config_module = ModuleType("detectron2.config")
    config_module.LazyConfig = SimpleNamespace(load=lambda _: cfg, apply_overrides=lambda value, _: value)
    config_module.instantiate = lambda value: model if value == "model" else (lambda row: dict(
        row, image=torch.zeros(3, 32, 32), width=100, height=100))
    data_module = ModuleType("detectron2.data")
    data_module.get_detection_dataset_dicts = lambda **_: [{"image_id": 1}]
    data_module.MetadataCatalog = SimpleNamespace(get=lambda _: SimpleNamespace(
        json_file=ctx.source["sources"]["annotations"]["path"]))
    monkeypatch.setitem(sys.modules, "detectron2.config", config_module)
    monkeypatch.setitem(sys.modules, "detectron2.data", data_module)

    def install(m, capture):
        m.capture = capture
        return m.class_embed[0]

    monkeypatch.setattr(usage, "install_capture_hooks", install)
    result = runner.run(args)
    assert result["complete"] and len(calls) == 6
    effects = result["interventions"]
    forward = effects["old"]["query_content"]["by_class"]["rare"]["full_panel_margin_change"]
    reverse = effects["new"]["query_content"]["by_class"]["rare"]["full_panel_margin_change"]
    assert forward != 0 and forward == pytest.approx(-reverse, abs=1e-12)
    assert len(result["bidirectional_check"]) == 3
    # The normal command resumes without even constructing a model.
    runner.run(args)
    assert len(calls) == 6
    # Original checkpoint files and the live endpoint are unmodified.
    for side, digest in checkpoint_digests.items():
        assert file_identity(ctx.source["sources"][side + "_checkpoint"]["path"])["sha256"] == digest
    assert all(torch.equal(v, ctx.states["new"][k]) for k, v in model.state_dict().items())
