from copy import deepcopy
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from lami_dino.models import TextPrototypeAggregator
from tools.audit_tpa_geometry import run, summarize_queries
from tools.compare_rare_pr_reports import file_identity
from tools.diagnose_rare_fp_regions import fingerprint
from tools.rare_region_pairing_ops import compare_region, replay_sample
from tools.tpa_geometry_audit_ops import (
    class_geometry, extract_shared_tpa, reconstruct_tpa, stage_geometry, tensor_digest,
    validate_reconstructed_bank,
)


def make_tpa(strength=1.5):
    torch.manual_seed(42)
    return TextPrototypeAggregator(dim=4, hidden_dim=3, num_prototypes=3, dropout=.4,
                                   tau=.07, slot_prior_strength=.2,
                                   prototype_mode_strength=strength).eval()


def checkpoint_for(tpa):
    return {"iteration": 85199, "model": {
        f"module.transformer.decoder.class_embed.{i}.tpa.{k}": v.clone()
        for i in range(2) for k, v in tpa.state_dict().items()}}


@pytest.mark.parametrize("strength", [0., .5, 1.5])
def test_reconstruction_exactly_matches_real_tpa_eval(strength):
    tpa = make_tpa(strength)
    prompts = torch.randn(5, 8, 4)
    state, info = extract_shared_tpa(checkpoint_for(tpa))
    result = reconstruct_tpa(prompts, state, .07)
    with torch.no_grad():
        expected, _ = tpa(prompts, with_loss=False, update_monitor_state=False)
    assert torch.equal(result["after"], expected)
    assert info["aliases_equal"] and len(info["prefixes"]) == 2
    for index in range(5):
        row = class_geometry(result, index)
        if strength == 0:
            assert row["mean_shift_over_value_center_norm"] == 0
        else:
            assert row["realized_radii"] == pytest.approx([row["target_radius"]] * 3, abs=2e-6)
        json.dumps(row, allow_nan=False)


def test_zero_mean_residuals_do_not_imply_center_preservation_after_normalizing_each():
    center = torch.tensor([[2., 1.]])
    residual = torch.tensor([[[1., 0.], [0., 1.], [-1., -1.]]])
    unit = residual / residual.norm(dim=-1, keepdim=True)
    before = center[:, None] + residual
    radius = torch.tensor([2.])
    after = center[:, None] + radius[:, None, None] * unit
    r = {"value_center": center, "residual": residual, "unit_residual": unit,
         "before": before, "after": after, "radius": radius, "enabled": True}
    row = class_geometry(r, 0)
    assert row["pre_mean_offset_over_value_center_norm"] == 0
    assert row["post_mean_offset_over_value_center_norm"] > .1
    assert row["pre_post_center_angle_deg"] > 0
    assert row["center_identity_max_abs_error"] < 1e-6


def test_degenerate_residual_and_center_are_flagged_not_nan():
    tpa = make_tpa()
    with torch.no_grad():
        tpa.value_proj.weight.zero_()
        tpa.value_proj.bias.zero_()
    state, _ = extract_shared_tpa(checkpoint_for(tpa))
    row = class_geometry(reconstruct_tpa(torch.zeros(1, 8, 4), state, .07), 0)
    assert row["clamped_residual_slots"] == 3
    assert row["value_center_radius_clamped"]
    assert row["pre_post_center_angle_deg"] is None
    assert row["before"]["effective_rank"] is None
    assert row["mean_shift_over_value_center_norm"] is None
    json.dumps(row, allow_nan=False)


def test_normalization_changes_mean_direction_when_slot_norms_differ():
    result = stage_geometry(torch.tensor([[10., 0.], [0., 1.]]))
    assert result["normalization_center_angle_deg"] == pytest.approx(39.2894, abs=1e-3)
    assert result["unit_slot_mean_norm"] == pytest.approx(2 ** -.5)


def test_unequal_shared_aliases_and_missing_weights_fail():
    checkpoint = checkpoint_for(make_tpa())
    checkpoint["model"]["module.transformer.decoder.class_embed.1.tpa.value_proj.weight"][0, 0] += .1
    with pytest.raises(ValueError, match="Unequal shared"):
        extract_shared_tpa(checkpoint)
    checkpoint = checkpoint_for(make_tpa())
    del checkpoint["model"]["module.transformer.decoder.class_embed.0.tpa.prototype_queries"]
    with pytest.raises(ValueError, match="Missing/nonfinite"):
        extract_shared_tpa(checkpoint)


def test_legacy_missing_strength_buffers_default_to_zero():
    checkpoint = checkpoint_for(make_tpa())
    checkpoint["model"] = {k: v for k, v in checkpoint["model"].items()
                           if not k.endswith(("prototype_mode_strength", "slot_prior_strength"))}
    state, _ = extract_shared_tpa(checkpoint)
    assert state["prototype_mode_strength"] == state["slot_prior_strength"] == 0


def fixture(tmp_path):
    tpa = make_tpa()
    checkpoint = checkpoint_for(tpa)
    ckpt = tmp_path / "checkpoint.pth"
    torch.save(checkpoint, ckpt)
    prompts = torch.randn(2, 8, 4)
    prompt_path = tmp_path / "prompts.npy"
    np.save(prompt_path, prompts.numpy())
    state, _ = extract_shared_tpa(checkpoint)
    reconstruction = reconstruct_tpa(prompts, state, .07)
    protocol = {"alpha": 0., "beta": .3, "novel_scale": 3., "max_dets": 3, "tpa_tau": .07, "cls_tau": .07}
    categories = {11: {"id": 11, "name": "koala", "frequency": "r"},
                  22: {"id": 22, "name": "other", "frequency": "c"}}
    ann = tmp_path / "annotations.json"
    ann.write_text(json.dumps({"categories": list(categories.values())}))
    inputs = {"schema_version": 1, "image_ids": [1, 2], "new_sha256": file_identity(ckpt)["sha256"],
              "annotations_sha256": file_identity(ann)["sha256"], **protocol}
    signature = fingerprint(inputs)
    directory = tmp_path / "parent_cache"
    (directory / "new").mkdir(parents=True)
    (directory / "manifest.json").write_text(json.dumps({"inputs": inputs, "fingerprint": signature}))
    bank = {"fingerprint": signature, "label": "new", "category_ids": [11, 22],
            "prototypes": reconstruction["after"], "prompt_sha256": tensor_digest(prompts),
            "temperature": .07, "logit_scale": 5., "cls_bias": -3., "tpa_tau": .07,
            "vlm_text": torch.eye(4)[:2], "vlm_temperature": 1., "novel_mask": torch.tensor([True, False]),
            "iteration": 85199, "slot_prior_strength": float(state["slot_prior_strength"]),
            "prototype_mode_strength": float(state["prototype_mode_strength"])}
    sample = {"fingerprint": signature, "label": "new", "image_id": 1,
              "features": torch.eye(4)[:2], "roi_features": torch.eye(4)[:2],
              "query_boxes": torch.tensor([[0., 0., 10., 10.], [20., 0., 30., 10.]]),
              "native_replay_check": {"logit_max_abs_error": 0., "score_max_abs_error": 0.}}
    torch.save(bank, directory / "new" / "bank.pt")
    replay = replay_sample(sample, bank, protocol)
    rows = []
    for image, kind in ((1, "fp"), (2, "tp")):
        torch.save({**sample, "image_id": image}, directory / "new" / f"{image}.pt")
        rows.append(compare_region({"category": "koala", "source_side": "new", "kind": kind,
                                    "image_id": image, "detection_id": image, "class_global_rank": image,
                                    "box_xyxy": [0., 0., 10., 10.], "score": float(replay["scores"][0, 0])},
                                   {"new": replay, "old": replay}, categories, protocol))
    source = {"complete": True, "regions": rows, "protocol": protocol,
              "annotations": file_identity(ann), "parent_fingerprint": signature,
              "source_details": {"sha256": "details-test"}, "checkpoints": {"new": file_identity(ckpt)},
              "sample_sources": [{"side": "new", "image_id": image, "directory": str(directory),
                                  "fingerprint": signature} for image in (1, 2)]}
    path = tmp_path / "regions.json"
    path.write_text(json.dumps(source))
    args = SimpleNamespace(source_json=str(path), output=str(tmp_path / "audit.json"), side="new",
                           checkpoint=None, prompt_bank=str(prompt_path), annotations=None, cpu_threads=1)
    return args, source, directory, bank, reconstruction


def test_cpu_audit_preserves_inputs_and_exact_native_and_clip_replay(tmp_path, monkeypatch):
    args, source, directory, _, _ = fixture(tmp_path)
    before = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    def forbidden(*args, **kwargs):
        raise AssertionError("No detector/image forward permitted")
    monkeypatch.setattr("tools.diagnose_rare_fp_regions.fill_missing_images", forbidden)
    report = run(args)
    assert report["complete"] and report["geometry_summary"]["all"]["classes"] == 2
    assert len(report["regions"]) == 2
    assert report["selected_ordering"][0]["selected_pairs"] == 1
    query = report["regions"][0]["queries"][0]
    assert query["reconstructed_logit_abs_error"] < 1e-8
    assert query["clip_log_probability_fixed"] == source["regions"][0]["source_candidates"][0]["clip_log_probability"]
    delta = query["after_minus_before"]
    assert delta["native_logit"] == pytest.approx(delta["center_response"] + delta["mode_delta"] + delta["bias"])
    assert json.loads(open(args.output).read()) == report
    for path, content in before.items():
        assert path.read_bytes() == content


@pytest.mark.parametrize("changed", ["checkpoint", "prompt", "bank", "query_score", "protocol"])
def test_mismatched_artifacts_fail_without_output(tmp_path, changed):
    args, source, directory, bank, _ = fixture(tmp_path)
    if changed == "checkpoint":
        checkpoint = torch.load(source["checkpoints"]["new"]["path"], weights_only=True)
        checkpoint["iteration"] = 0
        torch.save(checkpoint, source["checkpoints"]["new"]["path"])
    elif changed == "prompt":
        np.save(args.prompt_bank, np.ones((2, 8, 4), dtype=np.float32))
    elif changed == "bank":
        bank["prototypes"][0, 0] += 1.
        torch.save(bank, directory / "new" / "bank.pt")
    elif changed == "query_score":
        source["regions"][0]["source_candidates"][0]["fused_score"] += .1
    else:
        source["protocol"]["beta"] = .5
    with open(args.source_json, "w") as stream:
        json.dump(source, stream)
    with pytest.raises(ValueError):
        run(args)
    assert not (tmp_path / "audit.json").exists()


def test_missing_cache_never_regenerated(tmp_path):
    args, _, directory, _, _ = fixture(tmp_path)
    (directory / "new" / "1.pt").unlink()
    with pytest.raises(FileNotFoundError):
        run(args)
    assert not (directory / "new" / "1.pt").exists()


def test_protected_outputs(tmp_path):
    args, source, directory, _, _ = fixture(tmp_path)
    for path in (args.source_json, args.prompt_bank, source["checkpoints"]["new"]["path"],
                 str(directory / "new" / "bank.pt")):
        args.output = path
        with pytest.raises(ValueError, match="must not overwrite"):
            run(args)


def test_bank_check_rejects_direction_errors_even_for_small_raw_norms():
    r = {"after": torch.tensor([[[1e-7, 0.]]])}
    with pytest.raises(ValueError, match="differ from native"):
        validate_reconstructed_bank(r, {"prototypes": torch.tensor([[[0., 1e-7]]])})


def test_summary_excludes_ambiguous_and_deduplicates_queries(tmp_path):
    args, _, _, _, _ = fixture(tmp_path)
    report = run(args)
    rows = report["regions"]
    ambiguous = deepcopy(rows[0])
    ambiguous["queries"] *= 2
    summaries, ordering = summarize_queries(rows + rows + [ambiguous])
    assert sum(r["unique_queries"] for r in summaries) == 2
    assert ordering[0]["selected_pairs"] == 1


def test_existing_supplemental_cache_is_supported_without_parent_image(tmp_path):
    args, source, parent, bank, _ = fixture(tmp_path)
    supplement = tmp_path / "region_cache"
    (supplement / "new").mkdir(parents=True)
    inputs = {"image_ids": [1], "parent_fingerprint": source["parent_fingerprint"],
              "details_sha256": source["source_details"]["sha256"]}
    signature = fingerprint(inputs)
    (supplement / "manifest.json").write_text(json.dumps({"inputs": inputs, "fingerprint": signature}))
    sample = torch.load(parent / "new" / "1.pt", weights_only=True)
    torch.save({**sample, "fingerprint": signature}, supplement / "new" / "1.pt")
    torch.save({**bank, "fingerprint": signature}, supplement / "new" / "bank.pt")
    (parent / "new" / "1.pt").unlink()
    source["sample_sources"][0].update(directory=str(supplement), fingerprint=signature)
    with open(args.source_json, "w") as stream:
        json.dump(source, stream)
    assert run(args)["complete"]
    assert not (parent / "new" / "1.pt").exists()


def test_optional_old_side_uses_only_old_native_source_queries(tmp_path):
    args, source, directory, bank, _ = fixture(tmp_path)
    args.side = "old"
    (directory / "old").mkdir()
    source["checkpoints"] = {"old": source["checkpoints"]["new"]}
    manifest = json.loads((directory / "manifest.json").read_text())
    manifest["inputs"]["old_sha256"] = manifest["inputs"].pop("new_sha256")
    manifest["fingerprint"] = fingerprint(manifest["inputs"])
    signature = manifest["fingerprint"]
    (directory / "manifest.json").write_text(json.dumps(manifest))
    source["parent_fingerprint"] = signature
    for entry in source["sample_sources"]:
        entry.update(side="old", fingerprint=signature)
        sample = torch.load(directory / "new" / f"{entry['image_id']}.pt", weights_only=True)
        torch.save({**sample, "label": "old", "fingerprint": signature},
                   directory / "old" / f"{entry['image_id']}.pt")
    torch.save({**bank, "label": "old", "fingerprint": signature}, directory / "old" / "bank.pt")
    for row in source["regions"]:
        row.update(source_side="old", other_side="new")
    with open(args.source_json, "w") as stream:
        json.dump(source, stream)
    result = run(args)
    assert result["side"] == "old" and len(result["regions"]) == 2
