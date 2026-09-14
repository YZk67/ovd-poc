from copy import deepcopy
import json
from types import SimpleNamespace

import pytest
import torch

from lami_dino.pairing_diagnostic_ops import replay_classifier
from tools.analyze_rare_fp_logits import run
from tools.compare_rare_pr_reports import file_identity
from tools.diagnose_rare_fp_regions import fingerprint
from tools.rare_logit_decomposition_ops import decompose_entry, decompose_logit, paired_delta, summarize
from tools.rare_region_pairing_ops import compare_region, replay_sample


def terms(feature, prototypes, bias=-3.):
    return decompose_logit(torch.tensor(feature, dtype=torch.float32),
                           torch.tensor(prototypes, dtype=torch.float32),
                           temperature=.07, logit_scale=50., cls_bias=bias)


@pytest.mark.parametrize("k", [1, 5])
def test_identical_slots_have_no_mode_delta_or_log_k_bias(k):
    result = terms([1., .4], [[2., 1.]] * k)
    assert result["mode_delta"] == pytest.approx(0., abs=1e-10)
    assert result["dispersion_uplift"] == pytest.approx(0., abs=1e-10)
    assert result["native_logit"] == pytest.approx(result["center_response"] - 3.)
    assert result["closure_abs_error"] < 1e-10
    json.dumps(result, allow_nan=False)


def test_signed_mode_delta_is_not_the_nonnegative_lme_uplift():
    centered = terms([1., 0.], [[1., 1.], [1., -1.]])
    assert centered["center_response"] == pytest.approx(50.)
    assert centered["mode_delta"] < -14.
    assert centered["dispersion_uplift"] == pytest.approx(0., abs=1e-10)
    assert centered["center_to_mean_correction"] == pytest.approx(centered["mode_delta"])
    off_center = terms([0., 1.], [[1., 1.], [1., -1.]])
    assert off_center["center_response"] == 0.
    assert off_center["mode_delta"] > 30.
    assert off_center["mode_delta"] == pytest.approx(off_center["dispersion_uplift"])


def test_raw_centroid_is_averaged_before_normalizing_unequal_norm_slots():
    feature = torch.tensor([[0., 1.]])
    bank = torch.tensor([[[10., 0.], [0., 1.]]])
    result = terms(feature[0].tolist(), bank[0].tolist())
    native = float(replay_classifier(feature, bank, cls_bias=-3.))
    centroid = float(replay_classifier(feature, bank.mean(1, keepdim=True), cls_bias=-3.))
    assert result["native_logit"] == pytest.approx(native, abs=1e-4)
    assert result["centroid_logit"] == pytest.approx(centroid, abs=1e-4)
    assert result["center_response"] == pytest.approx(50. / (101 ** .5))
    assert result["mean_slot_response"] == pytest.approx(25.)


def test_bias_is_additive_and_does_not_change_mode_terms():
    first = terms([.2, 1.], [[1., 1.], [1., -1.]], bias=-3.)
    second = terms([.2, 1.], [[1., 1.], [1., -1.]], bias=-1.)
    for key in ("center_response", "mode_delta", "dispersion_uplift"):
        assert first[key] == second[key]
    assert second["native_logit"] - first["native_logit"] == pytest.approx(2.)


def test_zero_centroid_is_flagged_and_has_finite_decomposition():
    result = terms([1., 0.], [[1., 0.], [-1., 0.]])
    assert not result["center_direction_defined"]
    assert result["center_response"] == 0.
    assert result["mode_delta"] > 0.
    json.dumps(result, allow_nan=False)


def test_invalid_features_or_temperature_are_rejected():
    with pytest.raises(ValueError, match="finite"):
        terms([float("nan"), 0.], [[1., 0.]])
    with pytest.raises(ValueError, match="Temperature"):
        decompose_logit(torch.ones(2), torch.ones(1, 2), temperature=0., logit_scale=50., cls_bias=-3.)


def fixture(tmp_path):
    protocol = {"alpha": 0., "beta": .3, "novel_scale": 3., "max_dets": 3, "tpa_tau": .004375, "cls_tau": .07}
    categories = {11: {"id": 11, "name": "koala"}, 22: {"id": 22, "name": "other"}}
    ann = tmp_path / "annotations.json"
    ann.write_text(json.dumps({"categories": list(categories.values())}))
    inputs = {"schema_version": 1, "image_ids": [1, 2],
              "annotations_sha256": file_identity(ann)["sha256"], **protocol}
    signature = fingerprint(inputs)
    directory = tmp_path / "parent_cache"
    directory.mkdir()
    (directory / "manifest.json").write_text(json.dumps({"inputs": inputs, "fingerprint": signature}))
    replays, banks, samples = {}, {}, {}
    for side in ("old", "new"):
        banks[side] = {"fingerprint": signature, "label": side, "category_ids": [11, 22],
                       "prototypes": torch.tensor([[[1., 1.], [1., -1.]], [[0., 1.], [0., 2.]]]),
                       "temperature": .07, "logit_scale": 5., "cls_bias": -3.,
                       "vlm_text": torch.eye(2), "vlm_temperature": 1., "novel_mask": torch.tensor([True, False])}
        samples[side] = {"fingerprint": signature, "label": side, "image_id": 1,
                         "features": torch.eye(2), "roi_features": torch.eye(2),
                         "query_boxes": torch.tensor([[0., 0., 10., 10.], [20., 0., 30., 10.]]),
                         "native_replay_check": {"logit_max_abs_error": 0., "score_max_abs_error": 0.}}
        (directory / side).mkdir()
        torch.save(banks[side], directory / side / "bank.pt")
        for image in (1, 2):
            torch.save({**samples[side], "image_id": image}, directory / side / f"{image}.pt")
        replays[side] = replay_sample(samples[side], banks[side], protocol)
    rows = []
    for image, kind in ((1, "fp"), (2, "tp")):
        rows.append(compare_region({"category": "koala", "source_side": "new", "kind": kind,
                                    "image_id": image, "detection_id": image, "class_global_rank": image,
                                    "box_xyxy": [0., 0., 10., 10.], "score": float(replays["new"]["scores"][0, 0])},
                                   replays, categories, protocol))
    report = {"complete": True, "regions": rows, "protocol": protocol,
              "annotations": file_identity(ann), "parent_fingerprint": signature,
              "source_details": {"sha256": "details-test"},
              "sample_sources": [{"side": side, "image_id": image, "directory": str(directory), "fingerprint": signature}
                                 for side in ("old", "new") for image in (1, 2)]}
    path = tmp_path / "regions.json"
    path.write_text(json.dumps(report))
    args = SimpleNamespace(source_json=str(path), output=str(tmp_path / "decomposition.json"), annotations=None, cpu_threads=1)
    return args, report, directory, samples, banks, protocol


def test_cache_only_pipeline_checks_sources_and_preserves_inputs(tmp_path, monkeypatch):
    args, source, directory, _, _, _ = fixture(tmp_path)
    before = {p: p.read_bytes() for p in directory.rglob("*") if p.is_file()}
    before[args.source_json] = open(args.source_json, "rb").read()
    def forbidden(*args, **kwargs):
        raise AssertionError("No forward or cache regeneration allowed")
    monkeypatch.setattr("tools.diagnose_rare_fp_regions.fill_missing_images", forbidden)
    result = run(args)
    assert result["complete"] and len(result["regions"]) == 2
    assert result["selected_ordering"][0]["selected_pairs"] == 1
    assert all(abs(group["delta_mean"]["native_logit"]) < 1e-9 for group in result["paired_summary"])
    assert json.loads(open(args.output).read()) == result
    for path, content in before.items():
        with open(path, "rb") as stream:
            assert stream.read() == content


def test_controls_preserve_clip_and_bad_query_identity_fails(tmp_path):
    _, source, _, samples, banks, protocol = fixture(tmp_path)
    entry = source["regions"][0]["source_candidates"][0]
    result = decompose_entry(entry, samples["new"], banks["new"], 0, protocol)
    assert result["fixed_query_controls"]["native"]["fused_score"] == pytest.approx(entry["fused_score"], abs=5e-5)
    assert result["fixed_query_controls"]["centroid"]["fused_score"] > entry["fused_score"]
    bad = {**entry, "box_xyxy": [1., 0., 10., 10.]}
    with pytest.raises(ValueError, match="box"):
        decompose_entry(bad, samples["new"], banks["new"], 0, protocol)
    bad = {**entry, "fused_score": entry["fused_score"] + .01}
    with pytest.raises(ValueError, match="scores disagree"):
        decompose_entry(bad, samples["new"], banks["new"], 0, protocol)


def test_missing_cache_fails_without_forward(tmp_path, monkeypatch):
    args, _, directory, _, _, _ = fixture(tmp_path)
    (directory / "new" / "1.pt").unlink()
    with pytest.raises(FileNotFoundError):
        run(args)
    assert not (tmp_path / "decomposition.json").exists()


def test_input_and_cache_output_protection(tmp_path):
    args, _, directory, _, _, _ = fixture(tmp_path)
    for output in (args.source_json, str(directory / "new" / "bank.pt")):
        args.output = output
        with pytest.raises(ValueError, match="must not overwrite"):
            run(args)


def test_old_tp_controls_and_ambiguous_queries_do_not_double_count_summary(tmp_path):
    args, _, _, _, _, _ = fixture(tmp_path)
    report = run(args)
    rows = deepcopy(report["regions"])
    old_tp = {**deepcopy(rows[1]), "source_side": "old"}
    ambiguous = deepcopy(rows[0])
    ambiguous["source_candidates"] *= 2
    grouped, order = summarize(rows + [old_tp, ambiguous])
    assert sum(g["pairs"] for g in grouped) == 2
    assert order[0]["fp_queries"] == order[0]["tp_queries"] == 1


def test_summary_tracks_selected_fp_tp_order_flip_only():
    def row(kind, native, centroid):
        return {"source_side": "new", "kind": kind, "category": "koala", "image_id": 1,
                "new_minus_old_logit_terms": None,
                "source_candidates": [{"query_id": 0 if kind == "fp" else 1,
                                        "fixed_query_controls": {"native": {"fused_log_score": native},
                                                                 "centroid": {"fused_log_score": centroid}}}]}
    _, ordering = summarize([row("fp", -1., -3.), row("tp", -2., -2.)])
    assert ordering[0]["native_only_inversions"] == 1
    assert ordering[0]["fp_above_tp_native"] == 1
    assert ordering[0]["fp_above_tp_centroid"] == 0


def test_paired_delta_sign_and_additive_closure_with_different_features_and_bias():
    old = {"logit_terms": terms([1., .1], [[1., 0.]] * 5, bias=-3.)}
    new = {"logit_terms": terms([.3, 1.], [[1., 1.], [1., -1.]], bias=-2.)}
    delta = paired_delta(new, old, "new")
    assert delta == paired_delta(old, new, "old")
    assert delta["bias"] == 1.
    assert delta["native_logit"] == pytest.approx(delta["center_response"] + delta["mode_delta"] + 1.)


def test_cpu_pipeline_reuses_existing_supplemental_cache(tmp_path):
    args, source, parent, samples, banks, _ = fixture(tmp_path)
    supplemental = tmp_path / "region_cache"
    (supplemental / "new").mkdir(parents=True)
    inputs = {"image_ids": [1], "parent_fingerprint": source["parent_fingerprint"],
              "details_sha256": source["source_details"]["sha256"]}
    signature = fingerprint(inputs)
    (supplemental / "manifest.json").write_text(json.dumps({"inputs": inputs, "fingerprint": signature}))
    torch.save({**banks["new"], "fingerprint": signature}, supplemental / "new" / "bank.pt")
    torch.save({**samples["new"], "fingerprint": signature}, supplemental / "new" / "1.pt")
    (parent / "new" / "1.pt").unlink()
    for entry in source["sample_sources"]:
        if entry["side"] == "new" and entry["image_id"] == 1:
            entry.update(directory=str(supplemental), fingerprint=signature)
    with open(args.source_json, "w") as stream:
        json.dump(source, stream)
    assert run(args)["complete"]
    assert not (parent / "new" / "1.pt").exists()


def test_changed_protocol_is_rejected(tmp_path):
    args, source, _, _, _, _ = fixture(tmp_path)
    source["protocol"]["beta"] = .5
    with open(args.source_json, "w") as stream:
        json.dump(source, stream)
    with pytest.raises(ValueError, match="protocol differs"):
        run(args)
