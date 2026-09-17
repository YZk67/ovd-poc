from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

from tools import trace_rare_gt_queries as cli
from tools.rare_gt_query_ops import analyze_queries, selected_predictions, verify_predictions


GT = {"id": cli.GT_ID, "image_id": cli.IMAGE_ID, "category_id": 920, "bbox": [0., 0., 10., 10.]}
CATEGORIES = {920: {"name": "scarecrow"}, 1: {"name": "other"}}


def replay(scores=None, boxes=None, k=1):
    scores = torch.tensor(scores if scores is not None else [[.1, .2], [.3, .9]])
    boxes = torch.tensor(boxes if boxes is not None else [[0., 0., 10., 10.], [30., 30., 40., 40.]])
    top = scores.flatten().topk(k)
    selected = torch.zeros(scores.numel(), dtype=torch.bool)
    selected[top.indices] = True
    logits = torch.logit(scores.clamp(.001, .999))
    return dict(scores=scores, boxes=boxes, selected=selected.reshape_as(scores), cutoff=float(top.values[-1]),
                det_logits=logits, clip_logits=logits, det_logp=torch.nn.functional.logsigmoid(logits),
                clip_logp=logits.log_softmax(-1),
                log_scores=scores.log(), category_ids=[920, 1])


@pytest.mark.parametrize("scores,boxes,reason,count", [
    ([[.1, .2], [.3, .9]], [[30., 30., 40., 40.], [50., 50., 60., 60.]],
     "no_eligible_final_query_box", 0),
    ([[.1, .2], [.3, .9]], None, "eligible_queries_all_excluded_from_top300", 1),
    ([[.1, .95], [.3, .9]], None, "eligible_query_retained_under_other_categories", 1),
    ([[.99, .2], [.3, .9]], None, "eligible_true_class_retained", 1),
])
def test_distinguish_preselection_box_and_score_failures(scores, boxes, reason, count):
    result = analyze_queries(replay(scores, boxes), GT, CATEGORIES)
    assert result["by_iou"]["0.50"]["reason"] == reason
    assert result["by_iou"]["0.50"]["raw_eligible_queries"] == count
    assert len(result["all_queries"]) == 2
    json.dumps(result, allow_nan=False)


def test_best_score_eligible_not_best_iou_and_exact_tie_ranks():
    data = replay([[.4, .3], [.9, .9]], [[0., 0., 10., 10.], [1., 1., 10., 10.]], k=2)
    result = analyze_queries(data, GT, CATEGORIES)
    assert result["best_iou_query"]["query_id"] == 0
    best = result["by_iou"]["0.50"]["best_fused_true_class_eligible"]
    assert best["query_id"] == 1 and best["image_pair_rank_interval"] == [1, 2]
    assert result["by_iou"]["0.90"]["best_fused_true_class_eligible"]["query_id"] == 0


def test_query_indices_are_local_and_permutation_does_not_change_coverage():
    data = replay()
    changed = {k: v.flip(0) if torch.is_tensor(v) and v.ndim > 1 else v for k, v in data.items()}
    left, right = [analyze_queries(x, GT, CATEGORIES) for x in (data, changed)]
    assert left["best_iou_query"]["query_id"] != right["best_iou_query"]["query_id"]
    assert left["by_iou"]["0.50"]["reason"] == right["by_iou"]["0.50"]["reason"]


def prediction(score=.5, category=920, box=None):
    return {"image_id": cli.IMAGE_ID, "category_id": category,
            "bbox": box or [0., 0., 10., 10.], "score": score}


def test_prediction_comparison_uses_all_rows_categories_scores_boxes_and_multiplicity():
    data = [prediction(), prediction(category=1), prediction()]
    check = verify_predictions(data, list(reversed(data)))
    assert check["matched"] == 3 and check["ambiguous_rows"] == 2
    for wrong in ([prediction()] * 3, data[:2], [prediction(.7)] + data[1:],
                  [prediction(box=[20., 20., 10., 10.])] + data[1:]):
        with pytest.raises(ValueError):
            verify_predictions(wrong, data)
    assert verify_predictions([prediction(.500001)], [prediction()])["max_score_error"] < 5e-5


def test_ambiguous_matching_is_not_greedy():
    # A0 can match S0/S1; A1 can match S0 only. Both must match one-to-one.
    actual = [prediction(.50001), prediction(.49996)]
    saved = [prediction(.5), prediction(.50005)]
    assert verify_predictions(actual, saved)["matched"] == 2


def test_empty_boxes_removed_after_topk_without_refill():
    data = replay([[.99, .98], [.9, .8]], [[0., 0., 0., 0.], [1., 1., 10., 10.]], k=2)
    assert selected_predictions(data, cli.IMAGE_ID) == []
    assert analyze_queries(data, GT, CATEGORIES)["by_iou"]["0.50"]["reason"] == "eligible_queries_all_excluded_from_top300"


def checkpoint(iteration=56799, mode=0.):
    state = {"prototype_queries": torch.zeros(5, 256), "key_proj.weight": torch.zeros(256, 4),
             "key_proj.bias": torch.zeros(256), "value_proj.weight": torch.zeros(256, 4),
             "value_proj.bias": torch.zeros(256), "slot_prior_strength": torch.tensor(.2),
             "prototype_mode_strength": torch.tensor(mode)}
    return {"iteration": iteration, "model": {"transformer.decoder.class_embed.0.tpa." + k: v
                                              for k, v in state.items()}}


def test_checkpoint_iteration_radius_and_missing_buffers_rejected():
    cli.validate_checkpoint(checkpoint(), "old")
    cli.validate_checkpoint(checkpoint(85199), "new")
    for wrong in (checkpoint(70999), checkpoint(mode=1.5)):
        with pytest.raises(ValueError):
            cli.validate_checkpoint(wrong, "old")
    wrong = checkpoint()
    del wrong["model"]["transformer.decoder.class_embed.0.tpa.prototype_mode_strength"]
    with pytest.raises(ValueError, match="explicit"):
        cli.validate_checkpoint(wrong, "old")


def test_cache_identity_requires_checkpoint_assets_code_annotations_protocol():
    expected = dict(cli.PROTOCOL, annotations_sha256="ann", code_sha256="code", asset_sha256={"text": "text"})
    inputs = dict(expected, new_sha256="weights", image_ids=[cli.IMAGE_ID])
    assert cli.cache_compatible(inputs, expected, "new", "weights")
    for key, value in (("new_sha256", "other checkpoint"), ("annotations_sha256", "other ann"),
                       ("code_sha256", "other code"), ("asset_sha256", {}), ("beta", .4), ("image_ids", [])):
        assert not cli.cache_compatible({**inputs, key: value}, expected, "new", "weights")


def test_bounded_manifest_discovery_no_recursive_large_cache_scan(tmp_path):
    target = tmp_path / "run/pairing_cache/manifest.json"
    target.parent.mkdir(parents=True)
    target.write_text('{}')
    deep = tmp_path / "nested/run/pairing_cache/manifest.json"
    deep.parent.mkdir(parents=True)
    deep.write_text('{}')
    args = SimpleNamespace(reuse_cache=[], cache_search_root=str(tmp_path))
    assert cli.candidate_manifests(args, tmp_path / "own") == [target]


def test_source_cache_side_is_not_checkpoint_identity(tmp_path):
    root = tmp_path / "pairing_cache"
    expected = dict(cli.PROTOCOL, annotations_sha256="ann", code_sha256="code", asset_sha256={"text": "text"})
    inputs = dict(expected, old_sha256="12ep", new_sha256="8ep", image_ids=[cli.IMAGE_ID])
    cli.locked_manifest(root / "manifest.json", inputs)
    for side in cli.ITERATIONS:
        branch = root / side
        branch.mkdir()
        torch.save({}, branch / "bank.pt")
        torch.save({}, branch / f"{cli.IMAGE_ID}.pt")
    unrelated = tmp_path / "unrelated.json"
    unrelated.write_text('{"not": "a pairing cache"}')
    found = cli.find_caches([unrelated, root / "manifest.json"], expected,
                            {"old_checkpoint": {"sha256": "8ep"}, "new_checkpoint": {"sha256": "12ep"}})
    assert found["old"][1] == "new" and found["new"][1] == "old"


def test_dense_cache_rejects_partial_queries_wrong_bank_and_native_check(tmp_path):
    root = tmp_path / "pairing_cache"
    signature = cli.locked_manifest(root / "manifest.json", {"test": True})
    branch = root / "old"
    branch.mkdir()
    ids = list(range(1, 1204))
    bank = dict(fingerprint=signature, label="old", iteration=56799, category_ids=ids,
                prototype_mode_strength=0., slot_prior_strength=.2, prototypes=torch.zeros(1203, 5, 256),
                tpa_tau=.004375, temperature=.07, novel_mask=torch.tensor([c == 920 for c in ids]))
    sample = dict(fingerprint=signature, label="old", image_id=cli.IMAGE_ID, width=640, height=479,
                  features=torch.zeros(900, 256), roi_features=torch.zeros(900, 768),
                  query_boxes=torch.zeros(900, 4),
                  native_replay_check={"logit_max_abs_error": 1e-5, "score_max_abs_error": 1e-7})
    dataset = {"categories": [{"id": c, "frequency": "r" if c == 920 else "f"} for c in ids]}
    image = {"width": 640, "height": 479}

    def load(b=bank, s=sample):
        torch.save(b, branch / "bank.pt")
        torch.save(s, branch / f"{cli.IMAGE_ID}.pt")
        return cli.load_cache(root / "manifest.json", "old", "old", dataset, image)

    assert load()[1]["iteration"] == 56799
    with pytest.raises(ValueError, match="does not match"):
        load(s={**sample, "features": torch.zeros(300, 256)})
    with pytest.raises(ValueError, match="does not match"):
        load(b={**bank, "prototype_mode_strength": 1.5})
    with pytest.raises(ValueError, match="native"):
        load(s={**sample, "native_replay_check": {"logit_max_abs_error": .1, "score_max_abs_error": .1}})


def setup_run(tmp_path, monkeypatch, cache_only=False):
    data = replay([[.1, .2]] * 150, [[0., 0., 10., 10.]] * 150, k=300)
    saved = [{**r, "detection_id": n+1} for n, r in enumerate(selected_predictions(data, cli.IMAGE_ID))]
    image = {"id": cli.IMAGE_ID, "width": 100, "height": 100}
    annotations = tmp_path / "ann.json"
    annotations.write_text(json.dumps({"images": [image], "annotations": [GT],
                                       "categories": [{"id": k, **v} for k, v in CATEGORIES.items()]}))
    trace = {"complete": True, "category": "scarecrow", "gt": GT, "image": image,
             "endpoint_labels": {"A": "8ep", "B": "12ep"},
             "sources": {"annotations": cli.file_identity(annotations), "old_predictions": {}, "new_predictions": {}},
             "endpoints": {s: {"selected_detections_by_gt_iou": saved, "selected_predictions_count": 300,
                               "by_iou": {k: {"selected_true_class_eligible": 150, "selected_any_class_eligible": 300}
                                          for k in ("0.50", "0.75", "0.90")}}
                           for s in cli.ITERATIONS}}
    source = tmp_path / "trace.json"
    source.write_text(json.dumps(trace))
    old, new = tmp_path / "8.pth", tmp_path / "12.pth"
    torch.save(checkpoint(), old)
    torch.save(checkpoint(85199), new)
    config = tmp_path / "config.py"
    config.write_text('# test only')
    monkeypatch.setattr(cli, "model_code_hash", lambda path: "code")
    monkeypatch.setattr(cli, "input_asset_hashes", lambda *a: {})
    monkeypatch.setattr(cli, "load_cache", lambda *a: ({"native_replay_check": {}}, {}, {}))
    monkeypatch.setattr(cli, "replay_sample", lambda *a, **kw: data)
    calls = []

    def dump(args, side, panel, dataset, fingerprint, cache):
        assert panel == {"image_ids": [cli.IMAGE_ID]}
        assert args.device == "cpu" and args.max_dets == 300
        calls.append(side)
        branch = cache / side
        branch.mkdir()
        torch.save({}, branch / "bank.pt")
        torch.save({}, branch / f"{cli.IMAGE_ID}.pt")

    monkeypatch.setattr(cli, "dump_checkpoint", dump)
    args = SimpleNamespace(cpu_threads=1, trace_json=str(source), annotations=None, old_checkpoint=str(old),
                           new_checkpoint=str(new), config_file=str(config), output_dir=str(tmp_path / "output"),
                           reuse_cache=[], cache_search_root=None, cache_only=cache_only, device="cpu")
    return args, calls


def test_driver_exact_two_single_image_calls_then_zero_calls_on_reuse(tmp_path, monkeypatch):
    args, calls = setup_run(tmp_path, monkeypatch)
    result = cli.run(args)
    assert result["complete"] and result["new_forward_calls"] == 2
    assert result["training_updates"] == 0 and calls == ["old", "new"]
    args.cache_only = True
    again = cli.run(args)
    assert again["complete"] and again["new_forward_calls"] == 0
    assert calls == ["old", "new"]


def test_cache_only_missing_does_not_call_model_or_write_report(tmp_path, monkeypatch):
    args, calls = setup_run(tmp_path, monkeypatch, cache_only=True)
    with pytest.raises(FileNotFoundError, match="NO forward"):
        cli.run(args)
    assert calls == [] and not (Path(args.output_dir) / "report.json").exists()


def test_reproduction_failure_stops_before_other_endpoint_or_conclusion(tmp_path, monkeypatch):
    args, calls = setup_run(tmp_path, monkeypatch)
    monkeypatch.setattr(cli, "replay_sample", lambda *a, **kw: replay())
    with pytest.raises(ValueError, match="count mismatch"):
        cli.run(args)
    report = json.loads((Path(args.output_dir) / "report.json").read_text())
    assert calls == ["old"] and not report["complete"] and not report["endpoints"]


def test_cli_help_without_detectron_or_lvis():
    result = subprocess.run([sys.executable, str(Path(cli.__file__)), "--help"], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "--cache-only" in result.stdout
