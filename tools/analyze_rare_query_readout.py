#!/usr/bin/env python3
"""CPU-only query x terminal-TPA-bank x bias audit of an authenticated single-GT cache.

No model instantiation, GPU forward, training, full dataset evaluation, or cache
regeneration. Reads only the completed trace and its hashed tensor/manifest files.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
import torch.nn.functional as F

from lami_dino.checkpoint_init import load_trusted_torch_file
from lami_dino.diagnostic_ops import pairwise_box_iou_xyxy
from tools.compare_rare_pr_reports import file_identity, load_json, save_json
from tools.diagnose_detector_tpa_pairing import locked_manifest
from tools.diagnose_rare_fp_regions import read_manifest, validate_sample
from tools.rare_query_readout_ops import SIDES, check_banks, primary_attribution, readout_grid


def validate_source(source):
    if (source.get("complete") is not True or source["gt"]["id"] != 137708
            or source["gt"]["image_id"] != 218917 or source["gt"]["category_id"] != 920
            or source["image"]["id"] != 218917):
        raise ValueError("Need completed single-GT query trace for scarecrow 137708 / image 218917")
    protocol = {"alpha": 0., "beta": .3, "novel_scale": 3., "tpa_tau": .004375, "cls_tau": .07, "max_dets": 300}
    if any(source["protocol"].get(k) != v for k, v in protocol.items()):
        raise ValueError("Source inference protocol differs")
    for side, iteration in (("old", 56799), ("new", 85199)):
        endpoint = source["endpoints"][side]
        if (endpoint["iteration"] != iteration or endpoint["raw_queries"] != 900
                or endpoint["vocabulary_size"] != 1203
                or not endpoint["saved_predictions_check"]["all_predictions_reproduced"]
                or endpoint["saved_predictions_check"]["matched"] != 300):
            raise ValueError("Expected native-verified no-radius 8ep and 12ep endpoints")


def checked_identity(record):
    actual = file_identity(record["path"])
    if any(actual[k] != record[k] for k in ("sha256", "bytes")):
        raise ValueError(f"Cache bytes changed since verified trace: {record['path']}")
    return actual


def validate_endpoint(source, side, sample, bank, dense, manifest):
    endpoint = source["endpoints"][side]
    label = endpoint["cache"]["source_label"]
    validate_sample(sample, bank, manifest["fingerprint"], label, source["image"]["id"])
    inputs = manifest["inputs"]
    expected = source["protocol"]
    for key in ("alpha", "beta", "novel_scale", "tpa_tau", "cls_tau", "max_dets",
                "annotations_sha256", "asset_sha256", "code_sha256"):
        if inputs.get(key) != expected[key]:
            raise ValueError(f"Cache/source protocol differs: {key}")
    if (inputs.get(label+"_sha256") != expected[side+"_sha256"]
            or source["image"]["id"] not in inputs["image_ids"]
            or dense["fingerprint"] != source["fingerprint"]
            or dense["iteration"] != endpoint["iteration"] or bank["iteration"] != endpoint["iteration"]
            or dense["image_id"] != source["image"]["id"]
            or bank["prototype_mode_strength"] != 0. or abs(bank["slot_prior_strength"]-.2) > 1e-6
            or bank["temperature"] != expected["cls_tau"] or bank["tpa_tau"] != expected["tpa_tau"]
            or bank["category_ids"] != dense["category_ids"]
            or len(set(bank["category_ids"])) != 1203
            or (sample["width"], sample["height"]) != (source["image"]["width"], source["image"]["height"])):
        raise ValueError("Cache/checkpoint/geometry identity mismatch")
    for container, shapes in ((sample, {"features": (900, 768), "roi_features": (900, 768), "query_boxes": (900, 4)}),
                              (bank, {"prototypes": (1203, 5, 768), "vlm_text": (1203, 768), "novel_mask": (1203,)}),
                              (dense, {"boxes_xyxy": (900, 4), "detector_logits": (900, 1203),
                                       "clip_logits": (900, 1203), "fused_scores": (900, 1203), "selected": (900, 1203)})):
        for key, shape in shapes.items():
            t = container[key]
            if not torch.is_tensor(t) or tuple(t.shape) != shape or not torch.isfinite(t).all():
                raise ValueError(f"Invalid complete 768D cache tensor: {key}; expected {shape}")
    if (dense["selected"].dtype != torch.bool or bank["novel_mask"].dtype != torch.bool
            or int(dense["selected"].sum()) != 300
            or not torch.equal(dense["boxes_xyxy"], sample["query_boxes"])):
        raise ValueError("Dense native top300/box cache differs")
    kept, other = dense["fused_scores"][dense["selected"]], dense["fused_scores"][~dense["selected"]]
    if float(kept.min()) < float(other.max()) or abs(float(kept.min())-endpoint["image_cutoff"]) > 5e-5:
        raise ValueError("Dense mask is not the saved all-category native top300")


@torch.no_grad()
def fixed_entries(source, side, sample, bank, dense):
    """Verify geometry/native scores; keep ALL native IoU>=.5 candidates."""
    endpoint, gt = source["endpoints"][side], source["gt"]
    rows = endpoint["all_queries"]
    if len(rows) != 900 or {e["query_id"] for e in rows} != set(range(900)):
        raise ValueError("Source report must contain all unique query IDs")
    rows = sorted(rows, key=lambda e: e["query_id"])
    c = bank["category_ids"].index(gt["category_id"])
    if not bool(bank["novel_mask"][c]):
        raise ValueError("Target category is not novel")
    x, y, w, h = gt["bbox"]
    ious = pairwise_box_iou_xyxy(torch.tensor([[x, y, x+w, y+h]]), sample["query_boxes"].float())[0]
    fields = {"box_xyxy": (sample["query_boxes"], .05), "region_iou": (ious, 1e-5),
              "fused_score": (dense["fused_scores"][:, c], 5e-5),
              "detector_log_probability": (F.logsigmoid(dense["detector_logits"][:, c]), 2e-4),
              "clip_log_probability": (F.log_softmax(dense["clip_logits"], -1)[:, c], 2e-4)}
    errors = {}
    for field, (actual, tolerance) in fields.items():
        saved = torch.tensor([e[field] for e in rows], dtype=torch.float64)
        error = float((actual.double()-saved).abs().max())
        if not math.isfinite(error) or error > tolerance:
            raise ValueError(f"Source query rows disagree with authenticated dense cache: {field}")
        errors[field] = error
    ids = (ious >= .5).nonzero().flatten().tolist()
    if not ids:
        raise ValueError("No eligible candidate; cannot perform the requested fixed-candidate audit")
    best = max(ids, key=lambda q: float(dense["fused_scores"][q, c]))
    declared = endpoint["by_iou"]["0.50"]["best_fused_true_class_eligible"]["query_id"]
    if best != declared or len(ids) != endpoint["by_iou"]["0.50"]["raw_eligible_queries"]:
        raise ValueError("Primary candidate or eligible set changed since trace")
    entries = []
    for q in ids:
        row = dict(rows[q])
        if abs(row["image_topk_threshold"]-endpoint["image_cutoff"]) > 5e-5:
            raise ValueError("Candidate cutoff differs from native image cutoff")
        # Recompute CLIP only for these ROIs; never change it across bank/bias swaps.
        clip = F.log_softmax(sample["roi_features"][q:q+1].float() @ bank["vlm_text"].float().t()
                            * bank["vlm_temperature"], -1)[0, c]
        if abs(float(clip)-row["clip_log_probability"]) > 2e-4:
            raise ValueError("ROI/text cache no longer reproduces fixed CLIP score")
        entries.append(row)
    return entries, best, c, errors


def display(report):
    print("\n=== Fixed-candidate query x terminal bank x bias (CPU only) ===")
    print("features     q bank bias     logit       pdet   class-rank      fused  /native-cutoff")
    for row in report["primary_attribution"]["corners"]:
        print(f"{row['feature_side']:8} {row['query_id']:4} {row['bank_side']:4} {row['bias_side']:4} "
              f"{row['detector_logit']:9.4f} {row['detector_probability']:10.6f} "
              f"{str(row['detector_class_rank_interval']):>12} {row['fused_score']:10.6f} "
              f"{row['frozen_cutoff_ratio']:13.6f}")
    data = report["primary_attribution"]
    print("\n=== Primary endpoint delta: new minus old ===")
    print("Pre-bias conditional effects:", data["pre_bias_swap_effects"])
    print("Detector logit allocation:", data["logit_contributions"])
    print("Weighted log-detector allocation:", data["weighted_log_detector_contributions"])
    print("CLIP log term:", data["weighted_clip_log_delta"])
    print("Native cutoff competition term:", data["competition_log_margin_term"])
    print("Closure errors:", data["closure_errors"])
    print("Scope: native boxes/CLIP stay fixed per feature source. No counterfactual top300 or AP. "
          "Projected query includes upstream TPA and classifier.linear; it is NOT decoder-only.")


def run(args):
    if args.cpu_threads < 1:
        raise ValueError("cpu_threads must be positive")
    torch.set_num_threads(args.cpu_threads)
    source = load_json(args.source_json)
    validate_source(source)
    output = Path(args.output).resolve()
    records = {"source_report": file_identity(args.source_json)}
    for side in SIDES:
        for kind in ("manifest", "bank", "sample"):
            records[f"{side}_{kind}"] = checked_identity(source["endpoints"][side]["cache"][kind])
        records[side+"_dense"] = checked_identity(source["endpoints"][side]["dense_scores"])
    if any(output == Path(r["path"]).resolve() for r in records.values()):
        raise ValueError("Output must not overwrite an input")
    samples, banks, dense, entries, primary, checks = ({} for _ in range(6))
    for side in SIDES:
        print(f"[CPU load] {side}: authenticated cache only; no checkpoint/model forward", flush=True)
        sample = load_trusted_torch_file(records[side+"_sample"]["path"])
        bank = load_trusted_torch_file(records[side+"_bank"]["path"])
        scores = load_trusted_torch_file(records[side+"_dense"]["path"])
        manifest = read_manifest(records[side+"_manifest"]["path"])
        validate_endpoint(source, side, sample, bank, scores, manifest)
        entry, query, class_index, error = fixed_entries(source, side, sample, bank, scores)
        samples[side], banks[side], dense[side] = sample, bank, scores
        entries[side], primary[side], checks[side] = entry, query, error
        print(f"[fixed {side}] primary q={query}, all eligible q={[e['query_id'] for e in entry]}", flush=True)
    check_banks(banks)
    rows = readout_grid(samples, banks, entries, class_index, source["protocol"])
    # Native corners must reproduce the ENTIRE C-way classifier vector, not only
    # one selected score. The grid uses the same helper and float64 arithmetic.
    from lami_dino.pairing_diagnostic_ops import replay_classifier
    for side in SIDES:
        ids = [e["query_id"] for e in entries[side]]
        native = replay_classifier(samples[side]["features"][ids].double(), banks[side]["prototypes"].double(),
            temperature=banks[side]["temperature"], logit_scale=banks[side]["logit_scale"], cls_bias=banks[side]["cls_bias"])
        error = float((native-dense[side]["detector_logits"][ids].double()).abs().max())
        if error > 1e-4:
            raise ValueError(f"Native C-way detector replay failed: {side}, error={error}")
        checks[side]["native_all_category_logit_max_error"] = error
    for row in rows:
        if row["native_corner"]:
            old = next(e for e in entries[row["feature_side"]] if e["query_id"] == row["query_id"])
            if abs(row["fused_score"]-old["fused_score"]) > 5e-5:
                raise ValueError("Native fused score not reproduced")
    fingerprint = locked_manifest(output.with_name(output.stem+"_manifest.json"), {
        "schema_version": 1, "sources": records,
        "analysis_code": {name: file_identity(ROOT/"tools"/name)["sha256"]
                          for name in ("analyze_rare_query_readout.py", "rare_query_readout_ops.py")}})
    report = {"complete": True, "fingerprint": fingerprint, "sources": records,
              "image_id": source["image"]["id"], "gt_id": source["gt"]["id"],
              "protocol": source["protocol"], "new_forward_calls": 0, "training_updates": 0,
              "fixed_candidates": entries, "native_replay_checks": checks,
              "banks": {s: {k: banks[s][k] for k in ("iteration", "cls_bias", "temperature", "logit_scale")}
                        for s in SIDES},
              "all_candidate_readouts": rows,
              "primary_attribution": primary_attribution(rows, primary, source["protocol"]["beta"]),
              "limits": ["One post-hoc selected GT, not global rare AP or precision evidence.",
                         "The two primary queries/boxes are independently GT-anchored, not identical across checkpoints.",
                         "Feature-side difference includes decoder, classifier projection, upstream TPA/query fusion and other upstream paths.",
                         "Only the TERMINAL bank is swapped; initial query construction is not rerun.",
                         "CLIP/boxes/native cutoff are frozen within each candidate; counterfactual image ranks and TP/FP are NOT computed.",
                         "Bias is class-independent: it can change sigmoid/fused scores but cannot change within-query category order.",
                         "Factor allocations are arithmetic endpoint descriptions, not historical loss/update causality."]}
    save_json(output, report)
    display(report)
    print(f"[save] {output}", flush=True)
    return report


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-json", required=True, help="Completed trace_rare_gt_queries.py report")
    p.add_argument("--output", required=True)
    p.add_argument("--cpu-threads", type=int, default=4)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
