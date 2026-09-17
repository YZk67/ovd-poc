#!/usr/bin/env python3
"""Locate sampled scarecrow selection losses/recoveries between 8ep and 12ep.

Requires the completed trace_rare_gt_queries.py report and BOTH endpoint caches.
Only existing 9/10/11ep checkpoints may receive a single-image native forward:
at most THREE forwards, no training, no full-validation inference, no swaps.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from lami_dino.checkpoint_init import load_trusted_torch_file
from tools.analyze_rare_query_readout import checked_identity, validate_endpoint, validate_source
from tools.compare_rare_pr_reports import file_identity, load_json, save_json
from tools.diagnose_detector_tpa_pairing import check_replay, dump_checkpoint, input_asset_hashes, locked_manifest
from tools.diagnose_rare_fp_regions import model_code_hash, read_manifest
from tools.rare_gt_query_ops import analyze_queries, selected_predictions, verify_predictions
from tools.rare_gt_timeline_ops import EPOCH_ITERATIONS, IOU_KEYS, summarize_timeline
from tools.rare_region_pairing_ops import replay_sample
from tools.trace_rare_gt_queries import (
    GT_ID, IMAGE_ID, PROTOCOL, cache_compatible, candidate_manifests, load_cache, validate_checkpoint,
)


def endpoint_analysis(source, side, categories):
    """Authenticate and recheck cached endpoints; NEVER fall back to inference."""
    endpoint = source["endpoints"][side]
    records = {key: checked_identity(endpoint["cache"][key]) for key in ("manifest", "sample", "bank")}
    records["dense"] = checked_identity(endpoint["dense_scores"])
    sample = load_trusted_torch_file(records["sample"]["path"])
    bank = load_trusted_torch_file(records["bank"]["path"])
    dense = load_trusted_torch_file(records["dense"]["path"])
    manifest = read_manifest(records["manifest"]["path"])
    validate_endpoint(source, side, sample, bank, dense, manifest)
    replay = replay_sample(sample, bank, PROTOCOL, device="cpu")
    check = check_replay(dense["detector_logits"], dense["fused_scores"],
                         replay["det_logits"], replay["scores"], max_dets=300)
    original = dict(boxes=dense["boxes_xyxy"], scores=dense["fused_scores"],
                    selected=dense["selected"], category_ids=dense["category_ids"])
    predictions_check = verify_predictions(selected_predictions(replay, IMAGE_ID),
                                           selected_predictions(original, IMAGE_ID))
    analysis = analyze_queries(replay, source["gt"], categories)
    for key in IOU_KEYS:
        for field in ("raw_eligible_queries", "retained_true_class_queries", "retained_any_class_pairs"):
            if analysis["by_iou"][key][field] != endpoint["by_iou"][key][field]:
                raise ValueError("Endpoint coverage changed since native-verified source; NO recapture allowed")
    return {**analysis, "iteration": endpoint["iteration"], "cache": endpoint["cache"],
            "dense_scores": records["dense"], "endpoint_replay_check": check,
            "saved_predictions_check": predictions_check,
            "verification_scope": "Authenticated endpoint dense replay and all 300 source image predictions"}


def checkpoint_plan(source, directory):
    original = Path(source["protocol"]["old_checkpoint"]).parent.resolve()
    if original != Path(source["protocol"]["new_checkpoint"]).parent.resolve():
        raise ValueError("Source endpoints must come from one no-radius training directory")
    directory = Path(directory).resolve() if directory else original
    if directory != original:
        # A moved run directory is allowed, but a similarly named run is not.
        for side, filename in (("old", "model_0056799.pth"), ("new", "model_final.pth")):
            if file_identity(directory / filename)["sha256"] != source["protocol"][side + "_sha256"]:
                raise ValueError("Relocated checkpoint directory does not contain the source endpoints")
    present, missing = {}, {}
    for epoch in (9, 10, 11):
        path = directory / f"model_{EPOCH_ITERATIONS[epoch]:07d}.pth"
        if not path.exists():
            missing[epoch] = str(path)
            print(f"[missing] {epoch}ep: {path}; skip, do not train or substitute another checkpoint", flush=True)
            continue
        if not path.is_file():
            raise ValueError(f"Checkpoint is not a regular file: {path}")
        print(f"[check] {epoch}ep: {path}", flush=True)
        identity = file_identity(path)
        checkpoint = load_trusted_torch_file(path)
        validate_checkpoint(checkpoint, f"{epoch}ep", expected_iteration=EPOCH_ITERATIONS[epoch])
        del checkpoint
        present[epoch] = identity
    return directory, present, missing


def find_stage_cache(manifests, protocol, identity):
    for path in manifests:
        raw = load_json(path)
        if not isinstance(raw, dict) or not isinstance(raw.get("inputs"), dict):
            continue
        for label in ("old", "new"):
            if cache_compatible(raw["inputs"], protocol, label, identity["sha256"]):
                read_manifest(path)  # Validate fingerprint before accepting any payload.
                branch = path.parent / label
                if (branch / "bank.pt").is_file() and (branch / f"{IMAGE_ID}.pt").is_file():
                    return path, label
    return None


def cached_stage(found, epoch, dataset, source):
    sample, bank, provenance = load_cache(*found, f"{epoch}ep", dataset, source["image"],
                                          expected_iteration=EPOCH_ITERATIONS[epoch])
    replay = replay_sample(sample, bank, PROTOCOL, device="cpu")
    analysis = analyze_queries(replay, source["gt"], {c["id"]: c for c in dataset["categories"]})
    return {**analysis, "iteration": EPOCH_ITERATIONS[epoch], "cache": provenance,
            "native_replay_check": sample["native_replay_check"],
            "verification_scope": "Native single-image dense classifier/fusion replay; no independent saved full-validation JSON"}


def capture_inputs(source, identity):
    return {**source["protocol"], "image_ids": [IMAGE_ID],
            "old_checkpoint": identity["path"], "new_checkpoint": identity["path"],
            "old_sha256": identity["sha256"], "new_sha256": identity["sha256"]}


def prepare(args):
    if args.cpu_threads < 1:
        raise ValueError("--cpu-threads must be positive")
    torch.set_num_threads(args.cpu_threads)
    source = load_json(args.source_json)
    validate_source(source)
    source_identity = file_identity(args.source_json)
    annotation = args.annotations or source["sources"]["annotations"]["path"]
    annotation_identity = file_identity(annotation)
    if annotation_identity["sha256"] != source["protocol"]["annotations_sha256"]:
        raise ValueError("Annotations changed since endpoint trace")
    dataset = load_json(annotation)
    if ([g for g in dataset["annotations"] if g["id"] == GT_ID] != [source["gt"]]
            or [im for im in dataset["images"] if im["id"] == IMAGE_ID] != [source["image"]]
            or sum(g["category_id"] == 920 for g in dataset["annotations"]) != 1):
        raise ValueError("Unique target GT/image differs from endpoint trace")
    output = Path(args.output_dir).resolve()
    directory, present, missing = checkpoint_plan(source, args.checkpoint_dir)
    protected = [Path(args.source_json).resolve(), Path(annotation).resolve(), directory]
    protected += [Path(source["endpoints"][s]["cache"]["manifest"]["path"]).resolve().parent
                  for s in ("old", "new")]
    if any(output == p or output in p.parents or (p.is_dir() and p in output.parents) for p in protected):
        raise ValueError("Use a separate output directory; never overwrite source caches/checkpoints")
    if output.exists() and any(output.iterdir()) and not (output / "manifest.json").is_file():
        raise ValueError("Nonempty output without our manifest; use a new directory and tee to a sibling .log")

    # Verify BOTH endpoints and EVERY reusable middle cache BEFORE any GPU call.
    categories = {c["id"]: c for c in dataset["categories"]}
    stages = {}
    for side, epoch in (("old", 8), ("new", 12)):
        print(f"[endpoint cache] {epoch}ep; CPU only, no recapture", flush=True)
        stages[epoch] = endpoint_analysis(source, side, categories)
    found = {}
    for epoch, identity in present.items():
        own = output / f"ep{epoch:02d}" / "pairing_cache"
        manifests = candidate_manifests(args, own)
        found[epoch] = find_stage_cache(manifests, source["protocol"], identity)
        if found[epoch]:
            print(f"[reuse] {epoch}ep: {found[epoch][0]}", flush=True)
            stages[epoch] = cached_stage(found[epoch], epoch, dataset, source)
    pending = [e for e in present if e not in stages]
    if args.cache_only and pending:
        raise FileNotFoundError(f"--cache-only: missing middle caches {pending}; NO forward started")
    config_identity = file_identity(args.config_file) if Path(args.config_file).is_file() else None
    if pending:
        if not config_identity or model_code_hash(args.config_file) != source["protocol"]["code_sha256"]:
            raise ValueError("Model/config code differs from source; NO mixed-protocol forward allowed")
        asset_args = SimpleNamespace(phase="dump", config_file=args.config_file)
        if input_asset_hashes(asset_args, output) != source["protocol"]["asset_sha256"]:
            raise ValueError("Model/text assets differ from source; NO forward started")
        if not args.prepare_only and args.device.startswith("cuda") and not torch.cuda.is_available():
            raise ValueError("Single-image forwards require working CUDA/Detrex in the lami interpreter")
    inputs = {"schema_version": 1, "source": source_identity, "annotations": annotation_identity,
              "checkpoints": present, "missing_checkpoints": missing, "config": config_identity,
              "analysis_code": {name: file_identity(ROOT / "tools" / name)["sha256"] for name in (
                  "trace_rare_gt_timeline.py", "rare_gt_timeline_ops.py", "trace_rare_gt_queries.py",
                  "analyze_rare_query_readout.py", "rare_gt_query_ops.py", "rare_region_pairing_ops.py")}}
    # JSON round-trip keeps manifest integer epoch keys stable on resume.
    import json
    signature = locked_manifest(output / "manifest.json", json.loads(json.dumps(inputs)))
    print(f"[budget] image={IMAGE_ID}, GT={GT_ID}; cached epochs={sorted(stages)}; "
          f"missing checkpoint epochs={sorted(missing)}; planned new forwards={len(pending)} <= 3; "
          "training updates=0; no full-validation inference", flush=True)
    return SimpleNamespace(source=source, dataset=dataset, annotation=annotation, output=output,
                           present=present, missing=missing, stages=stages, pending=pending,
                           inputs=inputs, signature=signature)


def display(report):
    for key, timeline in report["timeline"].items():
        print(f"\n=== Single-GT native timeline @IoU={key} ===", flush=True)
        print("epoch   iter  eligible kept    IoU      det-p     CLIP-p det/CLIP-rank    fused   cutoff  ratio   pair-rank   state")
        for row in timeline["rows"]:
            best = row["best_fused_true_class_eligible"]
            if best:
                detail = (f"{best['region_iou']:.4f} {best['detector_probability']:10.6f} "
                          f"{best['clip_probability']:10.6f} {best['detector_rank']:4}/{best['clip_rank']:<4} "
                          f"{best['fused_score']:9.6f} {row['image_cutoff']:8.6f} "
                          f"{best['score_threshold_ratio']:6.3f} {str(best['image_pair_rank_interval']):>12}")
            else:
                detail = f"no eligible final query; maximum IoU={row['maximum_query_iou']:.4f}"
            print(f"{row['epoch']:5} {row['iteration']:6} {row['eligible_queries']:9} "
                  f"{row['retained_true_class_queries']:4} {detail} {row['state']}"
                  + (" [boundary-sensitive]" if row['boundary_sensitive'] else ""))
        print("ALL observed loss/recovery brackets:", timeline["transitions"])
        print("Last observed loss bracket before final miss:", timeline["last_observed_loss_bracket_before_final_miss"])
        print("Resolution:", timeline["resolution"])
    print("Scope: observed checkpoint selection changes only; query IDs are local. "
          "No monotonicity, exact loss iteration, official AP, or training-source claim.", flush=True)


def run(args):
    ctx = prepare(args)
    report = {"complete": False, "fingerprint": ctx.signature, "inputs": ctx.inputs,
              "gt": ctx.source["gt"], "image": ctx.source["image"], "protocol": ctx.source["protocol"],
              "planned_new_forward_calls": len(ctx.pending), "capture_calls_this_invocation": 0,
              "new_forward_calls": 0, "training_updates": 0, "missing_checkpoints": ctx.missing,
              "stages": ctx.stages,
              "limits": ["Only scarecrow GT137708, image218917; not a cross-class AP diagnosis.",
                         "Use all 900 native final queries; highest true-class fused score among eligible boxes.",
                         "Every checkpoint constructs its own native queries, boxes and CLIP ROIs; no bank swaps.",
                         "Query IDs cannot be matched across checkpoints; GT is the geometric anchor.",
                         "Retention is candidate coverage, not official one-to-one matching or AP.",
                         "Missing checkpoints widen brackets; no training or automatic endpoint recapture.",
                         "All observed losses AND recoveries are listed; no monotonicity or persistence assumption.",
                         "A bracket identifies when selection differs at sampled endpoints, not which loss/update caused it."]}
    if args.prepare_only:
        save_json(ctx.output / "plan.json", {"fingerprint": ctx.signature, "pending_epochs": ctx.pending,
                                            "missing_checkpoints": ctx.missing, "training_updates": 0})
        print("[prepared] rerun without --prepare-only; no model forward or training started", flush=True)
        return report
    save_json(ctx.output / "report.json", report)
    for epoch in ctx.pending:
        if report["capture_calls_this_invocation"] >= 3:
            raise RuntimeError("Hard forward budget exhausted")
        cache = ctx.output / f"ep{epoch:02d}" / "pairing_cache"
        identity = ctx.present[epoch]
        # Catch replacement of a checkpoint between preflight and capture.
        checked_identity(identity)
        signature = locked_manifest(cache / "manifest.json", capture_inputs(ctx.source, identity))
        dump_args = SimpleNamespace(**PROTOCOL, config_file=args.config_file, annotations=ctx.annotation,
                                    old_checkpoint=identity["path"], new_checkpoint=identity["path"],
                                    device=args.device, query_chunk_size=128, log_interval=1)
        torch.manual_seed(ctx.source["protocol"].get("seed", 42))
        report["capture_calls_this_invocation"] += 1
        save_json(ctx.output / "report.json", report)
        dump_checkpoint(dump_args, "old", {"image_ids": [IMAGE_ID]}, ctx.dataset, signature, cache)
        report["new_forward_calls"] += 1
        ctx.stages[epoch] = cached_stage((cache / "manifest.json", "old"), epoch, ctx.dataset, ctx.source)
        save_json(ctx.output / "report.json", report)
    report["timeline"] = summarize_timeline(ctx.stages, sorted(ctx.missing))
    report["complete"] = True
    save_json(ctx.output / "report.json", report)
    display(report)
    print(f"[save] {ctx.output / 'report.json'}", flush=True)
    return report


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-json", required=True, help="Completed trace_rare_gt_queries.py report, NOT readout report")
    p.add_argument("--output-dir", required=True, help="Separate new directory; same manifest allows cache reuse on retry")
    p.add_argument("--checkpoint-dir", help="Defaults to the source no-radius run; only exact 9/10/11ep filenames checked")
    p.add_argument("--config-file", default="lami_dino/configs/dino_convnext_large_4scale_12ep_lvis_no_radius.py")
    p.add_argument("--annotations", help="Optional relocated identical annotation JSON")
    p.add_argument("--reuse-cache", nargs="*", default=[], help="Explicit existing dense pairing_cache directories")
    p.add_argument("--cache-search-root", help="Bounded */pairing_cache and */cache manifest discovery")
    p.add_argument("--cache-only", action="store_true", help="Fail before any forward if an available checkpoint lacks cache")
    p.add_argument("--prepare-only", action="store_true", help="Preflight identities/budget; do not forward or train")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--cpu-threads", type=int, default=4)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
