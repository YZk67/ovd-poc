#!/usr/bin/env python3
"""One full LVIS evaluation: native no-radius 10ep with only its decoder core from 8ep.

Trusted local checkpoints only. No training, optimizer, resume, sweep, or source
writes. A new output directory is required; failed attempts are kept for inspection.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # Prefer this tools package over Detectron2's.

import torch

from lami_dino.checkpoint_init import load_trusted_torch_file
from tools.compare_rare_pr_reports import file_identity, load_json, save_json
from tools.diagnose_rare_fp_regions import model_code_hash, read_manifest
from tools.query_path_update_ops import canonical_name, canonical_state, key_group
from tools.summarize_eq2_counterfactual import METRICS, read_last_result
from tools.tpa_geometry_audit_ops import extract_shared_tpa


# Already evaluated native 10ep result, not a new baseline run or an AP estimate.
BASELINE = dict(zip(METRICS, (
    41.7624, 53.9108, 44.1375, 31.1347, 51.7525, 57.7654, 42.3031, 39.6927, 43.8346,
)))
PROTOCOL = {"alpha": 0., "beta": .3, "novel_scale": 3., "tpa_tau": .004375,
            "cls_tau": .07, "max_dets": 300}


def verified_identity(expected, label):
    print(f"[identity] {label}: {expected['path']}", flush=True)
    actual = file_identity(expected["path"])
    if actual["sha256"] != expected["sha256"]:
        raise ValueError(f"Input changed since query audit: {label}")
    return actual


def validate_audit(path):
    report = load_json(path)
    if not report.get("complete") or "decoder_core" not in report["inputs"]["variants"]:
        raise ValueError("Need a complete query-path audit including decoder_core")
    inputs = report["inputs"]
    sources = inputs["sources"]
    verified_identity(inputs["stage_report"], "stage report")
    parent_path = Path(inputs["stage_report"]["path"]).parent / "pairing_cache/manifest.json"
    parent = read_manifest(parent_path)
    if parent["fingerprint"] != inputs["parent_fingerprint"]:
        raise ValueError("Query audit and pairing manifest differ")
    protocol = parent["inputs"]
    if any(protocol.get(key) != value for key, value in PROTOCOL.items()):
        raise ValueError("Expected the locked no-radius 8ep/10ep inference protocol")
    for name in ("old_checkpoint", "new_checkpoint", "annotations", "prompt_bank", "config_file"):
        verified_identity(sources[name], name)
    for side in ("old", "new"):
        if protocol[side + "_sha256"] != sources[side + "_checkpoint"]["sha256"]:
            raise ValueError("Native cache and checkpoint identities differ")
    if model_code_hash(sources["config_file"]["path"]) != inputs["model_code_sha256"]:
        raise ValueError("Model/config code changed since the query audit")
    if protocol["code_sha256"] != inputs["model_code_sha256"]:
        raise ValueError("Native cache and query audit model code differ")
    for relative, digest in inputs["audit_code"].items():
        verified_identity({"path": str(ROOT / relative), "sha256": digest}, "audit code")
    for asset, digest in protocol["asset_sha256"].items():
        verified_identity({"path": asset, "sha256": digest}, "model asset")
    # This JSON is the full-validation baseline, not the small diagnostic panel.
    verified_identity(sources["new_predictions"], "native 10ep predictions")
    baseline_log = Path(sources["new_predictions"]["path"]).parent / "log.txt"
    values = read_last_result(baseline_log)
    if any(not math.isclose(v, BASELINE[k], rel_tol=0, abs_tol=.0002)
           for k, v in zip(METRICS, values)):
        raise ValueError("Baseline log does not match the recorded native 10ep metrics")
    return report, file_identity(baseline_log)


def endpoint_state(checkpoint, iteration):
    tpa, info = extract_shared_tpa(checkpoint)
    if (info["iteration"] != iteration or tpa["prototype_queries"].shape != (5, 256)
            or float(tpa["prototype_mode_strength"]) != 0.
            or abs(float(tpa["slot_prior_strength"]) - .2) > 1e-6):
        raise ValueError(f"Expected no-radius K5 slot-prior=.2 checkpoint at iteration {iteration}")
    state = {}
    for key, value in checkpoint["model"].items():
        name = key[7:] if key.startswith("module.") else key
        if name in state:
            raise ValueError(f"Duplicate model key after DDP prefix removal: {name}")
        state[name] = value
    return state


def hybrid_state(old, new, audited_keys):
    """Return a non-mutating state mapping; preserve every non-core value from new."""
    a, _ = canonical_state(old)
    b, _ = canonical_state(new)
    if a.keys() != b.keys():
        raise ValueError("Endpoint state keys differ")
    for key in a:
        if a[key].shape != b[key].shape or a[key].dtype != b[key].dtype:
            raise ValueError(f"Endpoint shape/dtype differs: {key}")
    core = sorted(k for k in b if key_group(k) == "decoder_core")
    if not core or core != sorted(audited_keys):
        raise ValueError("Decoder core keys differ from the completed query audit")
    result = {k: (a[canonical_name(k)] if canonical_name(k) in core else v)
              for k, v in new.items()}
    changed = [k for k in core if not torch.equal(a[k], b[k])]
    if not changed:
        raise ValueError("Decoder core has no endpoint changes")
    return result, {"swapped_canonical_keys": core, "changed_canonical_keys": changed,
                    "preserved_raw_keys": [k for k in new if canonical_name(k) not in core]}


def verify_hybrid(saved, old, new, core):
    if set(saved) != {"model", "decoder_rollback"}:
        raise ValueError("Hybrid must be weights-only, without iteration/optimizer/trainer/scheduler")
    state = saved["model"]
    canonical_state(state)  # Check finite values and shared aliases after serialization.
    if state.keys() != new.keys():
        raise ValueError("Serialized hybrid model keys differ from the 10ep checkpoint")
    a, _ = canonical_state(old)
    for key, value in state.items():
        expected = a[canonical_name(key)] if canonical_name(key) in core else new[key]
        if (value.dtype != expected.dtype or value.shape != expected.shape
                or not torch.equal(value, expected)):
            raise ValueError(f"Hybrid verification failed: {key}")


def evaluation_command(config, checkpoint, output, num_gpus):
    # Do not pass --resume, --ddebug, custom score hooks, GT boxes, or cached CLIP.
    options = [
        f"train.init_checkpoint={json.dumps(str(checkpoint))}",
        f"train.output_dir={json.dumps(str(output))}",
        f"dataloader.evaluator.output_dir={json.dumps(str(output))}",
        "dataloader.test.dataset.names=lvis_v1_val",
        "model.alpha=0.0", "model.beta=0.3", "model.novel_scale=3.0",
        "model.classifier.tpa_num_prototypes=5",
        "model.classifier.tpa_tau=0.004375", "model.classifier.tpa_cls_tau=0.07",
        "model.classifier.tpa_slot_prior_strength=0.2",
        "model.classifier.tpa_prototype_mode_strength=0.0",
        "model.classifier.tpa_eval_legacy_logsumexp=False",
        "model.classifier.tpa_eval_logit_bias=0.0", "model.tpa_eval_mode_scale=1.0",
        "model.soft_category_topk=3", "model.inference_query_class_topk=0",
        "model.select_box_nums_for_evaluation=300", "model.score_ensemble=True",
        "train.device=cuda", "model.device=cuda",
    ]
    return [sys.executable, "-u", str(ROOT / "tools/train_net.py"),
            "--config-file", str(config), "--num-gpus", str(num_gpus), "--eval-only", *options]


def run_evaluation(command, output):
    with (output / "console.log").open("x", encoding="utf-8") as log:
        with subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True, bufsize=1) as process:
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
            status = process.wait()
    if status:
        raise subprocess.CalledProcessError(status, command)


def collect_results(output):
    # train_net can catch evaluation exceptions and exit 0. Never call that a success.
    console = (output / "console.log").read_text(errors="replace")
    if "Skipping evaluation" in console or "Traceback (most recent call last)" in console:
        raise RuntimeError("Evaluation failed/skipped; inspect console.log (no successful summary)")
    metrics = dict(zip(METRICS, read_last_result(output / "log.txt")))
    if any(not math.isfinite(v) or not 0 <= v <= 100 for v in metrics.values()):
        raise ValueError("Invalid LVIS metrics")
    predictions = output / "lvis_instances_results.json"
    if not predictions.is_file() or predictions.stat().st_size <= 2:
        raise ValueError("No nonempty full-validation prediction JSON; evaluation incomplete")
    return {"complete": True, "baseline_native_10ep": BASELINE,
            "hybrid_10ep_decoder8": metrics,
            "delta_hybrid_minus_native10": {k: metrics[k] - BASELINE[k] for k in METRICS},
            "predictions": file_identity(predictions),
            "scope": "Full native LVIS inference. One post-hoc intervention, not loss attribution, "
                     "training replay, or proof that freezing the decoder would improve training."}


def run(args):
    if args.num_gpus < 1 or args.cpu_threads < 1:
        raise ValueError("Positive GPU count and CPU threads required")
    output = Path(args.output_dir).expanduser().resolve()
    if output.exists():
        raise ValueError("Use a NEW output directory; never overwrite checkpoints or stale evaluation logs")
    torch.set_num_threads(args.cpu_threads)
    report, baseline_log = validate_audit(args.audit_report)
    sources = report["inputs"]["sources"]
    states = {}
    for side, iteration in (("old", 56799), ("new", 70999)):
        print(f"[load CPU] {side}: iteration={iteration}", flush=True)
        ckpt = load_trusted_torch_file(sources[side + "_checkpoint"]["path"])
        states[side] = endpoint_state(ckpt, iteration)
        del ckpt
    state, checks = hybrid_state(states["old"], states["new"], report["inventory"]["decoder_core"]["keys"])
    output.mkdir(parents=True, exist_ok=False)  # Exclusive claim; originals remain read-only.
    checkpoint = output / "hybrid_10ep_decoder8_eval_only.pth"
    provenance = {"eval_only": True, "base_iteration": 70999, "decoder_core_iteration": 56799,
                  "sources": {s: sources[s + "_checkpoint"] for s in ("old", "new")}, **checks}
    with checkpoint.open("xb") as handle:
        torch.save({"model": state, "decoder_rollback": provenance}, handle)
    saved = load_trusted_torch_file(checkpoint)
    verify_hybrid(saved, states["old"], states["new"], checks["swapped_canonical_keys"])
    print(f"[verified] {len(checks['swapped_canonical_keys'])} decoder-core tensors from 8ep; "
          f"{len(checks['preserved_raw_keys'])} remaining tensors unchanged from 10ep", flush=True)
    # Free both endpoints and serialization validation before launching the GPU subprocess.
    del states, state, saved
    gc.collect()
    command = evaluation_command(sources["config_file"]["path"], checkpoint, output, args.num_gpus)
    manifest = {"audit_report": file_identity(args.audit_report), "baseline_log": baseline_log,
                "hybrid_checkpoint": file_identity(checkpoint), "provenance": provenance,
                "protocol": PROTOCOL, "command": command,
                "runner": file_identity(__file__), "training_updates": 0}
    save_json(output / "manifest.json", manifest)
    if args.prepare_only:
        print(f"[prepared only] {checkpoint}; evaluation has NOT run", flush=True)
        return manifest
    print("[evaluate] ONE full LVIS evaluation, native boxes / CLIP ROI / top-300", flush=True)
    run_evaluation(command, output)
    result = collect_results(output)
    result["manifest"] = file_identity(output / "manifest.json")
    save_json(output / "summary.json", result)
    print("\n=== Native 10ep vs 10ep with 8ep decoder_core ===")
    print(f"{'variant':>24} " + " ".join(f"{k:>8}" for k in METRICS))
    for name in ("baseline_native_10ep", "hybrid_10ep_decoder8", "delta_hybrid_minus_native10"):
        print(f"{name:>24} " + " ".join(f"{result[name][k]:8.4f}" for k in METRICS))
    print(f"[save] {output / 'summary.json'}", flush=True)
    return result


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-report", required=True, help="Complete audit_query_path_updates.py report")
    parser.add_argument("--output-dir", required=True, help="NEW directory; do not pre-create it or tee into it")
    parser.add_argument("--num-gpus", type=int, default=4)
    parser.add_argument("--cpu-threads", type=int, default=2)
    parser.add_argument("--prepare-only", action="store_true", help="CPU build/verify only, no inference")
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
