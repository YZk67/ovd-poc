#!/usr/bin/env python3
"""Split 8ep decoder classification gradients on the PREVIOUS audit's samples.

One GPU; at most 128 training-image exposures and 16 validation forwards.
No training, optimizer, new controls, AP evaluation, or 10ep model forward.
Requires the completed parent loss-source report AND its local capture caches.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import math
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from tools import audit_decoder_loss_sources as base
from tools.compare_rare_pr_reports import file_identity, load_json, save_json
from tools.decoder_loss_audit_ops import (
    CLASSIFICATION_PARTS, DETAIL_GROUPS, GROUPS, split_classification_losses, summarize_probes,
)
from tools.diagnose_rare_fp_regions import fingerprint
from tools.evaluate_decoder_rollback import validate_audit, verified_identity


def close(a, b, label, *, rtol=2e-3, atol=2e-4):
    if a is None or b is None:
        if a is not b:
            raise ValueError(f"Replay missing/connected values differ: {label}")
        return
    if not math.isfinite(a) or not math.isfinite(b) or not math.isclose(a, b, rel_tol=rtol, abs_tol=atol):
        raise ValueError(f"Replay differs from parent: {label}: {a} vs {b}")


def verify_micro(current, parent):
    for key in ("mapped_inputs", "fedloss_category_indices"):
        if current[key] != parent[key]:
            raise ValueError(f"Parent training replay changed {key}")
    for group in GROUPS:
        if current["loss_keys"][group] != parent["loss_keys"][group]:
            raise ValueError(f"Native loss grouping changed: {group}")
    if current["weighted_losses"].keys() != parent["weighted_losses"].keys():
        raise ValueError("Native weighted loss keys changed")
    for key, value in current["weighted_losses"].items():
        close(value, parent["weighted_losses"][key], key, rtol=2e-4, atol=2e-5)


def verify_window(current, prior):
    """Compare full averaged vectors, not only their norm or probe projection."""
    if (current["side"] != "old" or prior["side"] != "old" or current["window"] != prior["window"]
            or current["train_annotations"] != prior["train_annotations"]):
        raise ValueError("Parent gradient cache endpoint/window/annotations differ")
    result = {}
    for group in GROUPS:
        a, b = current["gradients"][group], prior["gradients"][group]
        if a is None or b is None:
            if a is not b:
                raise ValueError(f"Parent gradient connection differs: {group}")
            result[group] = {"connected": False}
            continue
        if a.shape != b.shape or a.dtype != b.dtype or not torch.isfinite(a).all() or not torch.isfinite(b).all():
            raise ValueError(f"Parent gradient vector layout/finiteness differs: {group}")
        error = (a.double() - b.double()).norm().item()
        relative = error / max(b.double().norm().item(), 1e-12)
        if relative > 5e-4 or not torch.allclose(a, b, rtol=5e-4, atol=2e-6):
            raise ValueError(f"Parent full gradient vector replay differs: {group}")
        result[group] = {"connected": True, "relative_l2_error": relative,
                         "max_abs_error": (a-b).abs().max().item()}
    return result


def validate_parent(path):
    parent = load_json(path)
    if (not parent.get("complete") or not parent.get("paired_training_inputs_verified")
            or parent.get("training_updates") != 0 or parent.get("optimizer_created") is not False):
        raise ValueError("Need a complete read-only paired decoder loss-source audit")
    inputs = parent["inputs"]
    if fingerprint(inputs) != parent["fingerprint"]:
        raise ValueError("Parent report fingerprint differs from its inputs")
    if str(torch.__version__) != inputs["torch_version"]:
        raise ValueError("Use the same PyTorch environment as the parent audit")
    # These two helpers were extended only to support the split audit. Every
    # model/training/config/data helper must remain identical to the parent.
    extended = {"tools/audit_decoder_loss_sources.py", "tools/decoder_loss_audit_ops.py"}
    for relative, digest in inputs["code"].items():
        target = (ROOT / relative).resolve()
        if ROOT not in target.parents:
            raise ValueError("Invalid code identity path")
        if relative not in extended and file_identity(target)["sha256"] != digest:
            raise ValueError(f"Model/training code differs from parent: {relative}")
    if not extended.issubset(inputs["code"]):
        raise ValueError("Parent lacks expected audit code identities")
    verified_identity(inputs["query_audit"], "parent query audit")
    query, _ = validate_audit(inputs["query_audit"]["path"])
    if query["inputs"]["sources"] != inputs["sources"]:
        raise ValueError("Parent and query audit source identities differ")
    captures = {}
    directory = Path(path).resolve().parent
    for side in ("old", "new"):
        identity = parent["captures"][side]
        if Path(identity["path"]).resolve() != directory / f"{side}_capture.json":
            raise ValueError("Parent capture directory differs; retain the original audit directory")
        verified_identity(identity, f"parent {side} capture")
        capture = base.read_locked(identity["path"], parent["fingerprint"])
        if (capture["side"] != side or capture["iteration"] != (56799 if side == "old" else 70999)
                or not capture["weights_unchanged"] or capture["training_updates"] != 0
                or capture["optimizer_created"] is not False):
            raise ValueError("Parent capture is not the expected read-only endpoint")
        captures[side] = capture
    # Reuse original 10ep anchor labels/boxes, with no outcome-based resampling.
    for control in inputs["controls"]:
        image_id = control["image_id"]
        value = base.read_locked(directory / "new" / f"validation_{image_id}.json", parent["fingerprint"])
        expected = [p for p in captures["new"]["probes"] if p["image_id"] == image_id]
        if value["side"] != "new" or value["image_id"] != image_id or value["probes"] != expected:
            raise ValueError("Parent control cache differs from certified capture")
    return parent, query, captures["old"]


def prepare(args):
    parent, query, reference = validate_parent(args.parent_audit)
    source = parent["inputs"]
    for key in ("windows", "microbatches", "batch_size", "seed", "device"):
        setattr(args, key, source[key])
    args.controls_per_split = max(sum(c["panel"] == split for c in source["controls"])
                                  for split in ("control_rare", "control_base"))
    budget = base.check_budget(args) // 2
    if budget > 128 or len(source["focus_images"]) + len(source["controls"]) > 16:
        raise ValueError("Hard 8ep budget exceeded (128 train images / 16 validation forwards)")
    if len(reference["windows"]) != args.windows or any(
            len(w["microbatches"]) != args.microbatches for w in reference["windows"]):
        raise ValueError("Parent capture window/microbatch count differs from budget")
    directory = Path(args.parent_audit).resolve().parent
    stage = Path(query["inputs"]["stage_report"]["path"]).parent
    output = Path(args.output_dir).resolve()
    protected = [directory, stage, Path(source["query_audit"]["path"]).resolve(),
                 *[Path(v["path"]).resolve() for v in source["sources"].values()]]
    if any(output == p or output in p.parents or p in output.parents for p in protected):
        raise ValueError("Use a separate output directory; parent caches and sources are read-only")
    code = {relative: file_identity(ROOT / relative)["sha256"] for relative in source["code"]}
    code[str(Path(__file__).resolve().relative_to(ROOT))] = file_identity(__file__)["sha256"]
    identity = {"parent_audit": file_identity(args.parent_audit), "parent_fingerprint": parent["fingerprint"],
                "control_anchor_files": [file_identity(directory / "new" / f"validation_{c['image_id']}.json")
                                         for c in source["controls"]],
                "parent_gradient_files": [file_identity(directory / "old" / f"training_window_{w}.pt")
                                          for w in range(args.windows)],
                "sources": source["sources"], "side": "old", "iteration": 56799,
                "windows": args.windows, "microbatches": args.microbatches, "batch_size": args.batch_size,
                "seed": args.seed, "device": args.device, "training_image_exposures": budget,
                "focus_images": source["focus_images"], "controls": source["controls"],
                "groups": list(DETAIL_GROUPS), "torch_version": str(torch.__version__), "code": code}
    signature = fingerprint(identity)
    manifest = output / "manifest.json"
    if manifest.exists():
        if load_json(manifest) != {"inputs": identity, "fingerprint": signature}:
            raise ValueError("Split audit inputs/code changed; use a new output directory")
    else:
        if args.analyze_only:
            raise ValueError("--analyze-only needs completed split captures; no GPU fallback")
        if output.exists() and any(output.iterdir()):
            raise ValueError("Refusing nonempty output without split audit manifest")
        save_json(manifest, {"inputs": identity, "fingerprint": signature})
    print(f"[budget] 8ep ONLY: {budget} training-image exposures, "
          f"{len(source['focus_images']) + len(source['controls'])} validation forwards maximum; no updates", flush=True)
    return SimpleNamespace(parent=parent, report=query, reference=reference, sources=source["sources"],
        stage=stage, output=output, dataset=load_json(source["sources"]["annotations"]["path"]),
        records={"old": [{**r, "panel": "focus", "frequency": "r"}
                          for r in query["interventions"]["old"]["native"]["regions"]]},
        controls=source["controls"], signature=signature, identity=identity,
        loss_splitter=split_classification_losses, training_reference=reference["windows"],
        verify_micro=verify_micro, control_reference_dir=directory,
        verify_window=lambda current, w: verify_window(current, base.read_locked(
            directory / "old" / f"training_window_{w}.pt", parent["fingerprint"], tensor=True)),
        control_reference_signature=parent["fingerprint"])


def verify_replay(ctx, capture):
    reference = ctx.reference
    if (capture["side"] != "old" or capture["iteration"] != 56799 or not capture["weights_unchanged"]
            or capture["training_updates"] != 0 or capture["optimizer_created"] is not False
            or capture["parameters"] != reference["parameters"]):
        raise ValueError("Split capture is not the unchanged 8ep decoder")
    if len(capture["windows"]) != len(reference["windows"]):
        raise ValueError("Split capture has incomplete windows")
    for current, prior in zip(capture["windows"], reference["windows"]):
        if current["train_annotations"] != prior["train_annotations"]:
            raise ValueError("Parent training annotation replay differs")
        if len(current["microbatches"]) != len(prior["microbatches"]):
            raise ValueError("Split capture has incomplete microbatches")
        for a, b in zip(current["microbatches"], prior["microbatches"]):
            verify_micro(a, b)
        for group in GROUPS:
            a, b = current["summary"][group], prior["summary"][group]
            if a["connected"] != b["connected"]:
                raise ValueError("Parent gradient connections differ")
            close(a["norm"], b["norm"], f"{group} gradient norm")
        error = current["classification_reconstruction"]
        if not math.isfinite(error["relative_l2_error"]) or error["relative_l2_error"] > 5e-4:
            raise ValueError("Classification gradient reconstruction failed")
        for group in GROUPS:
            check = current["parent_gradient_replay"][group]
            if check["connected"] != prior["summary"][group]["connected"]:
                raise ValueError("Parent full gradient replay connection differs")
            if check["connected"] and (not math.isfinite(check["relative_l2_error"])
                                       or check["relative_l2_error"] > 5e-4):
                raise ValueError("Parent full gradient replay check failed")
    if capture["control_coverage"] != reference["control_coverage"]:
        raise ValueError("Control coverage differs from parent; no replacement controls allowed")
    if len(capture["probes"]) != len(reference["probes"]):
        raise ValueError("Validation probe count differs from parent")
    max_raw_error = 0.
    for a, b in zip(capture["probes"], reference["probes"]):
        for key in ("panel", "category", "kind", "image_id", "count", "expected_count", "matches"):
            if a[key] != b[key]:
                raise ValueError(f"Validation probe selection changed: {key}")
        if len(a["effects"]) != len(b["effects"]):
            raise ValueError("Validation effect windows differ from parent")
        for effects, previous in zip(a["effects"], b["effects"]):
            for group in GROUPS:
                for metric in ("raw_descent_derivative", "unit_descent_derivative"):
                    close(effects[group][metric], previous[group][metric], f"{a['category']}/{group}/{metric}")
                max_raw_error = max(max_raw_error, abs(effects[group]["raw_descent_derivative"]
                                                       - previous[group]["raw_descent_derivative"]))
            total = effects["classification"]["raw_descent_derivative"]
            parts = sum(effects[k]["raw_descent_derivative"] for k in CLASSIFICATION_PARTS)
            close(total, parts, "raw classification derivative reconstruction")
    return {"verified": True, "max_parent_raw_derivative_abs_error": max_raw_error,
            "full_gradient_replay": [w["parent_gradient_replay"] for w in capture["windows"]],
            "gradient_reconstruction": [w["classification_reconstruction"] for w in capture["windows"]]}


def analyze(ctx, capture):
    replay = verify_replay(ctx, capture)
    rows = summarize_probes(capture["probes"], len(capture["windows"]), groups=DETAIL_GROUPS)
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["panel"], row["category"], row["loss_group"]].append(row)
    signs = []
    print("\n=== 8ep classification split: conditional unit-descent TP-FP derivatives ===")
    for (panel, category, source), entries in sorted(grouped.items()):
        values = [r["unit_descent_derivative"]["tp_minus_fp"] for r in entries]
        present = [v for v in values if v is not None]
        signs.append({"panel": panel, "category": category, "source": source,
                      "negative_windows": sum(v < 0 for v in present), "valid_windows": len(present),
                      "unit_values": values})
        print(f"{panel:12} {category:20} {source:15} " +
              " ".join("NA" if v is None else f"{v:+.6g}" for v in values), flush=True)
    report = {"complete": True, "fingerprint": ctx.signature, "inputs": ctx.identity,
              "capture": file_identity(ctx.output / "old_capture.json"), "parent_replay": replay,
              "gradient_summaries": [w["summary"] for w in capture["windows"]],
              "loss_keys": capture["windows"][0]["microbatches"][0]["loss_keys"],
              "local_margin_derivatives": rows, "window_signs": signs,
              "control_coverage": capture["control_coverage"],
              "training_updates": 0, "optimizer_created": False,
              "scope": ctx.parent["scope"] + [
                  "Only 8ep was recaptured; exact parent samples/labels/augmentations/FedLoss are replay-checked.",
                  "Classification total overlaps its children: do NOT add total to final/aux/DN/encoder.",
                  "Raw derivatives add; unit-normalized derivatives have different norms and do NOT add.",
                  "Encoder classification is recorded separately and must have no direct decoder-core path.",
                  "This report does not select a loss to remove or start training; four windows are not significance tests."]}
    save_json(ctx.output / "report.json", report)
    print(f"[save] {ctx.output / 'report.json'}; no optimizer or updates", flush=True)
    return report


def run(args):
    if args.cpu_threads < 1:
        raise ValueError("cpu-threads must be positive")
    torch.set_num_threads(args.cpu_threads)
    ctx = prepare(args)
    if args.prepare_only:
        return ctx.identity
    path = ctx.output / "old_capture.json"
    if args.analyze_only and not path.exists():
        raise ValueError("Missing split capture; --analyze-only never falls back to GPU")
    capture = (base.read_locked(path, ctx.signature) if path.exists()
               else base.capture_side(args, ctx, "old", None))
    return analyze(ctx, capture)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-audit", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--cpu-threads", type=int, default=2)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--prepare-only", action="store_true")
    mode.add_argument("--analyze-only", action="store_true")
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
