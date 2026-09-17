#!/usr/bin/env python3
"""Paired 8ep/12ep, detached-feature Eq.2 training audit. No parameter updates.

Trusted checkpoints/caches only. Defaults to 4 batches x 2 images per endpoint.
Native training capture needs one GPU; --analyze-only never constructs a model.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from lami_dino.checkpoint_init import load_trusted_torch_file
from tools.audit_tpa_gradients import save_gradients
from tools.capture_tpa_gradients import state_versions
from tools.compare_rare_pr_reports import file_identity, load_json, save_json
from tools.decoder_loss_audit_ops import isolated_rng
from tools.eq2_training_audit_ops import (
    VARIANTS, assemble_records, assignment_signature, audit_branch,
    capture_head_inputs, compare_stages, detached_matcher, matcher_settings,
)
from tools.evaluate_decoder_rollback import endpoint_state
from tools.tpa_geometry_audit_ops import tensor_digest


STAGES = {"8ep": 56799, "12ep": 85199}
SCOPE = (
    "Formula-only, fixed native features/prototype outputs/boxes on sampled train_norare batches. "
    "Negative means training target=0, not an officially evaluated false positive. "
    "Gradients are partial derivatives to classifier inputs and raw prototype outputs, "
    "NOT TPA/decoder parameter gradients or AdamW/clipped updates. No rare/validation supervision, "
    "no training, no AP evaluation. Stage interactions are descriptive, not causal attribution "
    "of the 8ep-to-12ep APr change or a reconstruction of the old checkpoint's training."
)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def check_budget(args):
    if (not 1 <= args.batches <= 8 or not 1 <= args.batch_size <= 4
            or not 0 <= args.seed < 2**31 or args.cpu_threads < 1):
        raise ValueError("Budget: 1..8 batches, 1..4 images/batch, nonnegative seed, positive CPU threads")
    return 2 * args.batches * args.batch_size


def prepare(args):
    budget = check_budget(args)
    output = Path(args.output_dir).resolve()
    sources = {s: Path(getattr(args, "checkpoint_" + s)).resolve() for s in STAGES}
    config = Path(args.config_file).resolve()
    for source in (*sources.values(), config):
        if output == source or output in source.parents or output == source.parent:
            raise ValueError("Use a separate output directory, not a checkpoint/config directory")
    print(f"[budget] <= {budget} image exposures total, no parameter updates", flush=True)
    code = set()
    for directory in ("lami_dino", "configs/common", "detrex/modeling", "detrex/layers",
                      "detrex/data", "detectron2/detectron2/data", "detectron2/detectron2/config"):
        code.update((ROOT / directory).rglob("*.py"))
    for filename in ("audit_eq2_training_stages.py", "eq2_training_audit_ops.py", "audit_tpa_gradients.py",
                     "capture_tpa_gradients.py", "decoder_loss_audit_ops.py", "evaluate_decoder_rollback.py",
                     "tpa_geometry_audit_ops.py", "compare_rare_pr_reports.py"):
        code.add(ROOT / "tools" / filename)
    inputs = {"checkpoints": {}, "config": file_identity(config), "batches": args.batches,
              "batch_size": args.batch_size, "seed": args.seed, "image_exposures": budget,
              "capture_device": args.device, "torch_version": str(torch.__version__),
              "code": {str(p.relative_to(ROOT)): file_identity(p)["sha256"] for p in sorted(code)}}
    for side, source in sources.items():
        print(f"[identity] {side}: {source}", flush=True)
        inputs["checkpoints"][side] = file_identity(source)
    manifest = {"inputs": inputs, "fingerprint": digest(inputs)}
    path = output / "manifest.json"
    if path.exists():
        if load_json(path) != manifest:
            raise ValueError("Inputs/code/budget changed; use a new output directory")
    else:
        if args.analyze_only:
            raise ValueError("--analyze-only requires an existing capture; no GPU fallback")
        if output.exists() and any(output.iterdir()):
            raise ValueError("Nonempty output directory has no matching manifest")
        # Validate both endpoints before any GPU model is constructed.
        for side, source in sources.items():
            ckpt = load_trusted_torch_file(source)
            endpoint_state(ckpt, STAGES[side])
            del ckpt
        save_json(path, manifest)
    return output, manifest


def verified_file(identity):
    if file_identity(identity["path"]) != identity:
        raise ValueError("Changed input/cache: " + identity["path"])


def load_capture(output, side, signature):
    path = output / (side + "_capture.json")
    if not path.exists():
        return None
    capture = load_json(path)
    if (capture.get("fingerprint") != signature or capture.get("side") != side
            or capture.get("iteration") != STAGES[side] or not capture.get("weights_unchanged")):
        raise ValueError("Invalid endpoint receipt")
    for source in capture["assets"] + [capture["train_annotations"]]:
        verified_file(source)
    for row in capture["batches"]:
        verified_file(row["cache"])
    return capture


def paired_metadata(reference, candidate):
    for key in ("mapped_inputs", "category_indices", "sampled_rare_count", "branch_layout", "rng_after_forward"):
        if reference[key] != candidate[key]:
            raise ValueError("Endpoints are not paired: " + key)


def verify_pair(captures, expected_batches):
    a, b = (captures[s] for s in STAGES)
    for key in ("assets", "train_annotations", "matcher", "classifier", "native_training_protocol"):
        if a[key] != b[key]:
            raise ValueError("Stage protocol mismatch: " + key)
    if len(a["batches"]) != expected_batches or len(b["batches"]) != expected_batches:
        raise ValueError("Incomplete paired capture")
    for x, y in zip(a["batches"], b["batches"]):
        paired_metadata(x, y)


def capture_stage(args, output, manifest, side, reference=None):
    if not args.device.startswith("cuda") or not torch.cuda.is_available() or torch.distributed.is_initialized():
        raise ValueError("Native capture requires one CUDA process in the lami environment, not DDP")
    from detectron2.config import LazyConfig, instantiate
    from detectron2.data import MetadataCatalog
    from detectron2.utils.events import EventStorage

    cfg = LazyConfig.load(manifest["inputs"]["config"]["path"])
    cfg.model.device = args.device
    if (cfg.dataloader.train.dataset.names != "lvis_v1_train_norare"
            or cfg.model.get("teacher_rpsa", False) or not cfg.model.use_fed_loss):
        raise ValueError("Require train_norare/FedLoss without teacher RPSA")
    asset_paths = {str(Path(cfg.model[k]).resolve()) for k in (
        "query_path", "eval_query_path", "vlm_query_path", "clip_head_path",
        "seen_classes", "all_classes", "cat_freq_path") if cfg.model.get(k)}
    asset_paths.update(str(Path(cfg.model.classifier[k]).resolve()) for k in (
        "zs_weight_path", "eval_zs_weight_path", "text_embed_path", "eval_text_embed_path")
        if cfg.model.classifier.get(k))
    assets = [file_identity(p) for p in sorted(asset_paths)]
    protocol = {k: cfg.train.get(k) for k in ("gradient_accumulation_steps", "lr_scheduler_max_iter",
                 "separate_tpa_grad_clip", "tpa_conflict_projection")}
    cfg.dataloader.train.total_batch_size = args.batch_size
    cfg.dataloader.train.num_workers = 0
    device_index = torch.device(args.device).index or 0
    torch.cuda.set_device(device_index)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    with isolated_rng(args.seed, device_index):
        model = instantiate(cfg.model)
    verified_file(manifest["inputs"]["checkpoints"][side])
    ckpt = load_trusted_torch_file(manifest["inputs"]["checkpoints"][side]["path"])
    model.load_state_dict(endpoint_state(ckpt, STAGES[side]), strict=True)
    del ckpt
    model.to(args.device).train()
    for param in model.parameters():
        param.requires_grad_(False)
    model.tpa_advance_step = False
    if any(isinstance(m, torch.nn.modules.batchnorm._BatchNorm) and m.training for m in model.modules()):
        raise ValueError("Training BatchNorm is not supported by this single-GPU audit")
    versions = state_versions(model)
    settings = matcher_settings(model.criterion.matcher)
    classifier = [{"scale": float(h.norm_temperature), "tau": float(h.tpa_cls_tau),
                   "norm_weight": bool(h.norm_weight), "use_bias": bool(h.use_bias)} for h in model.class_embed]
    if any(not h["norm_weight"] or h["use_bias"] or h["tau"] != .07 or h["scale"] != 50. for h in classifier):
        raise ValueError("Expected normalized scale=50, cls_tau=.07, bias-free classifier")
    rows = []
    with isolated_rng(args.seed, device_index), EventStorage(start_iter=STAGES[side]+1):
        loader = iter(instantiate(cfg.dataloader.train))
        annotation = file_identity(MetadataCatalog.get(cfg.dataloader.train.dataset.names).json_file)
        if reference:
            for key, value in (("assets", assets), ("train_annotations", annotation), ("matcher", settings),
                               ("classifier", classifier), ("native_training_protocol", protocol)):
                if reference[key] != value:
                    raise ValueError("Stage protocol mismatch before native forward: " + key)
        for batch in range(args.batches):
            data = next(loader)
            if len(data) != args.batch_size:
                raise ValueError("Unexpected training batch size")
            mapped = []
            for d in data:
                classes = d["instances"].gt_classes
                if any(c < 0 or c >= len(model.novel_idx) or bool(model.novel_idx[c]) for c in classes.tolist()):
                    raise ValueError("Rare/invalid GT in training audit")
                mapped.append({"image_id": int(d["image_id"]), "shape": list(d["image"].shape),
                               "image": tensor_digest(d["image"]), "classes": tensor_digest(classes),
                               "boxes": tensor_digest(d["instances"].gt_boxes.tensor)})
            if reference and mapped != reference["batches"][batch]["mapped_inputs"]:
                raise ValueError("Mapped data/augmentation differ; stopping before forward")
            with isolated_rng(args.seed+10000+batch, device_index), torch.no_grad(), capture_head_inputs(model) as captured:
                losses = model(data)
                records = assemble_records(model, captured, losses)
                sampled = captured[2]["category_indices"]
                rng = [tensor_digest(torch.get_rng_state()), tensor_digest(torch.cuda.get_rng_state(device_index))]
            if state_versions(model) != versions or any(p.grad is not None for p in model.parameters()):
                raise ValueError("Model weights/buffers changed or parameter gradients populated")
            row = {"batch": batch, "mapped_inputs": mapped, "category_indices": sampled,
                   "sampled_rare_count": int(model.novel_idx[sampled].sum()), "rng_after_forward": rng,
                   "branch_layout": [{"name": r["name"], "shape": list(r["logits"].shape),
                                      "normalizer": r["normalizer"], "weight": r["weight"],
                                      "alpha": r["alpha"], "gamma": r["gamma"],
                                      "DN_assignment": None if r["hungarian"] else assignment_signature(r["indices"])}
                                     for r in records]}
            if reference:
                paired_metadata(reference["batches"][batch], row)
            path = output / side / f"batch_{batch}.pt"
            save_gradients(path, {"fingerprint": manifest["fingerprint"], "side": side,
                                  "metadata": row, "records": records})
            rows.append({**row, "cache": file_identity(path)})
            print(f"[capture] {side} batch={batch+1}/{args.batches} images={[d['image_id'] for d in mapped]} "
                  f"classes={len(sampled)} heads={len(records)}", flush=True)
            del data, losses, records, captured
        del loader
    receipt = {"fingerprint": manifest["fingerprint"], "side": side, "iteration": STAGES[side],
               "assets": assets, "train_annotations": annotation, "matcher": settings, "classifier": classifier,
               "native_training_protocol": protocol, "batches": rows, "weights_unchanged": True,
               "optimizer_created": False, "training_updates": 0, "capture_precision": "float32, no AMP/TF32"}
    save_json(output / (side+"_capture.json"), receipt)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return receipt


def analyze(args, output, manifest, captures):
    verify_pair(captures, args.batches)
    device = args.analysis_device or ("cpu" if args.analyze_only else args.device)
    stages = {}
    for side in STAGES:
        stages[side] = []
        matcher = detached_matcher(captures[side]["matcher"])
        for row in captures[side]["batches"]:
            cache = load_trusted_torch_file(row["cache"]["path"])
            if cache.get("fingerprint") != manifest["fingerprint"] or cache.get("side") != side:
                raise ValueError("Stale/wrong-side feature cache")
            paired_metadata(row, cache["metadata"])
            branches = []
            for record in cache["records"]:
                print(f"[formula gradients] {side} batch={row['batch']} {record['name']}", flush=True)
                branches.append(audit_branch(record, matcher, device))
            stages[side].append({"batch": row["batch"], "category_indices": row["category_indices"], "branches": branches})
            del cache
    report = {"complete": True, "scope": SCOPE, "inputs": manifest, "captures": captures,
              "analysis_device": device, "training_updates": 0, "formulas": list(VARIANTS),
              "stages": stages, **compare_stages(stages)}
    # Both endpoints retain their own native query features and prototypes. Only
    # the formula within each endpoint is intervened on; no feature swap.
    save_json(output / "report.json", report)
    print("\n=== Eq.2 fixed-assignment training-formula summary ===", flush=True)
    print("stage variant                  negative |dL/dlogit|  positive |dL/dlogit|  rematched GT%", flush=True)
    for side in STAGES:
        for variant, values in report["stage_summaries"][side]["all"]["variants"].items():
            neg = values["fixed_assignment/negative/mean_abs_logit_gradient"]
            pos = values["fixed_assignment/positive/mean_abs_logit_gradient"]
            rematch = values["rematched_GT_fraction"]
            fmt = lambda v: "n/a" if v is None else f"{v:.6g}"
            print(f"{side:5} {variant:25} {fmt(neg):>18} {fmt(pos):>21} {fmt(None if rematch is None else 100*rematch):>15}", flush=True)
    print("\n" + SCOPE, flush=True)
    print(f"[save] {output / 'report.json'}", flush=True)
    return report


def run(args):
    torch.set_num_threads(args.cpu_threads)
    output, manifest = prepare(args)
    captures = {}
    for side in STAGES:
        capture = load_capture(output, side, manifest["fingerprint"])
        if capture is None:
            if args.analyze_only:
                raise ValueError(f"Missing {side} capture; --analyze-only will not run GPU inference")
            capture = capture_stage(args, output, manifest, side, captures.get("8ep"))
        captures[side] = capture
    verify_pair(captures, args.batches)
    if not args.capture_only:
        return analyze(args, output, manifest, captures)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint-8ep", required=True)
    p.add_argument("--checkpoint-12ep", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--config-file", default="lami_dino/configs/dino_convnext_large_4scale_12ep_lvis_no_radius.py")
    p.add_argument("--batches", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--analysis-device", help="default: capture device; CPU for --analyze-only")
    p.add_argument("--cpu-threads", type=int, default=4)
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--analyze-only", action="store_true")
    mode.add_argument("--capture-only", action="store_true")
    return p.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
