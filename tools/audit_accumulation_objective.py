#!/usr/bin/env python3
"""Four-rank, zero-update 2x2 audit of GT normalization and FedLoss sampling.

Not a physical-batch-32 replay; no optimizer, clipping, AP evaluation or training
change. See docs/accumulation_objective_audit.md. Trusted checkpoints only.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from copy import deepcopy
import gc
import hashlib
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.distributed as dist

from tools.accumulation_objective_ops import (
    category_overlap, compare_gradients, dn_layout, normalization_plan,
    objective_gradients, paired_fedloss,
)
from tools.compare_rare_pr_reports import file_identity, save_json


SCOPE = [
    "Fixed no-radius 8ep weights, same mapped train_norare images and paired forward RNG.",
    "2x2: native independent vs shared-window FedLoss; micro vs pooled-GT normalization.",
    "GT normalization changes detection class/L1/GIoU (final/aux/encoder/DN) only; APR/RPSA keep native weights.",
    "Shared categories also change query initialization, DN label semantics, TPA/APR/RPSA and matching: NOT a final-loss-only mask.",
    "Raw FP32 parameter gradients BEFORE APR conflict routing, clipping or AdamW. No mixed-precision replay.",
    "DN grouping and image padding remain local to each four-image forward. Stateful training BatchNorm is rejected.",
    "This is a conditional objective comparison, NOT a physical eight-GPU batch-32 or historical training replay.",
    "A gradient difference is not proof of harm, rare AP improvement, or the cause of an 8ep-to-12ep change.",
]


def prepare(args):
    from lami_dino.checkpoint_init import load_trusted_torch_file
    from tools.decoder_aux_ablation_ops import validate_resume
    from tools.evaluate_decoder_rollback import endpoint_state

    if args.num_gpus != 4 or not 1 <= args.windows <= 4 or not 0 <= args.seed < 2**31:
        raise ValueError("Require four GPUs, 1..4 windows, nonnegative 31-bit seed")
    if args.cpu_threads < 1:
        raise ValueError("cpu-threads must be positive")
    output = Path(args.output_dir).resolve()
    protected = [Path(args.checkpoint).resolve(), Path(args.config_file).resolve()]
    if any(output == p or output in p.parents or p in output.parents for p in protected):
        raise ValueError("Output must be separate from protected source files/directories")
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use a new empty output directory; no silent reuse/overwrite")
    inputs = {"schema_version": 1, "checkpoint": file_identity(args.checkpoint),
              "config": file_identity(args.config_file), "windows": args.windows,
              "seed": args.seed, "world_size": 4, "physical_batch": 16,
              "accumulation": 2, "effective_batch": 32, "reference_world_size": 8,
              "torch": str(torch.__version__), "cpu_threads": args.cpu_threads}
    ckpt = load_trusted_torch_file(args.checkpoint)
    endpoint_state(ckpt, 56799)
    inputs["resume_identity_only"] = validate_resume(ckpt)
    del ckpt
    paths = set()
    for folder in ("lami_dino", "configs/common", "detrex/modeling", "detrex/utils",
                   "detrex/layers", "detrex/data", "detectron2/detectron2/data"):
        paths.update((ROOT / folder).rglob("*.py"))
    paths.update(ROOT / "tools" / name for name in (
        "audit_accumulation_objective.py", "accumulation_objective_ops.py",
        "audit_pairing_optimizer_sources.py", "decoder_aux_ablation_ops.py",
        "decoder_loss_audit_ops.py", "capture_tpa_gradients.py", "evaluate_decoder_rollback.py",
        "query_path_update_ops.py", "train_net.py", "compare_rare_pr_reports.py",
        "tpa_geometry_audit_ops.py"))
    inputs["code"] = {str(p.relative_to(ROOT)): file_identity(p)["sha256"] for p in sorted(paths)}
    signature = hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest()
    save_json(output / "manifest.json", {"inputs": inputs, "fingerprint": signature})
    print(f"[budget] {args.windows} windows; {32*args.windows} paired image exposures per policy; "
          f"{64*args.windows} forward image exposures total; four GPUs; ZERO updates", flush=True)
    return inputs, signature


def verify_identity(identity):
    if file_identity(identity["path"]) != identity:
        raise ValueError("Source file changed: " + identity["path"])


@contextmanager
def forward_observations(model, expected_normalizer):
    """Observe native criterion denominators, DN layout and Hungarian assignments."""
    result = {"normalizers": [], "matches": []}
    original_loss = model.criterion.get_loss
    original_dn = model.prepare_for_cdn

    def get_loss(loss, outputs, targets, indices, num_boxes, **kwargs):
        result["normalizers"].append(float(num_boxes))
        return original_loss(loss, outputs, targets, indices, num_boxes, **kwargs)

    def prepare_dn(*args, **kwargs):
        output = original_dn(*args, **kwargs)
        meta = output[-1]
        result["dn"] = {k: int(meta[k]) if meta else 0 for k in ("dn_num", "single_padding")}
        return output

    def match_hook(module, inputs, output):
        result["matches"].append([
            [src.detach().cpu().tolist(), tgt.detach().cpu().tolist()] for src, tgt in output
        ])

    model.criterion.get_loss = get_loss
    model.prepare_for_cdn = prepare_dn
    handle = model.criterion.matcher.register_forward_hook(match_hook)
    try:
        yield result
        if not result["normalizers"] or "dn" not in result:
            raise ValueError("Missing native criterion/DN observations")
        allowed = {expected_normalizer, expected_normalizer * result["dn"]["dn_num"]}
        if any(n not in allowed for n in result["normalizers"]):
            raise ValueError("Actual loss normalizers differ from the audited algebra")
    finally:
        handle.remove()
        model.criterion.get_loss = original_loss
        model.prepare_for_cdn = original_dn


def parameter_groups(model):
    from tools.query_path_update_ops import canonical_name, key_group
    inventory, parameters, groups = [], [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        group = key_group(canonical_name(name))
        if group in ("unclassified", "fixed_protocol", "frozen_clip"):
            raise ValueError("Unexpected trainable parameter: " + name)
        parameters.append(p)
        groups.append(group)
        inventory.append({"name": name, "group": group, "shape": list(p.shape)})
    if not parameters or "upstream_tpa" not in groups or "decoder_core" not in groups:
        raise ValueError("Missing trainable TPA/decoder")
    return inventory, parameters, groups


def validate_pairing(native, shared):
    """Check every rank/microbatch, not just rank zero's data or RNG."""
    if len(native) != len(shared):
        raise ValueError("Incomplete rank observations")
    result = []
    for rank, (a_rows, b_rows) in enumerate(zip(native, shared)):
        if len(a_rows) != 2 or len(b_rows) != 2:
            raise ValueError("Expected two microbatches per rank")
        for a, b in zip(a_rows, b_rows):
            for key in ("inputs", "seed", "rng_after", "native_indices", "dn", "normalizers"):
                if a[key] != b[key]:
                    raise ValueError("Unpaired FedLoss comparison: " + key)
            if len(a["matches"]) != len(b["matches"]):
                raise ValueError("Matcher invocation layout changed")
            changed, total = 0, 0
            for aa, bb in zip(a["matches"], b["matches"]):
                if len(aa) != len(bb):
                    raise ValueError("Matcher batch layout changed")
                for x, y in zip(aa, bb):
                    changed += x != y
                    total += 1
            result.append({"rank": rank, "micro": a["micro"],
                           "changed_image_branch_assignments": changed,
                           "image_branch_assignments": total,
                           "category_overlap": category_overlap(a["selected_indices"], b["selected_indices"])})
    for micro in range(2):
        for policy in (native, shared):
            if len({tuple(rows[micro]["selected_indices"]) for rows in policy}) != 1:
                raise ValueError("FedLoss sets differ across ranks")
    if len({tuple(row["selected_indices"]) for rows in shared for row in rows}) != 1:
        raise ValueError("Shared policy did not share categories across the full window")
    return result


def worker(args, inputs, signature):
    from detectron2.config import LazyConfig, instantiate
    from detectron2.data import MetadataCatalog
    from detectron2.utils.events import EventStorage
    from detrex.utils.utils import get_fed_loss_inds
    from lami_dino.checkpoint_init import load_trusted_torch_file, validate_backbone_trainable_scope
    from tools.audit_pairing_optimizer_sources import _allreduce_cpu_parts, _forward_seed, _input_record
    from tools.capture_tpa_gradients import state_versions
    from tools.decoder_aux_ablation_ops import state_digest
    from tools.evaluate_decoder_rollback import endpoint_state
    from tools.query_path_update_ops import clear_eval_caches
    from tools.tpa_geometry_audit_ops import tensor_digest

    torch.set_num_threads(args.cpu_threads)
    if not torch.cuda.is_available() or not dist.is_initialized() or dist.get_world_size() != 4:
        raise ValueError("Require the four-GPU lami environment")
    rank, device_index = dist.get_rank(), torch.cuda.current_device()
    device = torch.device("cuda", device_index)
    random.seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)
    torch.cuda.manual_seed(args.seed + rank)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    cfg = LazyConfig.load(inputs["config"]["path"])
    cfg = LazyConfig.apply_overrides(cfg, [
        "model.device=cuda", "train.device=cuda",
        "model.classifier.tpa_prototype_mode_strength=0.0",
        "model.classifier.tpa_train_aggregation=calibrated",
    ])
    if (cfg.dataloader.train.dataset.names != "lvis_v1_train_norare"
            or not cfg.dataloader.train.mapper.is_train or not cfg.model.use_fed_loss
            or cfg.model.cluster_fed_loss or cfg.model.get("teacher_rpsa", False)
            or cfg.train.gradient_accumulation_steps != 2 or cfg.train.lr_scheduler_max_iter != 85200):
        raise ValueError("Require no-radius native train_norare/FedLoss, accumulation=2, horizon=85200")
    cfg.dataloader.train.total_batch_size = 16
    cfg.dataloader.train.num_workers = 0
    asset_paths = {str(Path(cfg.model[k]).resolve()) for k in (
        "query_path", "eval_query_path", "vlm_query_path", "clip_head_path",
        "seen_classes", "all_classes", "cat_freq_path") if cfg.model.get(k)}
    asset_paths.update(str(Path(cfg.model.classifier[k]).resolve()) for k in (
        "zs_weight_path", "eval_zs_weight_path", "text_embed_path", "eval_text_embed_path")
        if cfg.model.classifier.get(k))
    assets = [file_identity(p) for p in sorted(asset_paths)]
    model = instantiate(cfg.model).to(device)
    ckpt = load_trusted_torch_file(inputs["checkpoint"]["path"])
    model.load_state_dict(endpoint_state(ckpt, 56799), strict=True)
    del ckpt
    model.train()
    validate_backbone_trainable_scope(model.backbone, cfg.train.backbone_trainable_scope)
    if any(isinstance(m, torch.nn.modules.batchnorm._BatchNorm) and m.training for m in model.modules()):
        raise ValueError("Training BatchNorm changes batch statistics; needs a separate audit")
    model.tpa_advance_step = False
    versions = state_versions(model)
    before_digest = state_digest(model.state_dict())
    inventory, parameters, groups = parameter_groups(model)
    loader = iter(instantiate(cfg.dataloader.train))
    annotations = file_identity(MetadataCatalog.get("lvis_v1_train_norare").json_file)
    runtime = {"torch": str(torch.__version__), "cuda": torch.version.cuda,
               "cudnn": torch.backends.cudnn.version(), "device": torch.cuda.get_device_name(device),
               "precision": "FP32 forward and raw gradients; TF32 disabled"}
    protocol = {
        "iteration": 56800, "fed_loss_num_cat": int(model.fed_loss_num_cat),
        "dn_number": int(model.dn_number), "label_noise_ratio": float(model.label_noise_ratio),
        "box_noise_scale": float(model.box_noise_scale),
        "classifier": {k: cfg.model.classifier.get(k) for k in (
            "tpa_num_prototypes", "tpa_tau", "tpa_cls_tau", "tpa_dropout",
            "tpa_slot_prior_strength", "tpa_prototype_mode_strength", "tpa_train_aggregation")},
        "tpa_task_gradient_scale": float(model.tpa_task_gradient_scale),
        "configured_routing_not_applied": bool(cfg.train.tpa_conflict_projection),
        "configured_separate_clipping_not_applied": bool(cfg.train.separate_tpa_grad_clip),
    }
    if rank == 0:
        save_json(Path(args.output_dir) / "capture_inputs.json",
                  {"fingerprint": signature, "annotations": annotations, "assets": assets,
                   "runtime": runtime, "protocol": protocol, "inventory": inventory})
    windows = []
    with EventStorage(start_iter=56800):
        for window in range(args.windows):
            batches = [next(loader), next(loader)]
            if any(len(data) != 4 for data in batches):
                raise ValueError("Expected four images per rank and microbatch")
            mapped = [_input_record(data) for data in batches]
            if any(c < 0 or c >= model.num_classes or bool(model.novel_idx[c])
                   for rows in mapped for row in rows for c in row["class_values"]):
                raise ValueError("Rare/invalid GT entered train_norare audit")
            all_mapped = [None] * 4
            dist.all_gather_object(all_mapped, mapped)
            counts = [sum(len(row["class_values"]) for rr in all_mapped for row in rr[m]) for m in range(2)]
            plan = normalization_plan(counts)
            all_gt = sorted({c for rr in all_mapped for rows in rr for row in rows for c in row["class_values"]})
            if len(all_gt) > model.fed_loss_num_cat:
                raise ValueError("Window GT union exceeds FedLoss budget; cannot preserve vocabulary shape")
            shared_seed = args.seed + 1000 + window
            with _forward_seed(shared_seed, device_index):
                if rank == 0:
                    shared = get_fed_loss_inds(torch.tensor(all_gt, device=device, dtype=torch.long),
                                              model.fed_loss_num_cat, model.num_classes, model.freq_weight)
                else:
                    shared = torch.empty(model.fed_loss_num_cat, device=device, dtype=torch.long)
                dist.broadcast(shared, src=0)
            gradients, records, scalars = {}, {}, {}
            for policy in ("native", "shared"):
                accumulated, rows = {}, []
                scalars[policy] = {"micro": 0., "pooled_gt": 0.}
                for micro in range(2):
                    data = deepcopy(batches[micro])  # native forward remaps GT labels in place
                    actual_inputs = _input_record(data)
                    if actual_inputs != mapped[micro]:
                        raise ValueError("Mapped images/GT changed before paired forward")
                    seed = args.seed + 10000 + window * 100 + micro * 10 + rank
                    with (_forward_seed(seed, device_index),
                          paired_fedloss(model, shared if policy == "shared" else None) as fed,
                          forward_observations(model, plan["criterion_normalizers"][micro]) as obs):
                        losses = model(data)
                        rng = [tensor_digest(torch.get_rng_state()), tensor_digest(torch.cuda.get_rng_state(device_index))]
                        parts, values = objective_gradients(
                            losses, parameters, plan["detection_loss_multipliers"][micro])
                    expected_dn = dn_layout([len(r["class_values"]) for r in mapped[micro]], model.dn_number)
                    if obs["dn"] != expected_dn:
                        raise ValueError("Native DN grouping differs from expected local grouping")
                    for name, vector in parts.items():
                        if name not in accumulated:
                            accumulated[name] = vector
                        else:
                            for total, current in zip(accumulated[name], vector):
                                total.add_(current)
                        scalars[policy][name] += values[name]
                    rows.append({"micro": micro, "inputs": actual_inputs, "seed": seed,
                                 "rng_after": rng, **fed, **obs,
                                 "sampled_rare_count": int(model.novel_idx[fed["selected_indices"]].sum()),
                                 "weighted_losses": {k: float(v.detach()) for k, v in losses.items() if k.startswith("loss")}})
                    clear_eval_caches(model)
                    del data, losses, parts
                    if state_versions(model) != versions or any(p.grad is not None for p in parameters):
                        raise ValueError("Read-only audit changed state or populated parameter .grad")
                    print(f"[rank {rank}] window={window+1}/{args.windows} {policy} micro={micro+1}/2", flush=True)
                for name in accumulated:
                    accumulated[name] = _allreduce_cpu_parts(accumulated[name], parameters)
                gradients[policy] = accumulated
                gathered = [None] * 4
                dist.all_gather_object(gathered, rows)
                records[policy] = gathered
            pairing = validate_pairing(records["native"], records["shared"])
            scalar_tensor = torch.tensor([scalars[p][n] for p in ("native", "shared") for n in ("micro", "pooled_gt")],
                                         device=device, dtype=torch.float64)
            dist.all_reduce(scalar_tensor)
            scalar_tensor /= 4
            if rank == 0:
                a, b = gradients["native"], gradients["shared"]
                comparisons = {
                    "normalization_with_native_categories": compare_gradients(a["micro"], a["pooled_gt"], groups),
                    "normalization_with_shared_categories": compare_gradients(b["micro"], b["pooled_gt"], groups),
                    "categories_with_micro_normalization": compare_gradients(a["micro"], b["micro"], groups),
                    "categories_with_pooled_normalization": compare_gradients(a["pooled_gt"], b["pooled_gt"], groups),
                    "both_changes_vs_native": compare_gradients(a["micro"], b["pooled_gt"], groups),
                }
                row = {"window": window, "normalization": plan, "shared_category_seed": shared_seed,
                       "shared_categories": shared.cpu().tolist(), "pairing": pairing,
                       "weighted_objectives": dict(zip(("native_micro", "native_pooled", "shared_micro", "shared_pooled"),
                                                       scalar_tensor.cpu().tolist())),
                       "comparisons": comparisons, "forwards": records}
                windows.append(row)
                save_json(Path(args.output_dir) / f"window_{window}.json", row)
                print(f"[window {window+1}] GT={counts} normalization multipliers={plan['detection_loss_multipliers']}", flush=True)
                for name, metrics in comparisons.items():
                    print(name, metrics["all_trainable"], flush=True)
                del a, b
            del gradients, accumulated, batches, records
            gc.collect()
            dist.barrier()
    del loader
    if state_versions(model) != versions or state_digest(model.state_dict()) != before_digest:
        raise ValueError("Model state changed during audit")
    if rank == 0:
        for identity in [inputs["checkpoint"], inputs["config"], annotations, *assets]:
            verify_identity(identity)
        for name, digest in inputs["code"].items():
            if file_identity(ROOT / name)["sha256"] != digest:
                raise ValueError("Audit code changed during capture: " + name)
        report = {"complete": True, "fingerprint": signature, "inputs": inputs,
                  "train_annotations": annotations, "assets": assets, "inventory": inventory,
                  "windows": windows, "runtime": runtime, "protocol": protocol,
                  "model_state_unchanged": True, "source_checkpoint_unchanged": True,
                  "optimizer_created": False, "optimizer_steps": 0, "full_validation_inference": 0,
                  "scope": SCOPE, "decision": "OBJECTIVE_DIFFERENCES_ONLY_NOT_PERFORMANCE_ATTRIBUTION"}
        path = Path(args.output_dir) / "report.json"
        save_json(path, report)
        save_json(Path(args.output_dir) / "COMPLETE.json", {"fingerprint": signature, "report": file_identity(path)})
        print(f"[save] {path}\nNo training/config/checkpoint changes; no AP conclusion.", flush=True)
    dist.barrier()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config-file", default="lami_dino/configs/dino_convnext_large_4scale_12ep_lvis.py")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-gpus", type=int, default=4)
    parser.add_argument("--windows", type=int, default=2)
    parser.add_argument("--seed", type=int, default=424242)
    parser.add_argument("--cpu-threads", type=int, default=2)
    parser.add_argument("--dist-url", default="auto")
    return parser.parse_args()


def main():
    args = parse_args()
    inputs, signature = prepare(args)
    from detectron2.engine import launch
    launch(worker, args.num_gpus, num_machines=1, machine_rank=0,
           dist_url=args.dist_url, args=(args, inputs, signature))


if __name__ == "__main__":
    main()
