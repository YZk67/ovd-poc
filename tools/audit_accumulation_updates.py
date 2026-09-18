#!/usr/bin/env python3
"""Replay native micro vs pooled-GT normalization through the update pipeline.

One forward per microbatch; identical FedLoss/matching. Live weights/moments
never step. Native AdamW steps run only on disposable same-device copies.
See docs/accumulation_update_audit.md. Trusted local checkpoints only.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import gc
import hashlib
import json
import math
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.distributed as dist

from tools.accumulation_objective_ops import (
    compare_gradients, dn_layout, normalization_plan, paired_fedloss, sample_global_categories,
)
from tools.accumulation_update_ops import (
    ARMS, add_optional, assign_gradients, capture_gradients, shadow_adamw_step,
    snapshot_gradients, verify_forward,
)
from tools.audit_accumulation_objective import forward_observations, parameter_groups, verify_identity
from tools.compare_rare_pr_reports import file_identity, load_json, save_json


SCOPE = [
    "Fixed full no-radius 8ep checkpoint and restored native optimizer/scaler state.",
    "Same mapped train_norare inputs/native FedLoss/RNG/matching as the source objective audit.",
    "ONE forward per microbatch, two total-gradient probes plus an unchanged APR probe.",
    "Only detection class/L1/GIoU (final/aux/encoder/DN) get pooled-GT coefficients; APR/RPSA unchanged.",
    "Native FP32 model forward; AMP-scaled total gradients, manual rank mean, native GradScaler unscale.",
    "Native Trainer APR routing and separate clipping; each arm recomputes its own clipping coefficients.",
    "Actual torch.optim.AdamW steps on disjoint same-device parameter/moment copies; no live optimizer step.",
    "Every window/arm starts at the SAME endpoint moments, LR and parameter state; no sequential continuation.",
    "No scheduler/scaler update, training run, validation inference or model checkpoint output.",
    "Not bitwise DDP bucket replay, physical eight-GPU training, historical provenance or APr attribution.",
    "Unchanged live prototypes are not evidence that a future pooled-GT training run cannot collapse.",
]


def validate_reference(report):
    if (not report.get("complete") or report.get("decision") != "OBJECTIVE_DIFFERENCES_ONLY_NOT_PERFORMANCE_ATTRIBUTION"
            or not report.get("model_state_unchanged") or not report.get("source_checkpoint_unchanged")
            or report.get("optimizer_steps") != 0):
        raise ValueError("Require the completed, zero-update objective-audit report")
    inputs = report["inputs"]
    expected = {"schema_version": 2, "world_size": 4, "physical_batch": 16,
                "accumulation": 2, "effective_batch": 32, "reference_world_size": 8}
    if any(inputs.get(k) != v for k, v in expected.items()):
        raise ValueError("Unexpected objective-audit batch/schema protocol")
    if not 1 <= inputs["windows"] <= 4 or len(report["windows"]) != inputs["windows"]:
        raise ValueError("Require 1..4 complete reference windows")
    if (report["protocol"]["iteration"] != 56800
            or report["protocol"]["classifier"]["tpa_prototype_mode_strength"] != 0):
        raise ValueError("Require the no-radius 8ep reference")
    digest = hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest()
    if digest != report["fingerprint"]:
        raise ValueError("Reference fingerprint mismatch")


def check_sources(report):
    identities = [report["inputs"]["checkpoint"], report["inputs"]["config"],
                  report["train_annotations"], *report["assets"]]
    for identity in identities:
        verify_identity(identity)
    for name, digest in report["inputs"]["code"].items():
        path = (ROOT / name).resolve()
        if ROOT not in path.parents or file_identity(path)["sha256"] != digest:
            raise ValueError("Reference audit source changed: " + name)


def prepare(args):
    report = load_json(args.objective_audit)
    validate_reference(report)
    if args.num_gpus != 4 or args.cpu_threads < 1:
        raise ValueError("Require four GPUs and positive CPU thread count")
    if str(torch.__version__) != report["inputs"]["torch"]:
        raise ValueError("Use the objective audit's PyTorch/lami environment")
    output = Path(args.output_dir).resolve()
    protected = [Path(args.objective_audit).resolve(),
                 Path(report["inputs"]["checkpoint"]["path"]).resolve(),
                 Path(report["inputs"]["config"]["path"]).resolve()]
    if any(output == p or output in p.parents or p in output.parents for p in protected):
        raise ValueError("Output must be separate from source files")
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use a new empty output directory; no overwrite/reuse")
    check_sources(report)
    code = {name: file_identity(ROOT / name)["sha256"] for name in (
        "tools/audit_accumulation_updates.py", "tools/accumulation_update_ops.py")}
    inputs = {"schema_version": 1, "objective_audit": file_identity(args.objective_audit),
              "reference_fingerprint": report["fingerprint"], "code": code,
              "cpu_threads": args.cpu_threads, "num_gpus": args.num_gpus,
              "windows": report["inputs"]["windows"]}
    signature = hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest()
    save_json(output / "manifest.json", {"inputs": inputs, "fingerprint": signature})
    n = inputs["windows"]
    print(f"[budget] {n} windows, {32*n} image forwards TOTAL, native categories only; "
          f"{2*n} disposable AdamW steps; ZERO live updates and ZERO LVIS inference", flush=True)
    return inputs, signature


def finish_arm(trainer, optimizer, parameters, scaled, present, local_apr, scaler_state):
    """All ranks call the actual trainer routing and clipping methods."""
    from torch.cuda.amp import GradScaler

    assign_gradients(parameters, scaled, present)
    scaler = GradScaler()
    scaler.load_state_dict(deepcopy(scaler_state))
    # Lazily initialize scale/growth tensors before calling native unscale_.
    scaler.scale(torch.zeros((), device=parameters[0].device))
    scaler.unscale_(optimizer)
    stages = {"raw": snapshot_gradients(parameters)}  # also rejects overflow
    trainer._route_tpa_gradients(tuple(
        None if g is None else g.to(parameters[0].device) for g in local_apr))
    stages["routed"] = snapshot_gradients(parameters)
    norms = trainer.clip_model_grads()
    stages["clipped"] = snapshot_gradients(parameters)
    # Lazy initialization may represent an integer init_scale as a float;
    # require unchanged scalar VALUES, not serialization-type identity.
    if scaler.state_dict() != scaler_state:
        raise ValueError("Scalars in the restored GradScaler state changed")
    norms = {"detector": float(norms[0]), "tpa": float(norms[1])}
    if any(not math.isfinite(v) for v in (*norms.values(), *trainer._last_tpa_projection_metrics.values())):
        raise FloatingPointError("Nonfinite native routing/clipping statistics")
    return stages, {"routing": trainer._last_tpa_projection_metrics,
                    "preclip_norms": norms,
                    "clip_coefficients": {k: min(1., .5/(v+1e-6)) for k, v in norms.items()},
                    "amp_scale": float(scaler.get_scale()), "finite_unscaled_gradients": True}


def worker(args, inputs, signature):
    from detectron2.config import LazyConfig, instantiate
    from detectron2.data import MetadataCatalog
    from detectron2.utils.events import EventStorage
    from detrex.utils.utils import get_fed_loss_inds
    from lami_dino.checkpoint_init import load_trusted_torch_file, validate_backbone_trainable_scope
    from tools.audit_pairing_optimizer_sources import _allreduce_cpu_parts, _forward_seed, _input_record
    from tools.capture_tpa_gradients import state_versions
    from tools.decoder_aux_ablation_ops import state_digest, validate_resume
    from tools.evaluate_decoder_rollback import endpoint_state
    from tools.query_path_update_ops import clear_eval_caches
    from tools.tpa_geometry_audit_ops import tensor_digest
    from tools.train_net import Trainer

    if not torch.cuda.is_available() or not dist.is_initialized() or dist.get_world_size() != 4:
        raise ValueError("Require initialized four-rank CUDA")
    reference = load_json(inputs["objective_audit"]["path"])
    verify_identity(inputs["objective_audit"])
    old_inputs = reference["inputs"]
    rank, device_index = dist.get_rank(), torch.cuda.current_device()
    device = torch.device("cuda", device_index)
    seed = old_inputs["seed"]
    torch.set_num_threads(args.cpu_threads)
    random.seed(seed + rank)
    np.random.seed(seed + rank)
    torch.manual_seed(seed + rank)
    torch.cuda.manual_seed(seed + rank)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    cfg = LazyConfig.load(old_inputs["config"]["path"])
    cfg = LazyConfig.apply_overrides(cfg, ["model.device=cuda", "train.device=cuda",
        "model.classifier.tpa_prototype_mode_strength=0.0", "model.classifier.tpa_train_aggregation=calibrated"])
    if (cfg.dataloader.train.dataset.names != "lvis_v1_train_norare"
            or not cfg.dataloader.train.mapper.is_train or not cfg.model.use_fed_loss
            or cfg.model.cluster_fed_loss or cfg.model.get("teacher_rpsa", False)
            or cfg.train.gradient_accumulation_steps != 2 or cfg.train.lr_scheduler_max_iter != 85200
            or not cfg.train.amp.enabled or not cfg.train.tpa_conflict_projection
            or not cfg.train.separate_tpa_grad_clip or not cfg.train.clip_grad.enabled
            or dict(cfg.train.clip_grad.params) != {"max_norm": .5, "norm_type": 2}):
        raise ValueError("Native audit/training protocol differs")
    cfg.dataloader.train.total_batch_size = 16
    cfg.dataloader.train.num_workers = 0
    model = instantiate(cfg.model).to(device)
    checkpoint = load_trusted_torch_file(old_inputs["checkpoint"]["path"])
    resume = validate_resume(checkpoint)
    if resume != old_inputs["resume_identity_only"]:
        raise ValueError("Endpoint resume state differs from source audit")
    model.load_state_dict(endpoint_state(checkpoint, 56799), strict=True)
    model.train()
    validate_backbone_trainable_scope(model.backbone, cfg.train.backbone_trainable_scope)
    if any(isinstance(m, torch.nn.modules.batchnorm._BatchNorm) and m.training for m in model.modules()):
        raise ValueError("Active training BatchNorm is unsupported")
    model.tpa_advance_step = False
    inventory, parameters, groups = parameter_groups(model)
    if inventory != reference["inventory"]:
        raise ValueError("Trainable inventory differs from objective audit")
    cfg.optimizer.params.model = model
    # The preceding objective audit had no optimizer. Bookkeeping must not
    # perturb the mapper/sampler RNG stream that we are replaying from it.
    with _forward_seed(seed, device_index):
        optimizer = instantiate(cfg.optimizer)
        optimizer.load_state_dict(checkpoint["trainer"]["optimizer"])
    scaler_state = deepcopy(checkpoint["trainer"]["grad_scaler"])
    if state_digest(optimizer.state_dict()) != resume["optimizer"]:
        raise ValueError("Restored optimizer state mismatch")
    del checkpoint
    # No trainer initialization, hooks, data iterator, scheduler or run_step.
    # Only these unchanged methods are used: _get_tpa, _route_tpa_gradients,
    # clip_grads, clip_model_grads. Native optimizer.step is NEVER called.
    trainer = Trainer.__new__(Trainer)
    trainer.model = model
    trainer.separate_tpa_grad_clip = True
    trainer.tpa_conflict_projection = True
    trainer.clip_grad_params = dict(cfg.train.clip_grad.params)
    tpa_parameters = list(trainer._get_tpa().parameters())
    tpa_ids = {id(p) for p in tpa_parameters}
    if tpa_ids != {id(p) for p, g in zip(parameters, groups) if g == "upstream_tpa"}:
        raise ValueError("Native TPA routing inventory differs")
    versions, model_digest = state_versions(model), state_digest(model.state_dict())
    loader = iter(instantiate(cfg.dataloader.train))
    annotations = file_identity(MetadataCatalog.get("lvis_v1_train_norare").json_file)
    if annotations != reference["train_annotations"]:
        raise ValueError("Training annotations changed")
    runtime = {"torch": str(torch.__version__), "cuda": torch.version.cuda,
               "cudnn": torch.backends.cudnn.version(), "device": torch.cuda.get_device_name(device)}
    if any(runtime[k] != reference["runtime"][k] for k in runtime):
        raise ValueError("Replay device/runtime differs from objective audit")
    windows = []
    with EventStorage(start_iter=56800):
        for window, ref in enumerate(reference["windows"]):
            batches = [next(loader), next(loader)]
            if any(len(data) != 4 for data in batches):
                raise ValueError("Expected four images per rank/microbatch")
            mapped = [_input_record(data) for data in batches]
            if any(c < 0 or c >= model.num_classes or bool(model.novel_idx[c])
                   for rows in mapped for row in rows for c in row["class_values"]):
                raise ValueError("Rare/invalid GT in training audit")
            if mapped != [row["inputs"] for row in ref["forwards"]["native"][rank]]:
                raise ValueError("Mapped training inputs differ from reference; no update comparison")
            all_mapped = [None] * 4
            dist.all_gather_object(all_mapped, mapped)
            counts = [sum(len(row["class_values"]) for rr in all_mapped for row in rr[m]) for m in range(2)]
            plan = normalization_plan(counts)
            if plan != ref["normalization"]:
                raise ValueError("GT normalization plan changed")
            micro_gt = [sorted({c for rr in all_mapped for row in rr[m] for c in row["class_values"]})
                        for m in range(2)]
            accumulated = {arm: None for arm in ARMS}
            present, apr, rows = [False]*len(parameters), None, []
            for micro in range(2):
                data = deepcopy(batches[micro])
                forward_seed = seed + 10000 + window*100 + micro*10 + rank
                def native_draw():
                    return sample_global_categories(model, micro_gt[micro], get_fed_loss_inds, rank=rank,
                                                    broadcast=lambda x: dist.broadcast(x, src=0))
                with (_forward_seed(forward_seed, device_index),
                      paired_fedloss(model, native_draw=native_draw) as fed,
                      forward_observations(model, plan["criterion_normalizers"][micro]) as obs):
                    losses = model(data)
                    if getattr(model, "tpa_stabilizing", False):
                        raise ValueError("Native APR routing is disabled during stabilization; unexpected 8ep state")
                    rng = [tensor_digest(torch.get_rng_state()), tensor_digest(torch.cuda.get_rng_state(device_index))]
                    parts, flags, local_apr, values = capture_gradients(
                        losses, parameters, tpa_parameters, plan["detection_loss_multipliers"][micro],
                        float(scaler_state["scale"]))
                row = {"micro": micro, "inputs": mapped[micro], "seed": forward_seed,
                       "rng_after": rng, **fed, **obs,
                       "weighted_losses": {k: float(v.detach()) for k, v in losses.items() if k.startswith("loss")},
                       "weighted_objectives": values}
                verify_forward(row, ref["forwards"]["native"][rank][micro])
                if obs["dn"] != dn_layout([len(x["class_values"]) for x in mapped[micro]], model.dn_number):
                    raise ValueError("DN grouping changed")
                for arm in ARMS:
                    accumulated[arm] = add_optional(accumulated[arm], parts[arm])
                apr = add_optional(apr, local_apr)
                present = [a or b for a, b in zip(present, flags)]
                rows.append(row)
                clear_eval_caches(model)
                del data, losses, parts, local_apr
                if state_versions(model) != versions or any(p.grad is not None for p in parameters):
                    raise ValueError("Capture changed live model state/grad buffers")
                print(f"[rank {rank}] window={window+1}/{len(reference['windows'])} micro={micro+1}/2 paired", flush=True)
            for arm in ARMS:
                accumulated[arm] = _allreduce_cpu_parts(accumulated[arm], parameters)
            flags = torch.tensor(present, dtype=torch.uint8, device=device)
            dist.all_reduce(flags, op=dist.ReduceOp.MAX)
            present = [bool(x) for x in flags.tolist()]
            arm_stages, arm_updates, receipts = {}, {}, {}
            try:
                for arm in ARMS:
                    print(f"[rank {rank}] window={window+1} {arm}: unscale / native route / native clip", flush=True)
                    stages, receipt = finish_arm(trainer, optimizer, parameters, accumulated[arm], present, apr, scaler_state)
                    if rank == 0:
                        print(f"[shadow AdamW] window={window+1} {arm}; live parameters/moments untouched", flush=True)
                        with _forward_seed(seed, device_index):
                            updates, shadow_receipt = shadow_adamw_step(optimizer, parameters, resume["optimizer"])
                        arm_updates[arm] = updates
                        arm_stages[arm] = stages
                        receipts[arm] = {**receipt, "shadow": shadow_receipt}
                    for p in parameters:
                        p.grad = None
                    del stages
                    dist.barrier()
            finally:
                for p in parameters:
                    p.grad = None
            if state_versions(model) != versions or state_digest(optimizer.state_dict()) != resume["optimizer"]:
                raise ValueError("Update audit changed live parameters/optimizer")
            records = [None] * 4
            dist.all_gather_object(records, rows)
            if rank == 0:
                comparisons = {stage: compare_gradients(arm_stages[ARMS[0]][stage], arm_stages[ARMS[1]][stage], groups)
                               for stage in ("raw", "routed", "clipped")}
                comparisons["adamw_update"] = compare_gradients(arm_updates[ARMS[0]], arm_updates[ARMS[1]], groups)
                result = {"window": window, "normalization": plan, "arms": receipts,
                          "comparisons": comparisons, "forwards": records,
                          "paired_native_forward_verified": True, "live_optimizer_unchanged": True}
                windows.append(result)
                save_json(Path(args.output_dir) / f"window_{window}.json", result)
                print(f"\n=== Window {window+1}: pooled GT minus native; relative difference / cosine ===", flush=True)
                for stage, metrics in comparisons.items():
                    for group in ("all_trainable", "decoder_core", "query_content", "upstream_tpa"):
                        m = metrics[group]
                        print(f"{stage:14s} {group:16s} relative={m['relative_difference_l2']} cosine={m['cosine']}", flush=True)
            del accumulated, arm_stages, arm_updates, batches, rows, records, apr
            gc.collect()
            dist.barrier()
    if state_versions(model) != versions or state_digest(model.state_dict()) != model_digest:
        raise ValueError("Persistent model state changed")
    if state_digest(optimizer.state_dict()) != resume["optimizer"]:
        raise ValueError("Live optimizer state changed")
    if rank == 0:
        check_sources(reference)
        verify_identity(inputs["objective_audit"])
        for name, digest in inputs["code"].items():
            if file_identity(ROOT / name)["sha256"] != digest:
                raise ValueError("Update audit code changed during capture")
        report = {"complete": True, "inputs": inputs, "fingerprint": signature,
                  "runtime": runtime, "inventory": inventory, "windows": windows,
                  "checkpoint": old_inputs["checkpoint"], "config": old_inputs["config"],
                  "protocol": {"iteration": 56800, "physical_batch": 16, "accumulation": 2,
                               "effective_batch": 32, "fedloss": "native_per_microbatch",
                               "classifier": reference["protocol"]["classifier"],
                               "clip_grad": dict(cfg.train.clip_grad.params),
                               "native_conflict_projection": True, "separate_tpa_clip": True,
                               "amp_scale": float(scaler_state["scale"])},
                  "resume_identity": resume, "scope": SCOPE,
                  "model_state_unchanged": True, "live_optimizer_state_unchanged": True,
                  "source_checkpoint_unchanged": True, "live_optimizer_steps": 0,
                  "shadow_optimizer_steps": 2*len(windows), "full_validation_inference": 0,
                  "decision": "CONDITIONAL_ONE_STEP_DIFFERENCES_NOT_APR_ATTRIBUTION"}
        path = Path(args.output_dir) / "report.json"
        save_json(path, report)
        save_json(Path(args.output_dir) / "COMPLETE.json", {"fingerprint": signature, "report": file_identity(path)})
        print(f"[save] {path}\nZERO live updates; no trained checkpoint or AP conclusion.", flush=True)
    dist.barrier()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--objective-audit", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-gpus", type=int, default=4)
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
