#!/usr/bin/env python3
"""Optimizer-aware loss-source screen for terminal query/bank co-adaptation.

This is a bounded, read-only DDP audit rooted at the complete no-radius 8ep
checkpoint.  It captures native effective-batch gradients, reproduces APR
conflict routing and separate clipping, and analytically evaluates AdamW steps
with one loss source removed from either the terminal query path or TPA bank.
No optimizer step is called and no model/checkpoint tensor is modified.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import gc
import json
import math
from pathlib import Path
import random
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.distributed as dist

from lami_dino.checkpoint_init import load_trusted_torch_file
from lami_dino.prototype_ops import route_conflicting_task_gradient
from tools.compare_rare_pr_reports import file_identity, load_json, save_json
from tools.decoder_aux_ablation_ops import state_digest, validate_resume
from tools.diagnose_rare_fp_regions import fingerprint
from tools.evaluate_decoder_rollback import endpoint_state, validate_audit
from tools.pairing_optimizer_source_ops import (
    SOURCES,
    TARGETS,
    adamw_delta,
    clip_coefficient,
    counterfactual_norm,
    split_loss_keys,
    summarize_screen,
    vector_metrics,
)
from tools.query_path_update_ops import canonical_name, key_group
from tools.tpa_geometry_audit_ops import tensor_digest


START_ITERATION = 56800
ENDPOINTS = {"old": 56799, "new": 70999}
QUERY_GROUPS = {"query_content", "decoder_core", "final_projection"}


def _json_line(path, value):
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, sort_keys=True) + "\n")


def _source_code_identity():
    paths = {
        Path(__file__).resolve(),
        ROOT / "tools/pairing_optimizer_source_ops.py",
        ROOT / "tools/train_net.py",
        ROOT / "tools/query_path_update_ops.py",
        ROOT / "lami_dino/prototype_ops.py",
    }
    return {str(path.relative_to(ROOT)): file_identity(path)["sha256"] for path in sorted(paths)}


def prepare(args):
    if args.num_gpus != 4:
        raise ValueError("This audit must use the original four-rank DDP topology")
    if not 1 <= args.windows <= 4 or not 0 <= args.seed < 2**31:
        raise ValueError("Hard budget: 1..4 effective-batch windows and a nonnegative 31-bit seed")
    report, _ = validate_audit(args.query_audit)
    sources = report["inputs"]["sources"]
    output = Path(args.output_dir).resolve()
    protected = [Path(args.query_audit).resolve(), *[Path(row["path"]).resolve() for row in sources.values()]]
    if any(output == path or output in path.parents or path in output.parents for path in protected):
        raise ValueError("Use a separate output directory; source reports/checkpoints are read-only")
    old_checkpoint = load_trusted_torch_file(sources["old_checkpoint"]["path"])
    resume = validate_resume(old_checkpoint)
    if old_checkpoint.get("iteration") != ENDPOINTS["old"]:
        raise ValueError("Expected the complete 8ep model_0056799.pth")
    trainer = old_checkpoint.get("trainer", {})
    if trainer.get("gradient_accumulation_steps") != 2:
        raise ValueError("Expected native accumulation=2")
    del old_checkpoint
    identity = {
        "schema_version": 1,
        "query_audit": file_identity(args.query_audit),
        "sources": sources,
        "resume": resume,
        "start_iteration": START_ITERATION,
        "historical_endpoint_iteration": ENDPOINTS["new"],
        "world_size": args.num_gpus,
        "physical_global_batch": 16,
        "gradient_accumulation_steps": 2,
        "effective_global_batch": 32,
        "windows": args.windows,
        "seed": args.seed,
        "targets": {"query": sorted(QUERY_GROUPS), "bank": ["upstream_tpa"]},
        "sources_screened": list(SOURCES),
        "code": _source_code_identity(),
        "torch_version": str(torch.__version__),
    }
    signature = fingerprint(identity)
    manifest = output / "manifest.json"
    if manifest.exists():
        if load_json(manifest) != {"inputs": identity, "fingerprint": signature}:
            raise ValueError("Audit inputs/code changed; use a new output directory")
        if (output / "COMPLETE.json").exists():
            raise ValueError("Audit already complete; read report.json instead of rerunning")
        if (output / "windows.jsonl").exists() or (output / "report.json").exists():
            raise ValueError("Incomplete audit has captured output; preserve it and use a new directory")
    else:
        if output.exists() and any(output.iterdir()):
            raise ValueError("Refusing a nonempty output directory without this audit manifest")
        output.mkdir(parents=True, exist_ok=True)
        save_json(manifest, {"inputs": identity, "fingerprint": signature})
    print(
        f"[budget] {args.windows} windows x 32 images = {args.windows * 32} training-image "
        "exposures; 4 GPUs; ZERO optimizer/model updates and ZERO LVIS evaluation",
        flush=True,
    )
    return SimpleNamespace(
        report=report,
        sources=sources,
        output=output,
        identity=identity,
        signature=signature,
    )


def _target_parameters(model):
    selected = {target: [] for target in TARGETS}
    seen = set()
    for raw_name, parameter in model.named_parameters():
        name = canonical_name(raw_name)
        group = key_group(name)
        target = "query" if group in QUERY_GROUPS else "bank" if group == "upstream_tpa" else None
        if target is None:
            continue
        if id(parameter) in seen:
            raise ValueError(f"Shared target parameter appeared twice: {raw_name}")
        seen.add(id(parameter))
        selected[target].append((raw_name, parameter))
    for target in selected:
        selected[target].sort(key=lambda row: canonical_name(row[0]))
        if not selected[target] or any(not parameter.requires_grad for _, parameter in selected[target]):
            raise ValueError(f"Missing/frozen target parameter group: {target}")
    if set(id(p) for _, p in selected["query"]) & set(id(p) for _, p in selected["bank"]):
        raise ValueError("Query and bank parameter groups overlap")
    return selected


def _loss_gradients(losses, targets, accumulation):
    keys = split_loss_keys(losses)
    parameters = [parameter for target in TARGETS for _, parameter in targets[target]]
    boundaries = {"query": (0, len(targets["query"])),
                  "bank": (len(targets["query"]), len(parameters))}
    result, connected = {}, {}
    for source in SOURCES:
        loss = sum(losses[key] for key in keys[source]) / float(accumulation)
        gradients = torch.autograd.grad(
            loss,
            parameters,
            retain_graph=True,
            allow_unused=True,
        )
        result[source], connected[source] = {}, {}
        for target, (start, stop) in boundaries.items():
            parts = []
            flags = []
            for gradient, parameter in zip(gradients[start:stop], parameters[start:stop]):
                flags.append(gradient is not None)
                value = torch.zeros_like(parameter) if gradient is None else gradient.detach()
                if not torch.isfinite(value).all():
                    raise FloatingPointError(f"Nonfinite {source}->{target} gradient")
                parts.append(value.float().cpu())
            result[source][target] = parts
            connected[source][target] = flags
    return result, connected, keys


def _merge_components(total, current):
    if total is None:
        return current
    for source in SOURCES:
        for target in TARGETS:
            if len(total[source][target]) != len(current[source][target]):
                raise ValueError("Target gradient layout changed between microbatches")
            total[source][target] = [a + b for a, b in zip(total[source][target], current[source][target])]
    return total


def _merge_connections(total, current):
    if total is None:
        return current
    for source in SOURCES:
        for target in TARGETS:
            total[source][target] = [a or b for a, b in zip(total[source][target], current[source][target])]
    return total


def _allreduce_model_grads(model, bucket_numel=4_000_000):
    world = dist.get_world_size()
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    presence = torch.tensor(
        [parameter.grad is not None for parameter in parameters],
        device=next(model.parameters()).device,
        dtype=torch.uint8,
    )
    dist.all_reduce(presence, op=dist.ReduceOp.MAX)
    for present, parameter in zip(presence.tolist(), parameters):
        if present and parameter.grad is None:
            # DDP treats a locally unused but globally used parameter as a zero
            # local contribution.  Reproduce that before the manual average.
            parameter.grad = torch.zeros_like(parameter)
    pending, count = [], 0

    def flush():
        nonlocal pending, count
        if not pending:
            return
        flat = torch.cat([parameter.grad.reshape(-1) for parameter in pending])
        dist.all_reduce(flat, op=dist.ReduceOp.SUM)
        flat.div_(world)
        offset = 0
        for parameter in pending:
            length = parameter.numel()
            parameter.grad.copy_(flat[offset:offset + length].view_as(parameter))
            offset += length
        pending, count = [], 0

    for present, parameter in zip(presence.tolist(), parameters):
        if not present:
            continue
        if parameter.grad is None:
            raise ValueError("Globally used parameter still has no local gradient")
        if pending and (parameter.grad.dtype != pending[0].grad.dtype or count + parameter.numel() > bucket_numel):
            flush()
        pending.append(parameter)
        count += parameter.numel()
    flush()


def _allreduce_cpu_parts(parts, parameters, bucket_numel=4_000_000):
    if len(parts) != len(parameters):
        raise ValueError("CPU component/parameter layout differs")
    world = dist.get_world_size()
    output = [None] * len(parts)
    start = 0
    while start < len(parts):
        stop, count = start, 0
        while stop < len(parts) and (stop == start or count + parts[stop].numel() <= bucket_numel):
            count += parts[stop].numel()
            stop += 1
        device = parameters[start].device
        flat = torch.cat([part.reshape(-1) for part in parts[start:stop]]).to(device)
        dist.all_reduce(flat, op=dist.ReduceOp.SUM)
        flat.div_(world)
        flat = flat.cpu()
        offset = 0
        for index in range(start, stop):
            length = parts[index].numel()
            output[index] = flat[offset:offset + length].view_as(parts[index]).clone()
            offset += length
        start = stop
    return output


def _global_components(components, targets):
    for source in SOURCES:
        for target in TARGETS:
            parameters = [parameter for _, parameter in targets[target]]
            components[source][target] = _allreduce_cpu_parts(
                components[source][target], parameters
            )
    return components


def _global_connections(connections):
    device = torch.device("cuda", torch.cuda.current_device())
    for source in SOURCES:
        for target in TARGETS:
            flags = torch.tensor(connections[source][target], device=device, dtype=torch.uint8)
            dist.all_reduce(flags, op=dist.ReduceOp.MAX)
            connections[source][target] = [bool(value) for value in flags.tolist()]
    return connections


def _norm(parts):
    return math.sqrt(sum(float(part.double().square().sum()) for part in parts))


def _reconstruction_error(full, components):
    numerator = 0.0
    denominator = 0.0
    for index, value in enumerate(full):
        reconstructed = sum((components[source][index] for source in SOURCES), torch.zeros_like(value))
        numerator += float((reconstructed.double() - value.double()).square().sum())
        denominator += float(value.double().square().sum())
    return math.sqrt(numerator) / max(math.sqrt(denominator), 1e-30)


def _route_tpa(parts, apr_parts):
    shapes = [tuple(part.shape) for part in parts]
    total = torch.cat([part.reshape(-1) for part in parts])
    apr = torch.cat([part.reshape(-1) for part in apr_parts])
    routed, stats = route_conflicting_task_gradient(total, apr)
    output, offset = [], 0
    for shape, part in zip(shapes, parts):
        length = part.numel()
        output.append(routed[offset:offset + length].view(shape).clone())
        offset += length
    return output, {key: float(value) for key, value in stats.items()}


def _optimizer_groups(optimizer):
    result = {}
    for group in optimizer.param_groups:
        for parameter in group["params"]:
            if id(parameter) in result:
                raise ValueError("Parameter appears in multiple optimizer groups")
            result[id(parameter)] = group
    return result


def _adam_updates(parameters, gradients, optimizer, groups):
    if len(parameters) != len(gradients):
        raise ValueError("AdamW parameter/gradient layout differs")
    output = []
    for parameter, gradient in zip(parameters, gradients):
        if parameter not in optimizer.state:
            raise ValueError("Target parameter lacks restored AdamW state")
        output.append(adamw_delta(parameter, gradient.to(parameter.device),
                                  optimizer.state[parameter], groups[id(parameter)]).cpu())
    return output


def _historical_deltas(targets, new_state):
    result = {}
    for target in TARGETS:
        result[target] = []
        for name, parameter in targets[target]:
            key = name[7:] if name.startswith("module.") else name
            if key not in new_state:
                raise ValueError(f"10ep checkpoint misses target parameter: {key}")
            value = new_state[key]
            if value.shape != parameter.shape or value.dtype != parameter.dtype:
                raise ValueError(f"10ep target parameter layout differs: {key}")
            result[target].append((value.detach().cpu() - parameter.detach().cpu()).float())
    return result


def _input_record(data):
    rows = []
    for item in data:
        classes = item["instances"].gt_classes
        rows.append({
            "image_id": int(item["image_id"]),
            "image": tensor_digest(item["image"]),
            "boxes": tensor_digest(item["instances"].gt_boxes.tensor),
            "classes": tensor_digest(classes),
            "class_values": classes.tolist(),
        })
    return rows


@contextmanager
def _forward_seed(seed, device_index):
    python_state, numpy_state = random.getstate(), np.random.get_state()
    try:
        with torch.random.fork_rng(devices=[device_index]):
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)


def _analyze_window(model, optimizer, targets, historical, components, connections, max_norm):
    tpa_ids = {id(parameter) for _, parameter in targets["bank"]}
    detector = [parameter for parameter in model.parameters()
                if parameter.requires_grad and id(parameter) not in tpa_ids and parameter.grad is not None]
    tpa = [parameter for _, parameter in targets["bank"]]
    query = [parameter for _, parameter in targets["query"]]
    if any(parameter.grad is None for parameter in tpa + query):
        raise ValueError("A target parameter has no native full gradient")
    full_query = [parameter.grad.detach().float().cpu() for parameter in query]
    full_tpa_unrouted = [parameter.grad.detach().float().cpu() for parameter in tpa]
    errors = {
        "query": _reconstruction_error(full_query, {s: components[s]["query"] for s in SOURCES}),
        "bank_pre_routing": _reconstruction_error(
            full_tpa_unrouted, {s: components[s]["bank"] for s in SOURCES}
        ),
    }
    if max(errors.values()) > 2e-3:
        raise ValueError(f"Loss-source gradients do not reconstruct the full gradient: {errors}")
    routed_tpa, native_routing = _route_tpa(full_tpa_unrouted, components["apr"]["bank"])
    detector_norm = math.sqrt(sum(float(parameter.grad.detach().double().square().sum()) for parameter in detector))
    tpa_norm = _norm(routed_tpa)
    normal_coefficients = {
        "detector": clip_coefficient(detector_norm, max_norm),
        "bank": clip_coefficient(tpa_norm, max_norm),
    }
    normal_pre = {"query": full_query, "bank": routed_tpa}
    normal_clipped = {
        "query": [part * normal_coefficients["detector"] for part in full_query],
        "bank": [part * normal_coefficients["bank"] for part in routed_tpa],
    }
    groups = _optimizer_groups(optimizer)
    normal_updates = {
        target: _adam_updates([p for _, p in targets[target]], normal_clipped[target], optimizer, groups)
        for target in TARGETS
    }
    rows = {}
    for source in SOURCES:
        rows[source] = {}
        for target in TARGETS:
            source_parts = components[source][target]
            source_norm = _norm(source_parts)
            if target == "query":
                cf_pre = [full - source_part for full, source_part in zip(full_query, source_parts)]
                cf_norm = counterfactual_norm(detector_norm, full_query, cf_pre)
                cf_coefficient = clip_coefficient(cf_norm, max_norm)
                cf_routing = None
            else:
                cf_total = [full - source_part for full, source_part in zip(full_tpa_unrouted, source_parts)]
                cf_apr = ([part - source_part for part, source_part in
                           zip(components["apr"]["bank"], source_parts)]
                          if source == "apr" else components["apr"]["bank"])
                cf_pre, cf_routing = _route_tpa(cf_total, cf_apr)
                cf_norm = _norm(cf_pre)
                cf_coefficient = clip_coefficient(cf_norm, max_norm)
            cf_clipped = [part * cf_coefficient for part in cf_pre]
            cf_updates = _adam_updates(
                [parameter for _, parameter in targets[target]],
                cf_clipped,
                optimizer,
                groups,
            )
            metrics = vector_metrics(normal_updates[target], cf_updates, historical[target])
            metrics.update({
                "source_gradient_l2": source_norm,
                "connected_parameter_count": sum(connections[source][target]),
                "parameter_count": len(connections[source][target]),
                "normal_preclip_l2": detector_norm if target == "query" else tpa_norm,
                "counterfactual_preclip_l2": cf_norm,
                "normal_clip_coefficient": normal_coefficients["detector" if target == "query" else "bank"],
                "counterfactual_clip_coefficient": cf_coefficient,
                "counterfactual_routing": cf_routing,
            })
            rows[source][target] = metrics
    return {
        "loss_reconstruction_relative_l2": errors,
        "native_routing": native_routing,
        "native_preclip_l2": {"detector": detector_norm, "bank": tpa_norm},
        "native_clip_coefficients": normal_coefficients,
        "native_updates": {
            target: vector_metrics(normal_updates[target], normal_updates[target], historical[target])
            for target in TARGETS
        },
        "sources": rows,
    }


def worker(args, identity, signature):
    from detectron2.config import LazyConfig, instantiate
    from detectron2.data import MetadataCatalog
    from detectron2.utils.events import EventStorage
    from tools.train_net import _broadcast_tpa_buffers, _maybe_convert_syncbn

    if not torch.cuda.is_available() or not dist.is_initialized() or dist.get_world_size() != 4:
        raise ValueError("Expected an initialized four-rank CUDA process group")
    rank = dist.get_rank()
    device_index = torch.cuda.current_device()
    device = torch.device("cuda", device_index)
    random.seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)
    torch.cuda.manual_seed(args.seed + rank)
    torch.backends.cudnn.benchmark = False

    sources = identity["sources"]
    cfg = LazyConfig.load(sources["config_file"]["path"])
    cfg = LazyConfig.apply_overrides(cfg, [
        "model.classifier.tpa_slot_prior_strength=0.2",
        "model.classifier.tpa_prototype_mode_strength=0.0",
        "model.tpa_eval_mode_scale=1.0",
        "model.alpha=0.0", "model.beta=0.3", "model.novel_scale=3.0",
        "model.device=cuda", "train.device=cuda",
    ])
    if (cfg.dataloader.train.dataset.names != "lvis_v1_train_norare"
            or not cfg.dataloader.train.mapper.is_train):
        raise ValueError("Expected native LVIS train_norare mapper")
    if (not cfg.train.tpa_conflict_projection or not cfg.train.separate_tpa_grad_clip
            or int(cfg.train.gradient_accumulation_steps) != 2):
        raise ValueError("Expected native APR routing, separate clipping, accumulation=2")
    if (not cfg.train.clip_grad.enabled
            or dict(cfg.train.clip_grad.params) != {"max_norm": .5, "norm_type": 2}):
        raise ValueError("Expected the native 0.5 L2 clipping policy")
    cfg.dataloader.train.total_batch_size = 16
    cfg.dataloader.train.num_workers = 0

    model = instantiate(cfg.model).to(device)
    model = _maybe_convert_syncbn(model, enable=getattr(cfg.train, "sync_batchnorm", True))
    checkpoint = load_trusted_torch_file(sources["old_checkpoint"]["path"])
    old_state = endpoint_state(checkpoint, ENDPOINTS["old"])
    model.load_state_dict(old_state, strict=True)
    model.train()
    model.tpa_advance_step = False
    _broadcast_tpa_buffers(model)
    cfg.optimizer.params.model = model
    optimizer = instantiate(cfg.optimizer)
    optimizer.load_state_dict(checkpoint["trainer"]["optimizer"])
    if state_digest(optimizer.state_dict()) != identity["resume"]["optimizer"]:
        raise ValueError("Restored optimizer state differs from the locked 8ep checkpoint")
    del checkpoint, old_state

    targets = _target_parameters(model)
    target_inventory = {
        target: [{"name": name, "canonical_name": canonical_name(name),
                  "shape": list(parameter.shape), "numel": parameter.numel()}
                 for name, parameter in targets[target]]
        for target in TARGETS
    }
    new_checkpoint = load_trusted_torch_file(sources["new_checkpoint"]["path"])
    new_state = endpoint_state(new_checkpoint, ENDPOINTS["new"])
    historical = _historical_deltas(targets, new_state)
    del new_checkpoint, new_state
    gc.collect()

    loader = iter(instantiate(cfg.dataloader.train))
    annotations = file_identity(MetadataCatalog.get("lvis_v1_train_norare").json_file)
    versions = {name: (id(value), value._version) for name, value in model.named_parameters()}
    sampled = {}
    original_filter = model.filter_content_info

    def record_filter(data):
        indices, result = original_filter(data)
        sampled["content_inds"] = indices.detach().cpu().tolist()
        return indices, result

    model.filter_content_info = record_filter
    windows = []
    try:
        with EventStorage(start_iter=START_ITERATION):
            for window in range(args.windows):
                optimizer.zero_grad()
                components = None
                connections = None
                micros = []
                for micro in range(2):
                    data = next(loader)
                    if len(data) != 4:
                        raise ValueError("Expected four images per rank/microbatch")
                    inputs = _input_record(data)
                    if any(c < 0 or c >= len(model.novel_idx) or bool(model.novel_idx[c])
                           for row in inputs for c in row["class_values"]):
                        raise ValueError("Rare/invalid GT entered train_norare audit")
                    sampled.clear()
                    forward_seed = args.seed + 100000 + window * 100 + micro * 10 + rank
                    with _forward_seed(forward_seed, device_index):
                        losses = model(data)
                    current, current_connections, loss_keys = _loss_gradients(losses, targets, 2)
                    components = _merge_components(components, current)
                    connections = _merge_connections(connections, current_connections)
                    total = sum(value for key, value in losses.items() if key.startswith("loss")) / 2.0
                    total.backward()
                    if "content_inds" not in sampled:
                        raise ValueError("FedLoss subset was not captured")
                    micros.append({
                        "micro": micro,
                        "rank": rank,
                        "forward_seed": forward_seed,
                        "inputs": inputs,
                        "content_inds": sampled["content_inds"],
                        "loss_keys": loss_keys,
                        "losses": {key: float(value.detach()) for key, value in losses.items()
                                   if key.startswith("loss")},
                    })
                    del current, current_connections, losses, total, data
                    print(f"[capture rank={rank}] window={window+1}/{args.windows} micro={micro+1}/2", flush=True)
                _allreduce_model_grads(model)
                components = _global_components(components, targets)
                connections = _global_connections(connections)
                analyzed = _analyze_window(
                    model,
                    optimizer,
                    targets,
                    historical,
                    components,
                    connections,
                    float(cfg.train.clip_grad.params.max_norm),
                )
                all_micros = [None for _ in range(dist.get_world_size())]
                dist.all_gather_object(all_micros, micros)
                if rank == 0:
                    flat = [row for rank_rows in all_micros for row in rank_rows]
                    for micro in range(2):
                        subsets = {tuple(row["content_inds"]) for row in flat if row["micro"] == micro}
                        if len(subsets) != 1:
                            raise ValueError("FedLoss subsets differ across ranks")
                    analyzed.update({"window": window, "microbatches": flat})
                    windows.append(analyzed)
                    _json_line(Path(args.output_dir) / "windows.jsonl", analyzed)
                    print(f"[window {window+1}] routing={analyzed['native_routing']} "
                          f"clip={analyzed['native_clip_coefficients']}", flush=True)
                optimizer.zero_grad()
                dist.barrier()
    finally:
        model.filter_content_info = original_filter

    if any((id(value), value._version) != versions[name] for name, value in model.named_parameters()):
        raise ValueError("A model parameter changed during the read-only audit")
    current_optimizer = state_digest(optimizer.state_dict())
    if current_optimizer != identity["resume"]["optimizer"]:
        raise ValueError("Optimizer state changed although optimizer.step was forbidden")
    if rank == 0:
        report = {
            "complete": True,
            "fingerprint": signature,
            "inputs": identity,
            "train_annotations": annotations,
            "target_inventory": target_inventory,
            "windows": windows,
            "screening_ranking": summarize_screen(windows),
            "optimizer_state_unchanged": True,
            "model_parameters_unchanged": True,
            "optimizer_steps": 0,
            "training_updates": 0,
            "scope": [
                "Finite AdamW counterfactuals use restored 8ep moments, native effective-batch accumulation, APR conflict routing and separate clipping.",
                "The query target is terminal only: query_content + decoder_core + final_projection; it is not the entire backbone/encoder feature path.",
                "Each counterfactual removes one loss source only from one target group; other gradients and AdamW moments/weight decay remain.",
                "Loss removal recomputes the affected detector/TPA clipping coefficient; clipping-mediated changes outside the reported target are not summarized.",
                "The finite unscaled gradients assume no AMP overflow; GradScaler state is validated but never advanced.",
                "Alignment with the observed 8ep-to-10ep parameter delta is a screening association, not evidence that the source caused APr decline.",
                "No validation image, AP evaluator, optimizer.step, checkpoint write or training continuation is used.",
                "A source must still pass a paired training intervention and full LVIS evaluation before any causal claim.",
            ],
        }
        save_json(Path(args.output_dir) / "report.json", report)
        save_json(Path(args.output_dir) / "COMPLETE.json", {
            "fingerprint": signature,
            "report": file_identity(Path(args.output_dir) / "report.json"),
        })
        print("\n=== Optimizer-aware query/bank loss-source screen ===", flush=True)
        for row in report["screening_ranking"]:
            print(row, flush=True)
        print(f"[save] {Path(args.output_dir) / 'report.json'}", flush=True)
    dist.barrier()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query-audit", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-gpus", type=int, default=4)
    parser.add_argument("--windows", type=int, default=2)
    parser.add_argument("--seed", type=int, default=424242)
    parser.add_argument("--dist-url", default="auto")
    return parser.parse_args()


def main():
    args = parse_args()
    ctx = prepare(args)
    from detectron2.engine import launch
    launch(
        worker,
        args.num_gpus,
        num_machines=1,
        machine_rank=0,
        dist_url=args.dist_url,
        args=(args, ctx.identity, ctx.signature),
    )


if __name__ == "__main__":
    main()
