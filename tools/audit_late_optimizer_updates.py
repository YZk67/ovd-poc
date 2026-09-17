#!/usr/bin/env python3
"""Bounded 10ep actual AdamW intervention: capture on four GPUs, evaluate on one.

Independent fresh training windows, each rooted at iteration 70999. This is
NOT a replay of the unavailable historical sampler/RNG sequence or 14k updates.
No checkpoint is written, no full LVIS evaluation, no persistent training.
"""
from __future__ import annotations

import argparse
import gc
import math
from pathlib import Path
import random
import shutil
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F

from lami_dino.checkpoint_init import load_trusted_torch_file
from tools.analyze_rare_query_readout import checked_identity
from tools.audit_pairing_optimizer_sources import (
    _allreduce_model_grads, _allreduce_cpu_parts, _forward_seed, _input_record,
)
from tools.compare_rare_pr_reports import file_identity, load_json, save_json
from tools.decoder_aux_ablation_ops import state_digest, validate_resume
from tools.diagnose_detector_tpa_pairing import (
    check_replay, classifier_options, input_asset_hashes, locked_manifest, save_tensor_file,
)
from tools.diagnose_rare_fp_regions import model_code_hash, read_manifest
from tools.evaluate_decoder_rollback import endpoint_state
from tools.late_optimizer_audit_ops import (
    ARMS, SOURCES, START, actual_step, cpu_copy, grouped_losses, model_snapshot,
    observable, restored_branch, restore_model, summarize, validate_gradients,
)
from tools.query_path_update_ops import clear_eval_caches
from tools.rare_gt_query_ops import analyze_queries, selected_predictions, verify_predictions
from tools.rare_region_pairing_ops import replay_sample
from tools.trace_rare_gt_queries import IMAGE_ID, GT_ID, PROTOCOL, load_cache, validate_checkpoint


def prepare(args):
    if not 1 <= args.windows <= 4 or args.num_gpus != 4 or not 0 <= args.seed < 2**31:
        raise ValueError("Hard budget: 1..4 independent windows, original four-rank topology, 31-bit seed")
    if args.cpu_threads < 1:
        raise ValueError("--cpu-threads must be positive")
    torch.set_num_threads(args.cpu_threads)
    print("[preflight] authenticate timeline, full 10ep checkpoint, optimizer and cached native image", flush=True)
    source = load_json(args.timeline)
    if (source.get("complete") is not True or source["gt"]["id"] != GT_ID
            or source["image"]["id"] != IMAGE_ID or source["gt"]["category_id"] != 920
            or any(source["protocol"].get(k) != v for k, v in PROTOCOL.items())):
        raise ValueError("Requires the completed native scarecrow 8/10/12ep timeline")
    checkpoint_id = checked_identity(source["inputs"]["checkpoints"]["10"])
    annotation_id = checked_identity(source["inputs"]["annotations"])
    config_id = checked_identity(source["inputs"]["config"])
    if model_code_hash(config_id["path"]) != source["protocol"]["code_sha256"]:
        raise ValueError("Model/config code changed since native timeline; do not mix protocols")
    output = Path(args.output_dir).resolve()
    protected = [Path(args.timeline).resolve(), *(Path(r["path"]).resolve()
                  for r in (checkpoint_id, annotation_id, config_id))]
    protected.extend(Path(source["stages"]["10"]["cache"][k]["path"]).resolve()
                     for k in ("manifest", "sample", "bank"))
    if any(output == p or output in p.parents or p in output.parents for p in protected):
        raise ValueError("Use a separate output directory; source files are read-only")
    checkpoint = load_trusted_torch_file(checkpoint_id["path"])
    validate_checkpoint(checkpoint, "10ep", expected_iteration=START-1)
    resume = validate_resume(checkpoint, start=START)
    print(f"[resume] iteration={START-1}; horizon=85200; accumulation=2; complete AdamW/scaler/scheduler", flush=True)
    del checkpoint
    dataset = load_json(annotation_id["path"])
    if ([r for r in dataset["annotations"] if r["id"] == GT_ID] != [source["gt"]]
            or [r for r in dataset["images"] if r["id"] == IMAGE_ID] != [source["image"]]):
        raise ValueError("Target GT/image differs from original annotations")
    cache = source["stages"]["10"]["cache"]
    for key in ("manifest", "sample", "bank"):
        checked_identity(cache[key])
    manifest = read_manifest(cache["manifest"]["path"])
    if manifest["inputs"][cache["source_label"]+"_sha256"] != checkpoint_id["sha256"]:
        raise ValueError("Native baseline cache is not from the full 10ep checkpoint")
    sample, bank, _ = load_cache(Path(cache["manifest"]["path"]), cache["source_label"],
                                "10ep", dataset, source["image"], expected_iteration=START-1)
    del sample, bank
    asset_args = SimpleNamespace(config_file=config_id["path"], phase="dump")
    if input_asset_hashes(asset_args, output) != source["protocol"]["asset_sha256"]:
        raise ValueError("Text/CLIP/config assets changed since the timeline")
    code = {p: file_identity(ROOT / p)["sha256"] for p in (
        "tools/audit_late_optimizer_updates.py", "tools/late_optimizer_audit_ops.py",
        "tools/audit_pairing_optimizer_sources.py", "tools/decoder_aux_ablation_ops.py",
        "tools/pairing_optimizer_source_ops.py", "tools/query_path_update_ops.py",
        "tools/train_net.py", "configs/common/lvis_schedule.py", "configs/common/optim.py")}
    inputs = dict(schema_version=1, timeline=file_identity(args.timeline), checkpoint=checkpoint_id,
                  annotations=annotation_id, config=config_id, cache=cache, resume=resume,
                  windows=args.windows, seed=args.seed, world_size=4, accumulation=2,
                  physical_batch=16, effective_batch=32, iteration=START, code=code,
                  torch_version=str(torch.__version__), cpu_threads=args.cpu_threads)
    signature = locked_manifest(output / "manifest.json", inputs)
    print(f"[budget] {args.windows} independent effective batches = {32*args.windows} training image exposures; "
          f"<= {len(ARMS)*args.windows} temporary AdamW steps; "
          f"<= {1+(1+len(ARMS))*args.windows} single-image evaluation forwards; "
          "persisted training updates=0, full-validation inference=0", flush=True)
    return inputs, signature


def load_config(inputs):
    from detectron2.config import LazyConfig
    cfg = LazyConfig.load(inputs["config"]["path"])
    options = classifier_options(SimpleNamespace(**PROTOCOL, device="cuda"), "old")
    cfg = LazyConfig.apply_overrides(cfg, options + ["model.classifier.tpa_slot_prior_strength=0.2"])
    if (cfg.dataloader.train.dataset.names != "lvis_v1_train_norare"
            or not cfg.dataloader.train.mapper.is_train or not cfg.train.amp.enabled
            or not cfg.train.tpa_conflict_projection or not cfg.train.separate_tpa_grad_clip
            or int(cfg.train.gradient_accumulation_steps) != 2
            or int(cfg.train.lr_scheduler_max_iter) != 85200
            or not cfg.train.sync_batchnorm or not cfg.train.clip_grad.enabled
            or dict(cfg.train.clip_grad.params) != {"max_norm": .5, "norm_type": 2}):
        raise ValueError("Training config differs from native norare/AMP/accumulation/clip/scheduler protocol")
    cfg.dataloader.train.total_batch_size = 16
    cfg.dataloader.train.num_workers = 0
    return cfg


def loaded_model(cfg, checkpoint, *, distributed):
    from detectron2.config import instantiate
    from tools.train_net import _maybe_convert_syncbn, _broadcast_tpa_buffers
    model = instantiate(cfg.model).cuda()
    if distributed:
        model = _maybe_convert_syncbn(model, enable=True)
    model.load_state_dict(endpoint_state(checkpoint, START-1), strict=True)
    if distributed:
        _broadcast_tpa_buffers(model)
    cfg.optimizer.params.model = model
    optimizer = instantiate(cfg.optimizer)
    if not isinstance(optimizer, torch.optim.AdamW):
        raise ValueError("Expected actual native AdamW optimizer")
    return model, optimizer


def native_state(cfg, model, optimizer, checkpoint, inputs):
    from detectron2.config import instantiate
    from detectron2.solver import LRMultiplier
    from tools.train_net import Trainer
    saved = checkpoint["trainer"]
    scheduler = LRMultiplier(optimizer, instantiate(cfg.lr_multiplier), max_iter=85200)
    optimizer.load_state_dict(cpu_copy(saved["optimizer"]))
    scheduler.load_state_dict(cpu_copy(saved["hooks"]["LRScheduler"]))
    scaler = torch.cuda.amp.GradScaler()
    scaler.load_state_dict(cpu_copy(saved["grad_scaler"]))
    if (state_digest(optimizer.state_dict()) != inputs["resume"]["optimizer"]
            or state_digest(scheduler.state_dict()) != inputs["resume"]["scheduler"]
            or state_digest(scaler.state_dict()) != inputs["resume"]["scaler"]
            or any(not math.isclose(a, b["lr"], rel_tol=1e-7) for a, b in
                   zip(scheduler.get_lr(), optimizer.param_groups))):
        raise ValueError("Full AdamW/scaler/scheduler restore or actual current LR mismatch")
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    opt_parameters = [p for g in optimizer.param_groups for p in g["params"] if p.requires_grad]
    if len({id(p) for p in opt_parameters}) != len(opt_parameters) or {id(p) for p in opt_parameters} != {id(p) for _, p in named}:
        raise ValueError("Optimizer does not cover every trainable parameter exactly once")
    for name, p in named:
        state = optimizer.state.get(p)
        if state:
            for key in ("exp_avg", "exp_avg_sq"):
                v = state.get(key)
                if v is None or v.shape != p.shape or v.dtype != p.dtype or not torch.isfinite(v).all():
                    raise ValueError(f"Invalid restored AdamW moment: {name}/{key}")
    trainer = Trainer(model, [], optimizer, amp=True, grad_scaler=scaler,
                      clip_grad_params={"max_norm": .5, "norm_type": 2}, separate_tpa_grad_clip=True,
                      tpa_conflict_projection=True, gradient_accumulation_steps=2, lr_scheduler_max_iter=85200)
    trainer.iter = START
    return trainer, scheduler, named


def artifact_paths(output, window):
    return output / f"window_{window:02d}.pt", output / f"window_{window:02d}.json"


def read_artifact(output, window, signature):
    path, meta = artifact_paths(output, window)
    if not path.exists() and not meta.exists():
        return None
    if not path.exists() or not meta.exists():
        raise ValueError("Incomplete window artifact; use a new output directory, do not silently recapture")
    record = load_json(meta)
    if record["fingerprint"] != signature or record["window"] != window:
        raise ValueError("Window identity mismatch")
    checked_identity(record["artifact"])
    return record


def capture_worker(args, inputs, signature):
    from detectron2.config import instantiate
    from detectron2.data import MetadataCatalog
    from detectron2.utils.events import EventStorage
    if not dist.is_initialized() or dist.get_world_size() != 4:
        raise ValueError("Gradient capture requires original four-rank process group")
    torch.set_num_threads(args.cpu_threads)
    rank, device = dist.get_rank(), torch.cuda.current_device()
    random.seed(args.seed+rank)
    np.random.seed(args.seed+rank)
    torch.manual_seed(args.seed+rank)
    torch.backends.cudnn.benchmark = False
    cfg = load_config(inputs)
    checkpoint = load_trusted_torch_file(inputs["checkpoint"]["path"])
    model, optimizer = loaded_model(cfg, checkpoint, distributed=True)
    snapshot = model_snapshot(model)
    parameter_hash = state_digest(dict(model.named_parameters()))
    loader = iter(instantiate(cfg.dataloader.train))
    train_annotation = file_identity(MetadataCatalog.get("lvis_v1_train_norare").json_file)
    if train_annotation["sha256"] == inputs["annotations"]["sha256"]:
        raise ValueError("Validation annotations must never enter the training audit")
    sampled = {}
    original_filter = model.filter_content_info
    def record_filter(data):
        indices, info = original_filter(data)
        sampled["content_inds"] = indices.detach().cpu().tolist()
        return indices, info
    model.filter_content_info = record_filter
    output = Path(args.output_dir)
    for window in range(args.windows):
        # Advance the deterministic loader even for previously completed windows.
        with _forward_seed(args.seed + 1000*window + rank, device):
            batches = [next(loader), next(loader)]
        if read_artifact(output, window, signature):
            print(f"[reuse rank={rank}] window={window}; no training forward", flush=True)
            continue
        restore_model(model, snapshot)
        model.train()
        trainer, _, named = native_state(cfg, model, optimizer, checkpoint, inputs)
        parameters = [p for _, p in named]
        estimate = sum(p.numel()*p.element_size() for p in parameters) * 6
        if shutil.disk_usage(output).free < estimate + 512*1024**2:
            raise OSError("Insufficient space for bounded gradient artifacts; no files deleted")
        components = {s: [torch.zeros_like(p, device="cpu") for p in parameters] for s in SOURCES}
        apr_reference = [torch.zeros_like(p, device="cpu") for p in parameters]
        index = {id(p): i for i, p in enumerate(parameters)}
        micros = []
        optimizer.zero_grad(set_to_none=True)
        with EventStorage(start_iter=START):
            for micro, data in enumerate(batches):
                if len(data) != 4:
                    raise ValueError("Expected exactly four images/rank/microbatch")
                input_rows = _input_record(data)
                if any(c < 0 or c >= len(model.novel_idx) or bool(model.novel_idx[c])
                       for row in input_rows for c in row["class_values"]):
                    raise ValueError("Rare GT must not enter train_norare")
                sampled.clear()
                trainer._set_tpa_step_advance(micro == 1)
                seed = args.seed + 100000 + window*100 + micro*10 + rank
                with _forward_seed(seed, device):
                    losses = model(data)  # Native Trainer does NOT autocast this outer forward.
                groups = grouped_losses(losses)
                apr = trainer._compute_apr_gradients(losses, gradient_scale=.5)
                if apr is None:
                    raise ValueError("Native APR routing reference unexpectedly disabled")
                for p, value in zip(trainer._get_tpa().parameters(), apr):
                    if value is not None:
                        apr_reference[index[id(p)]].add_(value.detach().cpu())
                del apr
                for source, keys in groups.items():
                    with torch.cuda.amp.autocast(enabled=True):
                        loss = sum(losses[k] for k in keys) / 2
                    if loss.requires_grad:
                        grads = torch.autograd.grad(trainer.grad_scaler.scale(loss), parameters,
                                                    retain_graph=True, allow_unused=True)
                        for i, value in enumerate(grads):
                            if value is not None:
                                components[source][i].add_((value.detach() / trainer.grad_scaler.get_scale()).cpu())
                        del grads
                with torch.cuda.amp.autocast(enabled=True):
                    total = sum(v for k, v in losses.items() if k.startswith("loss")) / 2
                trainer.grad_scaler.scale(total).backward()
                if "content_inds" not in sampled:
                    raise ValueError("FedLoss sampling was not captured")
                micros.append(dict(rank=rank, micro=micro, seed=seed, inputs=input_rows,
                                   content_inds=sampled["content_inds"], loss_keys=groups,
                                   losses={k: float(v.detach()) for k, v in losses.items() if k.startswith("loss")}))
                del losses, loss, total
                print(f"[capture rank={rank}] window={window+1}/{args.windows} micro={micro+1}/2", flush=True)
            trainer._set_tpa_step_advance(True)
            _allreduce_model_grads(model)  # Scaled accumulated gradients, as DDP before unscale.
            trainer.grad_scaler.unscale_(optimizer)
            for s in SOURCES:
                components[s] = _allreduce_cpu_parts(components[s], parameters)
            apr_reference = _allreduce_cpu_parts(apr_reference, parameters)
        payload = dict(fingerprint=signature, window=window, names=[n for n, _ in named],
                       present=[p.grad is not None for p in parameters], components=components,
                       apr_reference=apr_reference,
                       full=[cpu_copy(p.grad) if p.grad is not None else torch.zeros_like(p, device="cpu")
                             for p in parameters])
        reconstruction = validate_gradients(payload, named)
        parameter_names = {id(p): n for n, p in model.named_parameters()}
        payload["optimizer_layout"] = [[parameter_names[id(p)] for p in group["params"]]
                                        for group in optimizer.param_groups]
        metadata = [None] * 4
        dist.all_gather_object(metadata, micros)
        if any(len({tuple(rows[m]["content_inds"]) for rows in metadata}) != 1 for m in range(2)):
            raise ValueError("FedLoss classes differ across ranks")
        if rank == 0:
            payload["buffer_changes"] = {k: cpu_copy(v) for k, v in model.named_buffers()
                                          if not torch.equal(v.detach().cpu(), snapshot["buffers"][k])}
            path, meta = artifact_paths(output, window)
            save_tensor_file(path, payload)
            save_json(meta, dict(fingerprint=signature, window=window, artifact=file_identity(path),
                                 gradient_reconstruction_relative_l2=reconstruction,
                                 train_annotations=train_annotation, microbatches=metadata,
                                 changed_buffers=list(payload["buffer_changes"])))
            print(f"[captured] window={window}; AMP gradient reconstruction error={reconstruction:.3g}", flush=True)
        optimizer.zero_grad(set_to_none=True)
        del payload, components, apr_reference, batches, trainer
        gc.collect()
        dist.barrier()
    model.filter_content_info = original_filter
    restore_model(model, snapshot)
    if state_digest(dict(model.named_parameters())) != parameter_hash:
        raise ValueError("Capture unexpectedly modified model parameters")
    if state_digest(optimizer.state_dict()) != inputs["resume"]["optimizer"]:
        raise ValueError("Capture unexpectedly modified optimizer state")


def single_image_inputs(cfg, inputs):
    from detectron2.config import instantiate
    from detectron2.data import MetadataCatalog, get_detection_dataset_dicts, build_detection_test_loader
    mapper = cfg.dataloader.test.mapper
    aug = mapper.augmentation
    if (mapper.is_train or mapper.augmentation_with_crop is not None or len(aug) != 1
            or "ResizeShortestEdge" not in str(aug[0]._target_)):
        raise ValueError("Requires native deterministic resize-only validation preprocessing")
    edges = aug[0].short_edge_length
    if not isinstance(edges, int) and len(set(edges)) != 1:
        raise ValueError("Validation mapper randomly changes resolution")
    names = cfg.dataloader.test.dataset.names
    name = names if isinstance(names, str) else names[0]
    if not isinstance(names, str) and len(names) != 1:
        raise ValueError("Requires exactly one validation dataset")
    records = get_detection_dataset_dicts(names=name, filter_empty=False)
    if file_identity(MetadataCatalog.get(name).json_file)["sha256"] != inputs["annotations"]["sha256"]:
        raise ValueError("Configured validation annotations differ")
    categories = sorted(c["id"] for c in load_json(inputs["annotations"]["path"])["categories"])
    expected_mapping = {c: i for i, c in enumerate(categories)}
    if getattr(MetadataCatalog.get(name), "thing_dataset_id_to_contiguous_id", expected_mapping) != expected_mapping:
        raise ValueError("Validation category ordering differs from cached full vocabulary")
    chosen = [r for r in records if r["image_id"] == IMAGE_ID]
    if len(chosen) != 1:
        raise ValueError("Target image absent/duplicated")
    loader = build_detection_test_loader(dataset=chosen, mapper=instantiate(mapper), num_workers=0)
    return next(iter(loader))


@torch.no_grad()
def native_observation(model, capture, data, source, dataset):
    from tools.diagnose_tpa_usage import box_cxcywh_to_xyxy
    model.eval()
    clear_eval_caches(model)
    capture.clear()
    model(data)
    logits = capture["detector_logits"][0].float().cpu()
    scores = capture["final_scores"][0].float().cpu()
    roi = capture["roi_features"][0].float()
    clip = (roi @ model.vlm_content_query_embedding.float().t() * float(model.vlm_temperature)).cpu()
    boxes = box_cxcywh_to_xyxy(capture["query_boxes"][0].float()).clamp(0, 1).cpu()
    boxes *= boxes.new_tensor([source["image"]["width"], source["image"]["height"]]*2)
    top = scores.flatten().topk(300)
    selected = torch.zeros(scores.numel(), dtype=torch.bool)
    selected[top.indices] = True
    replay = dict(boxes=boxes, scores=scores, det_logits=logits, clip_logits=clip,
                  det_logp=F.logsigmoid(logits), clip_logp=F.log_softmax(clip, dim=-1),
                  log_scores=scores.clamp_min(1e-30).log(), cutoff=float(top.values[-1]),
                  selected=selected.reshape_as(scores), category_ids=sorted(c["id"] for c in dataset["categories"]),
                  novel_mask=model.novel_idx.bool().cpu())
    if logits.shape != (900, 1203) or not torch.isfinite(clip).all():
        raise ValueError("Native observation has invalid query/vocabulary/CLIP outputs")
    analysis = analyze_queries(replay, source["gt"], {c["id"]: c for c in dataset["categories"]})
    capture.clear()
    return analysis, replay


def evaluate(args, inputs, signature):
    from detectron2.utils.events import EventStorage
    from tools.diagnose_tpa_usage import install_capture_hooks
    if dist.is_initialized():
        raise ValueError("Evaluation starts AFTER four-rank capture exits; no replicated validation forwards")
    output = Path(args.output_dir)
    records = [read_artifact(output, w, signature) for w in range(args.windows)]
    if any(r is None for r in records):
        raise FileNotFoundError("Capture incomplete; --phase capture first. No optimizer step started")
    source = load_json(inputs["timeline"]["path"])
    dataset = load_json(inputs["annotations"]["path"])
    checkpoint = load_trusted_torch_file(inputs["checkpoint"]["path"])
    cfg = load_config(inputs)
    print("[evaluate] four-rank capture finished; restoring one-GPU temporary replica", flush=True)
    model, optimizer = loaded_model(cfg, checkpoint, distributed=False)
    snapshot = model_snapshot(model)
    native_state(cfg, model, optimizer, checkpoint, inputs)
    data = single_image_inputs(cfg, inputs)
    capture = {}
    install_capture_hooks(model, capture)
    bank = load_trusted_torch_file(inputs["cache"]["bank"]["path"])
    sample = load_trusted_torch_file(inputs["cache"]["sample"]["path"])
    cached = replay_sample(sample, bank, PROTOCOL, device="cpu")
    print("[baseline] unchanged 10ep full native forward; checking all queries and top-300 before updates", flush=True)
    baseline, observed = native_observation(model, capture, data, source, dataset)
    baseline_check = check_replay(observed["det_logits"], observed["scores"],
                                  cached["det_logits"], cached["scores"], max_dets=300)
    prediction_check = verify_predictions(selected_predictions(observed, IMAGE_ID), selected_predictions(cached, IMAGE_ID))
    max_box_error = float((observed["boxes"] - cached["boxes"]).abs().max())
    if max_box_error > .05:
        raise ValueError("Unupdated full 10ep forward does not reproduce ALL native query boxes")
    report = dict(complete=False, fingerprint=signature, inputs=inputs, image=source["image"], gt=source["gt"],
                  runtime=dict(torch=str(torch.__version__), cuda=torch.version.cuda,
                               cudnn=torch.backends.cudnn.version(), evaluation_gpu=torch.cuda.get_device_name(0)),
                  baseline=baseline, baseline_replay=baseline_check, baseline_predictions=prediction_check,
                  baseline_max_box_error=max_box_error, windows=[], temporary_steps=0, eval_forwards=1,
                  persisted_training_updates=0, full_validation_inference=0,
                  limitations=["Fresh fixed training windows, NOT historical sampler/RNG replay.",
                               "Every window starts at full iteration70999; windows are not a training trajectory.",
                               "Four-rank gradient averaging reproduces DDP averaging, not exact NCCL bucket rounding.",
                               "Coarse loss removal retains moments/decay and recomputes routing/clipping globally.",
                               "One selected GT: local conditional intervention, neither global AP nor historical cause."])
    save_json(output / "report.json", report)
    for record in records:
        payload = load_trusted_torch_file(record["artifact"]["path"])
        window = dict(window=record["window"], capture=record, arms={})
        with restored_branch(model, optimizer, snapshot, checkpoint["trainer"]["optimizer"],
                             buffer_changes=payload["buffer_changes"]):
            window["buffer_only"], _ = native_observation(model, capture, data, source, dataset)
            report["eval_forwards"] += 1
        # Check that residual batch-normalization/buffer drift is not called a loss-gradient effect.
        for arm in ARMS:
            with restored_branch(model, optimizer, snapshot, checkpoint["trainer"]["optimizer"],
                                 buffer_changes=payload["buffer_changes"]):
                trainer, scheduler, named = native_state(cfg, model, optimizer, checkpoint, inputs)
                validate_gradients(payload, named)
                parameter_names = {id(p): n for n, p in model.named_parameters()}
                if [[parameter_names[id(p)] for p in g["params"]] for g in optimizer.param_groups] != payload["optimizer_layout"]:
                    raise ValueError("Capture/evaluation optimizer parameter groups differ")
                with EventStorage(start_iter=START):
                    step = actual_step(trainer, named, payload, arm)
                scheduler.step()
                step["scheduler_after"] = scheduler.state_dict()
                step["next_lrs"] = [g["lr"] for g in optimizer.param_groups]
                analysis, _ = native_observation(model, capture, data, source, dataset)
                window["arms"][arm] = dict(step=step, analysis=analysis)
                report["temporary_steps"] += 1
                report["eval_forwards"] += 1
                print(f"[actual window={record['window']} arm={arm}] {observable(analysis)}", flush=True)
        report["windows"].append(window)
        save_json(output / "report.json", report)
        del payload
    if (state_digest(model_snapshot(model)) != state_digest(snapshot)
            or state_digest(optimizer.state_dict()) != inputs["resume"]["optimizer"]):
        raise ValueError("In-memory model/optimizer failed final restoration")
    checked_identity(inputs["checkpoint"])
    report.update(complete=True, decision=summarize(report["windows"]), source_checkpoint_unchanged=True,
                  replica_restored=True)
    save_json(output / "report.json", report)
    print("\n=== Late actual optimizer update audit ===", flush=True)
    print(report["decision"], flush=True)
    print(f"[save] {output / 'report.json'}", flush=True)


def run(args):
    inputs, signature = prepare(args)
    if args.prepare_only:
        print("[prepared] CPU identity/full-state checks only; no training/evaluation forward", flush=True)
        return
    if not torch.cuda.is_available():
        raise RuntimeError("Use the server lami environment with CUDA/detrex")
    output = Path(args.output_dir)
    completed = output / "report.json"
    if completed.exists():
        previous = load_json(completed)
        if previous.get("complete") and previous.get("fingerprint") == signature:
            print(f"[reuse] completed audit: {completed}; no work repeated", flush=True)
            return
    pending = [w for w in range(args.windows) if not read_artifact(output, w, signature)]
    if args.phase in ("all", "capture") and pending:
        if torch.cuda.device_count() < 4:
            raise ValueError("Capture requires four visible GPUs for the original batch topology")
        from detectron2.engine import launch
        launch(capture_worker, num_gpus_per_machine=4, dist_url="auto", args=(args, inputs, signature))
    if args.phase in ("all", "evaluate"):
        torch.cuda.set_device(0)
        evaluate(args, inputs, signature)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeline", default="/root/autodl-tmp/no_radius_scarecrow_timeline/report.json")
    parser.add_argument("--output-dir", default="/root/autodl-tmp/no_radius_10ep_actual_updates")
    parser.add_argument("--windows", type=int, default=2)
    parser.add_argument("--num-gpus", type=int, default=4, help="Fixed capture topology, not evaluation GPU count")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu-threads", type=int, default=2)
    parser.add_argument("--phase", choices=("all", "capture", "evaluate"), default="all")
    parser.add_argument("--prepare-only", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
