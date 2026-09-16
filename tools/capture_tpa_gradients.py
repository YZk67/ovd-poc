"""Bounded native training-loss probes. No optimizer, scheduler, or checkpoint save."""

from __future__ import annotations

from collections import defaultdict
from pprint import pformat
import random

import numpy as np
import torch

from lami_dino.checkpoint_init import load_trusted_torch_file
from tools.compare_rare_pr_reports import file_identity
from tools.tpa_geometry_audit_ops import WEIGHTS, tensor_digest
from tools.tpa_gradient_audit_ops import loss_gradients


def state_versions(model):
    """Detect in-place writes or replacement of parameters/persistent buffers."""
    tensors = dict(model.named_parameters())
    # Exclude nonpersistent diagnostic caches and externally loaded text banks.
    persistent = set(model.state_dict())
    tensors.update({k: v for k, v in model.named_buffers() if k in persistent})
    return {k: (id(v), v._version) for k, v in tensors.items()}


def collect_windows(model, tpa, loader, args):
    """Actual bounded capture loop, independently testable without CUDA/D2."""
    parameters = tuple(tpa.parameters())
    versions = state_versions(model)
    sampled = {}
    original_filter = model.filter_content_info
    def record_fedloss(data):
        indices, output = original_filter(data)
        sampled["category_indices"] = indices.detach().cpu().tolist()
        sampled["rare_category_count"] = int(model.novel_idx[indices].sum())
        return indices, output
    model.filter_content_info = record_fedloss
    windows = []
    try:
        for window in range(args.windows):
            sums, losses, micro_rows = {}, defaultdict(float), []
            for micro in range(args.microbatches):
                data = next(loader)
                if len(data) != args.batch_size:
                    raise ValueError("Unexpected micro-batch size")
                gt_classes = [d["instances"].gt_classes.tolist() for d in data]
                # Record pre-forward mapped inputs: forward may remap GT classes
                # in-place for FedLoss. These hashes check endpoint probe pairing.
                mapped_inputs = [{
                    "image_sha256": tensor_digest(d["image"]),
                    "gt_classes_sha256": tensor_digest(d["instances"].gt_classes),
                    "gt_boxes_sha256": (tensor_digest(d["instances"].gt_boxes.tensor)
                                        if hasattr(d["instances"], "gt_boxes") else None),
                } for d in data]
                if any(c < 0 or c >= len(model.novel_idx) or bool(model.novel_idx[c])
                       for classes in gt_classes for c in classes):
                    raise ValueError("A rare/invalid GT entered the training-loss audit")
                sampled.clear()
                loss_dict = model(data)
                gradients, loss_values, unused = loss_gradients(loss_dict, parameters)
                if "category_indices" not in sampled:
                    raise ValueError("Missing FedLoss subset capture")
                for name, gradient in gradients.items():
                    gradient = gradient.cpu() / args.microbatches
                    sums[name] = sums.get(name, torch.zeros_like(gradient)) + gradient
                for key, value in loss_values.items():
                    losses[key] += value / args.microbatches
                micro_rows.append({"image_ids": [int(d["image_id"]) for d in data],
                                   "global_gt_classes_before_remap": gt_classes,
                                   "mapped_image_shapes": [list(d["image"].shape) for d in data],
                                   "mapped_inputs": mapped_inputs,
                                   "fedloss_category_indices": sampled["category_indices"],
                                   "fedloss_rare_category_count": sampled["rare_category_count"],
                                   "unused_parameter_indices": unused, "losses": loss_values})
                print(f"[gradients] window={window + 1}/{args.windows} micro={micro + 1}/{args.microbatches} "
                      f"images={micro_rows[-1]['image_ids']} detector={sum(v for k,v in loss_values.items() if k not in ('loss_apr','loss_rpsa')):.5f} "
                      f"apr={loss_values['loss_apr']:.6f}", flush=True)
                del loss_dict, gradients, data
            if state_versions(model) != versions or any(p.grad is not None for p in model.parameters()):
                raise ValueError("Model tensors changed or .grad was populated; refusing audit")
            windows.append({"window": window, "gradients": sums, "mean_losses": dict(losses), "microbatches": micro_rows})
    finally:
        model.filter_content_info = original_filter
    return windows


def capture(args, geometry, expected_tpa):
    # Heavy imports are lazy: --reuse-gradients works without Detectron2/CUDA.
    from detectron2.config import LazyConfig, instantiate
    from detectron2.data import MetadataCatalog
    from detectron2.utils.events import EventStorage
    from omegaconf import OmegaConf

    if not torch.cuda.is_available() or not args.device.startswith("cuda"):
        raise ValueError("Native loss capture needs one CUDA device; use the lami interpreter")
    if torch.distributed.is_initialized():
        raise ValueError("This bounded audit is single-process, not DDP")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    cfg = LazyConfig.load(args.config_file)
    protocol = geometry["protocol"]
    cfg = LazyConfig.apply_overrides(cfg, [
        f"model.device={args.device}", f"train.device={args.device}",
        f"model.alpha={protocol['alpha']}", f"model.beta={protocol['beta']}",
        f"model.novel_scale={protocol['novel_scale']}",
        f"model.classifier.tpa_tau={protocol['tpa_tau']}",
        f"model.classifier.tpa_cls_tau={protocol['cls_tau']}",
        "model.classifier.tpa_eval_legacy_logsumexp=False", "model.classifier.tpa_eval_logit_bias=0.0",
        "model.tpa_eval_mode_scale=1.0",
    ])
    dataset_name = cfg.dataloader.train.dataset.names
    if dataset_name != "lvis_v1_train_norare" or not cfg.dataloader.train.mapper.is_train:
        raise ValueError("Audit requires the configured LVIS train_norare training mapper, not validation GT")
    if not cfg.train.tpa_conflict_projection or not cfg.train.separate_tpa_grad_clip:
        raise ValueError("Expected the current conflict-projection/separate-TPA-clip protocol")
    if not cfg.train.clip_grad.enabled or float(cfg.train.clip_grad.params.norm_type) != 2.:
        raise ValueError("Expected enabled L2 gradient clipping")
    if cfg.model.tpa_stabilization_steps != 0 or cfg.model.tpa_task_gradient_scale != 1.:
        raise ValueError("Audit requires full task gradients, no stabilization phase")
    if not cfg.model.use_fed_loss:
        raise ValueError("Expected native FedLoss sampling")
    print("[load] instantiate detector; full checkpoint, no optimizer or scheduler", flush=True)
    model = instantiate(cfg.model)
    checkpoint = load_trusted_torch_file(args.checkpoint or geometry["checkpoint"]["path"])
    state = checkpoint.get("model", checkpoint.get("state_dict", checkpoint))
    state = {k[7:] if k.startswith("module.") else k: v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    iteration = int(checkpoint["iteration"])
    del state, checkpoint
    model.to(args.device).train()
    tpa = model.transformer.decoder.class_embed[0].tpa
    if tuple(dict(tpa.named_parameters())) != WEIGHTS:
        raise ValueError("Unexpected TPA parameter order/layout")
    for name in WEIGHTS + ("slot_prior_strength", "prototype_mode_strength"):
        if not torch.equal(tpa.state_dict()[name].detach().cpu(), expected_tpa[name]):
            raise ValueError(f"Loaded TPA differs from geometry checkpoint: {name}")
    if iteration != geometry["tpa_state"]["iteration"]:
        raise ValueError("Checkpoint iteration differs from geometry audit")
    classifier = model.transformer.decoder.class_embed[0]
    # For this run both banks must be the same fixed full-vocabulary prompt bank.
    prompt_digest = tensor_digest(torch.from_numpy(np.load(geometry["prompt_bank"]["path"], allow_pickle=False)).float())
    if any(tensor_digest(t) != prompt_digest for t in (classifier.train_text_feats, classifier.eval_text_feats)):
        raise ValueError("Configured training/eval prompt tensors differ from geometry audit")
    parameters = tuple(tpa.parameters())
    parameter_ids = {id(p) for p in parameters}
    for p in model.parameters():
        p.requires_grad_(id(p) in parameter_ids)
    # Freezing other weights preserves the partial derivative w.r.t. TPA, but
    # training dropout/denoising/matching still run normally.
    if any(isinstance(m, torch.nn.modules.batchnorm._BatchNorm) and m.training for m in model.modules()):
        raise ValueError("Training BatchNorm needs a separate distributed-statistics audit")
    model.tpa_advance_step = False
    cfg.dataloader.train.total_batch_size = args.batch_size
    cfg.dataloader.train.num_workers = 0
    loader = iter(instantiate(cfg.dataloader.train))
    metadata = MetadataCatalog.get(dataset_name)
    train_annotation = file_identity(metadata.json_file)
    # Human-readable snapshot without an optional formatter dependency after GPU work.
    resolved_config = pformat(OmegaConf.to_container(cfg, resolve=True), sort_dicts=True)
    with EventStorage(start_iter=iteration):
        windows = collect_windows(model, tpa, loader, args)
    result = {"windows": windows, "train_annotations": train_annotation, "iteration": iteration,
              "dataset": dataset_name, "clip_max_norm": float(cfg.train.clip_grad.params.max_norm),
              "resolved_config": resolved_config, "torch_version": str(torch.__version__),
              "tpa_step": int(tpa._step), "tpa_dropout": tpa.dropout.p,
              "apr_internal_weights": list(tpa._effective_lambdas()),
              "outer_apr_weight": float(model.criterion.weight_dict["loss_apr"]),
              "weights_unchanged": True, "optimizer_created": False,
              "notes": ["Training dropout, denoising, Hungarian matching and FedLoss remain active.",
                        "One-GPU FedLoss subsets differ from distributed all-rank sampling; this is NOT an exact original optimizer batch.",
                        "RPSA stays in task for routing; gradients are averaged before projection.",
                        "Only TPA parameter derivatives are requested; all other weights are constants.",
                        "Every window uses the same checkpoint and iteration, with no parameter update."]}
    del loader, parameters, classifier, tpa, model
    torch.cuda.empty_cache()
    return result
