#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
"""
Training script using the new "LazyConfig" python config files.

This scripts reads a given python config file and runs the training or evaluation.
It can be used to train any models or dataset as long as they can be
instantiated by the recursive construction defined in the given config file.

Besides lazy construction of models, dataloader, etc., this scripts expects a
few common configuration parameters currently defined in "configs/common/train.py".
To add more complicated training logic, you can easily add other configs
in the config file and implement a new train_net.py to handle them.
"""
import logging
import math
import os
import sys
import time
import warnings
import numpy as np
import torch

# Suppress warnings for cleaner training output
warnings.filterwarnings("ignore", category=FutureWarning, module="timm")
warnings.filterwarnings("ignore", message="torch.meshgrid.*indexing.*")
from torch.nn.parallel import DataParallel, DistributedDataParallel

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import LazyConfig, instantiate
from detectron2.engine import (
    SimpleTrainer,
    default_argument_parser,
    default_setup,
    default_writers,
    hooks,
    launch,
)
from detectron2.engine.defaults import create_ddp_model
from detectron2.evaluation import inference_on_dataset, print_csv_format
from detectron2.utils import comm
from detectron2.utils.events import get_event_storage

# ==== Added by ChatGPT ====
import torch.distributed as dist
from torch import nn
# ==== End ====

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

from lami_dino.checkpoint_init import (
    load_backbone_only,
    validate_backbone_trainable_scope,
)
from lami_dino.prototype_ops import route_conflicting_task_gradient

logger = logging.getLogger("detrex")


class Trainer(SimpleTrainer):
    """
    We've combine Simple and AMP Trainer together.
    """

    def __init__(
        self,
        model,
        dataloader,
        optimizer,
        amp=False,
        clip_grad_params=None,
        separate_tpa_grad_clip=False,
        tpa_conflict_projection=False,
        grad_scaler=None,
    ):
        super().__init__(model=model, data_loader=dataloader, optimizer=optimizer)

        unsupported = "AMPTrainer does not support single-process multi-device training!"
        if isinstance(model, DistributedDataParallel):
            assert not (model.device_ids and len(model.device_ids) > 1), unsupported
        assert not isinstance(model, DataParallel), unsupported

        if amp:
            if grad_scaler is None:
                from torch.cuda.amp import GradScaler

                grad_scaler = GradScaler()
            self.grad_scaler = grad_scaler

        # set True to use amp training
        self.amp = amp

        # gradient clip hyper-params
        self.clip_grad_params = clip_grad_params
        self.separate_tpa_grad_clip = bool(separate_tpa_grad_clip)
        self.tpa_conflict_projection = bool(tpa_conflict_projection)

        self._last_tpa_grad_norm_pre_clip = float("nan")
        self._last_tpa_grad_norm_post_clip = float("nan")
        self._last_tpa_lr = float("nan")
        self._last_tpa_projection_metrics = {
            "task_grad_norm": float("nan"),
            "apr_grad_norm": float("nan"),
            "task_apr_cosine": float("nan"),
            "conflict_projected": float("nan"),
            "routed_grad_norm": float("nan"),
        }

    def run_step(self):
        """
        Implement the standard training logic described above.
        """
        assert self.model.training, "[Trainer] model was changed to eval mode!"
        assert torch.cuda.is_available(), "[Trainer] CUDA is required for AMP training!"
        from torch.cuda.amp import autocast

        start = time.perf_counter()
        """
        If you want to do something with the data, you can wrap the dataloader.
        """
        data = next(self._data_loader_iter)
        data_time = time.perf_counter() - start

        """
        If you want to do something with the losses, you can wrap the model.
        """
        loss_dict = self.model(data)
        with autocast(enabled=self.amp):
            if isinstance(loss_dict, torch.Tensor):
                losses = loss_dict
                loss_dict = {"total_loss": loss_dict}
            else:
                losses = sum(v for k, v in loss_dict.items() if k.startswith("loss"))

        """
        If you need to accumulate gradients or do something similar, you can
        wrap the optimizer with your custom `zero_grad()` method.
        """
        self.optimizer.zero_grad()
        apr_gradients = self._compute_apr_gradients(loss_dict)

        if self.amp:
            self.grad_scaler.scale(losses).backward()
            # Unscale before both diagnostics and clipping. GradScaler.step()
            # accepts an optimizer that has already been unscaled.
            self.grad_scaler.unscale_(self.optimizer)
            self._route_tpa_gradients(apr_gradients)
            self._capture_tpa_optimization_metrics(before_clip=True)
            if self.clip_grad_params is not None:
                self.clip_model_grads()
            self._capture_tpa_optimization_metrics(before_clip=False)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            losses.backward()
            self._route_tpa_gradients(apr_gradients)
            self._capture_tpa_optimization_metrics(before_clip=True)
            if self.clip_grad_params is not None:
                self.clip_model_grads()
            self._capture_tpa_optimization_metrics(before_clip=False)
            self.optimizer.step()

        self._write_metrics(loss_dict, data_time)
        self._write_tpa_metrics()

    def _get_tpa(self):
        model = self.model.module if hasattr(self.model, "module") else self.model
        try:
            return getattr(model.transformer.decoder.class_embed[0], "tpa", None)
        except (AttributeError, IndexError):
            return None

    def _compute_apr_gradients(self, loss_dict):
        """Differentiate APR separately so conflicting task gradients are known."""
        model = self.model.module if hasattr(self.model, "module") else self.model
        if (
            not self.tpa_conflict_projection
            or getattr(model, "tpa_stabilizing", False)
            or not isinstance(loss_dict, dict)
            or "loss_apr" not in loss_dict
        ):
            return None
        tpa = self._get_tpa()
        if tpa is None:
            return None
        parameters = list(tpa.parameters())
        return torch.autograd.grad(
            loss_dict["loss_apr"],
            parameters,
            retain_graph=True,
            allow_unused=True,
        )

    def _route_tpa_gradients(self, apr_gradients):
        """Project away only the detector gradient that would increase APR."""
        if apr_gradients is None:
            return
        tpa = self._get_tpa()
        if tpa is None:
            return
        parameters = list(tpa.parameters())
        if len(parameters) != len(apr_gradients):
            raise RuntimeError("TPA parameter set changed between forward and backward")

        apr_flat = torch.cat([
            (
                gradient.detach().reshape(-1)
                if gradient is not None
                else torch.zeros_like(parameter).reshape(-1)
            )
            for parameter, gradient in zip(parameters, apr_gradients)
        ])
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(apr_flat, op=dist.ReduceOp.SUM)
            apr_flat.div_(dist.get_world_size())

        total_flat = torch.cat([
            (
                parameter.grad.detach().reshape(-1)
                if parameter.grad is not None
                else torch.zeros_like(parameter).reshape(-1)
            )
            for parameter in parameters
        ])
        routed_flat, stats = route_conflicting_task_gradient(total_flat, apr_flat)

        offset = 0
        for parameter in parameters:
            count = parameter.numel()
            routed = routed_flat[offset : offset + count].view_as(parameter)
            if parameter.grad is None:
                parameter.grad = routed.clone()
            else:
                parameter.grad.copy_(routed)
            offset += count
        self._last_tpa_projection_metrics = {
            key: float(value.item()) for key, value in stats.items()
        }

    def _capture_tpa_optimization_metrics(self, *, before_clip):
        """Measure the actual synchronized TPA gradient, not just grad presence."""
        tpa = self._get_tpa()
        if tpa is None:
            return
        parameters = list(tpa.parameters())
        grad_norms = [
            parameter.grad.detach().float().norm(2)
            for parameter in parameters
            if parameter.grad is not None
        ]
        grad_norm = (
            torch.stack(grad_norms).norm(2).item()
            if grad_norms
            else float("nan")
        )
        if before_clip:
            self._last_tpa_grad_norm_pre_clip = grad_norm
            tpa_param_ids = {id(parameter) for parameter in parameters}
            tpa_lrs = {
                float(group["lr"])
                for group in self.optimizer.param_groups
                if any(id(parameter) in tpa_param_ids for parameter in group["params"])
            }
            self._last_tpa_lr = min(tpa_lrs) if tpa_lrs else float("nan")
        else:
            self._last_tpa_grad_norm_post_clip = grad_norm

    def _write_tpa_metrics(self):
        """Record TPA diagnostics, including the ones that detect collapse.

        This used to read `self.model.transformer.text_proto_bank`, which never
        exists -- nothing instantiates TextPrototypeBank, and under DDP `self.model`
        has no `.transformer` either -- so it never fired and the collapse ran a
        full ablation without ever reaching metrics.json.

        These go straight to EventStorage rather than into loss_dict, because
        _write_metrics calls .detach() on every value (these are plain floats) and
        sums them all into total_loss, where a NaN diagnostic would abort training.
        """
        model = self.model.module if hasattr(self.model, "module") else self.model
        tpa = self._get_tpa()

        # Losses are reduced by SimpleTrainer, but these diagnostics used to be
        # written directly inside DINO.forward and therefore described only the
        # local rank. Reduce them explicitly so active=0 cannot coexist with a
        # non-zero globally reduced RPSA loss in metrics.json.
        rpsa_metric_names = {
            "rpsa_bg_ratio": "loss_rpsa_bg_ratio",
            "rpsa_valid_clusters": "loss_rpsa_valid_clusters",
            "rpsa_active": "loss_rpsa_active",
            "rpsa_empty_image_ratio": "loss_rpsa_empty_image_ratio",
            "rpsa_tokens": "loss_rpsa_tokens",
            "rpsa_center_orth_mse": "rpsa_center_orth_mse",
            "rpsa_pi_entropy": "rpsa_pi_entropy",
        }
        rpsa_stats = getattr(model.transformer, "rpsa_last_stats", {})
        device = next(model.parameters()).device
        rpsa_to_reduce = {
            key: torch.as_tensor(rpsa_stats[key], device=device, dtype=torch.float32).reshape(())
            for key in rpsa_metric_names
            if key in rpsa_stats
        }
        reduced_rpsa = comm.reduce_dict(rpsa_to_reduce, average=True) if rpsa_to_reduce else {}

        if not comm.is_main_process():
            return
        storage = get_event_storage()
        for key, value in reduced_rpsa.items():
            value = float(value)
            if math.isfinite(value):
                storage.put_scalar(rpsa_metric_names[key], value, smoothing_hint=False)

        if tpa is None:
            return
        for key, value in tpa.get_monitor_dict().items():
            value = float(value)
            if math.isfinite(value):
                storage.put_scalar(f"tpa/{key}", value, smoothing_hint=False)
        optimization_metrics = {
            "grad_norm_pre_clip": self._last_tpa_grad_norm_pre_clip,
            "grad_norm_post_clip": self._last_tpa_grad_norm_post_clip,
            "lr": self._last_tpa_lr,
            "stabilizing": float(getattr(model, "tpa_stabilizing", False)),
            "task_gradient_scale": float(
                getattr(model, "tpa_active_task_gradient_scale", 1.0)
            ),
        }
        optimization_metrics.update(self._last_tpa_projection_metrics)
        for key, value in optimization_metrics.items():
            if math.isfinite(value):
                storage.put_scalar(f"tpa/{key}", value, smoothing_hint=False)

        # Query-fusion routing is part of the paper's language interface, not a
        # hidden implementation detail.  Persist enough information to tell
        # whether top-R fusion remains genuinely soft or degenerates to hard
        # top-1 routing during a run.
        fusion_stats = getattr(model.transformer, "last_query_fusion_stats", {})
        for key, value in fusion_stats.items():
            value = float(value)
            if math.isfinite(value):
                storage.put_scalar(f"query_fusion/{key}", value, smoothing_hint=False)

    def clip_grads(self, params):
        params = list(filter(lambda p: p.requires_grad and p.grad is not None, params))
        if len(params) > 0:
            return torch.nn.utils.clip_grad_norm_(
                parameters=params,
                **self.clip_grad_params,
            )

    def clip_model_grads(self):
        """Clip TPA and detector gradients independently when configured.

        The collapse diagnostic measured a 16--42 TPA gradient norm inside a
        0.5 global clipping budget. A stronger anti-collapse barrier would
        otherwise shrink every detector gradient along with the TPA. The split
        preserves the declared max norm for each parameter block.
        """
        tpa = self._get_tpa()
        if not self.separate_tpa_grad_clip or tpa is None:
            return self.clip_grads(self.model.parameters())

        tpa_parameters = list(tpa.parameters())
        tpa_ids = {id(parameter) for parameter in tpa_parameters}
        detector_parameters = [
            parameter
            for parameter in self.model.parameters()
            if id(parameter) not in tpa_ids
        ]
        detector_norm = self.clip_grads(detector_parameters)
        tpa_norm = self.clip_grads(tpa_parameters)
        return detector_norm, tpa_norm


def do_test(cfg, model):
    if "evaluator" in cfg.dataloader:
        try:
            # Temporarily reduce num_workers for evaluation to prevent BrokenPipeError
            original_num_workers = cfg.dataloader.test.num_workers
            cfg.dataloader.test.num_workers = 0  # Use single process for evaluation
            
            ret = inference_on_dataset(
                model, instantiate(cfg.dataloader.test), instantiate(cfg.dataloader.evaluator), cfg.DDEBUG
            )
            print_csv_format(ret)
            
            # Restore original num_workers
            cfg.dataloader.test.num_workers = original_num_workers
            return ret
        except BrokenPipeError as e:
            logger = logging.getLogger("detectron2")
            logger.warning(f"BrokenPipeError during evaluation: {e}")
            logger.warning("Skipping evaluation due to multiprocessing issue - training will continue")
            # Restore original num_workers
            cfg.dataloader.test.num_workers = original_num_workers
            return {}
        except Exception as e:
            logger = logging.getLogger("detectron2")
            logger.warning(f"Error during evaluation: {e}")
            logger.warning("Skipping evaluation - training will continue")
            # Restore original num_workers
            cfg.dataloader.test.num_workers = original_num_workers
            return {}


# ==== Added by ChatGPT ====
def _maybe_convert_syncbn(model, enable=True):
    """
    Convert all BatchNorm to SyncBatchNorm before DDP wrapping.
    """
    if not enable:
        return model
    try:
        if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
            print("[Init] Converting BatchNorm -> SyncBatchNorm ...")
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    except Exception as e:
        print(f"[Warning] SyncBatchNorm conversion failed: {e}")
    return model


def _broadcast_tpa_buffers(model):
    """
    Broadcast critical TPA buffers/params so multi-GPU start from identical states.
    Safe to call even on 1-GPU.
    """
    try:
        if not (dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1):
            return
        rank = dist.get_rank()

        # Try to locate the last classification head where TPA usually lives:
        cls_head = None
        if hasattr(model, "transformer") and hasattr(model.transformer, "decoder"):
            dec = model.transformer.decoder
            if hasattr(dec, "class_embed") and len(dec.class_embed) > 0:
                cls_head = dec.class_embed[-1]
        elif hasattr(model, "decoder") and hasattr(model.decoder, "class_embed"):
            dec = model.decoder
            cls_head = dec.class_embed[-1] if len(dec.class_embed) > 0 else None

        if cls_head is None:
            return

        # broadcast text feature buffers if present
        for name in ["train_text_feats", "eval_text_feats"]:
            if hasattr(cls_head, name):
                buf = getattr(cls_head, name)
                if isinstance(buf, torch.Tensor):
                    if rank == 0:
                        print(f"[Rank {rank}] Broadcasting buffer: {name} (shape={tuple(buf.shape)})")
                    dist.broadcast(buf.data, src=0)

        # broadcast TPA prototype queries if present
        if hasattr(cls_head, "tpa") and hasattr(cls_head.tpa, "prototype_queries"):
            pq = cls_head.tpa.prototype_queries
            if rank == 0:
                print(f"[Rank {rank}] Broadcasting TPA prototype_queries (shape={tuple(pq.shape)})")
            dist.broadcast(pq.data, src=0)

    except Exception as e:
        print(f"[Warning] broadcast_tpa_buffers failed: {e}")
# ==== End ====

def do_train(args, cfg):
    """
    Args:
        cfg: an object with the following attributes:
            model: instantiate to a module
            dataloader.{train,test}: instantiate to dataloaders
            dataloader.evaluator: instantiate to evaluator for test set
            optimizer: instantaite to an optimizer
            lr_multiplier: instantiate to a fvcore scheduler
            train: other misc config defined in `configs/common/train.py`, including:
                output_dir (str)
                init_checkpoint (str)
                init_checkpoint_scope (str): ``full`` or ``backbone_only``
                amp.enabled (bool)
                max_iter (int)
                eval_period, log_period (int)
                device (str)
                checkpointer (dict)
                ddp (dict)
    """
    model = instantiate(cfg.model)
    logger = logging.getLogger("detectron2")
    logger.info("Model:\n{}".format(model))
    backbone_scope = getattr(cfg.train, "backbone_trainable_scope", "full")
    trainable_backbone = validate_backbone_trainable_scope(
        model.backbone, backbone_scope
    )
    logger.info(
        "Validated backbone trainable scope %s (%d trainable tensors): %s",
        backbone_scope,
        len(trainable_backbone),
        ", ".join(trainable_backbone) if trainable_backbone else "none",
    )
    logger.info(
        "TPA stabilization: %d APR-only steps; post-stabilization task "
        "gradient scale=%.4g; conflict projection=%s; separate gradient clipping=%s",
        getattr(model, "tpa_stabilization_steps", 0),
        getattr(model, "tpa_task_gradient_scale", 1.0),
        getattr(cfg.train, "tpa_conflict_projection", False),
        getattr(cfg.train, "separate_tpa_grad_clip", False),
    )
    model.to(cfg.train.device)

    # ==== Added by ChatGPT ====
    # Convert BN -> SyncBN (only when multi-GPU is initialized)
    syncbn_flag = getattr(cfg.train, "sync_batchnorm", True)
    model = _maybe_convert_syncbn(model, enable=syncbn_flag)

    # Proactively broadcast TPA buffers/params before DDP,
    # especially helpful when cfg.train.ddp.broadcast_buffers=False.
    _broadcast_tpa_buffers(model)
    # ==== End ====

    cfg.optimizer.params.model = model
    optim = instantiate(cfg.optimizer)

    train_loader = instantiate(cfg.dataloader.train)

    model = create_ddp_model(model, **cfg.train.ddp)

    trainer = Trainer(
        model=model,
        dataloader=train_loader,
        optimizer=optim,
        amp=cfg.train.amp.enabled,
        clip_grad_params=cfg.train.clip_grad.params if cfg.train.clip_grad.enabled else None,
        separate_tpa_grad_clip=getattr(cfg.train, "separate_tpa_grad_clip", False),
        tpa_conflict_projection=getattr(cfg.train, "tpa_conflict_projection", False),
    )

    checkpointer = DetectionCheckpointer(
        model,
        cfg.train.output_dir,
        trainer=trainer,
    )

    trainer.register_hooks(
        [
            hooks.IterationTimer(),
            hooks.LRScheduler(scheduler=instantiate(cfg.lr_multiplier)),
            hooks.PeriodicCheckpointer(checkpointer, **cfg.train.checkpointer)
            if comm.is_main_process()
            else None,
            hooks.EvalHook(cfg.train.eval_period, lambda: do_test(cfg, model)),
            hooks.PeriodicWriter(
                default_writers(cfg.train.output_dir, cfg.train.max_iter),
                period=cfg.train.log_period,
            )
            if comm.is_main_process()
            else None,
        ]
    )

    # Robust checkpoint resuming with automatic detection. A real resume always
    # restores the full training state. For a fresh paper run, however, the
    # configured CLIP checkpoint is restricted to the visual backbone so no
    # LaMI/DINO detector weights can leak into the initialization.
    logger = logging.getLogger("detectron2")
    resume_from_output = bool(args.resume and checkpointer.has_checkpoint())

    if resume_from_output:
        checkpointer.resume_or_load(cfg.train.init_checkpoint, resume=True)
    else:
        init_scope = getattr(cfg.train, "init_checkpoint_scope", "full")
        if init_scope == "backbone_only":
            unwrapped_model = model.module if hasattr(model, "module") else model
            report = load_backbone_only(
                unwrapped_model.backbone,
                cfg.train.init_checkpoint,
            )
            logger.info(
                "Loaded CLIP backbone only from %s: %d/%d tensors, %.2f%% "
                "parameter coverage; ignored %d non-backbone/incompatible tensors",
                report["checkpoint_path"],
                report["loaded_tensor_count"],
                report["target_tensor_count"],
                100.0 * report["parameter_coverage"],
                report["ignored_tensor_count"],
            )
        elif init_scope == "full":
            checkpointer.resume_or_load(cfg.train.init_checkpoint, resume=False)
        else:
            raise ValueError(
                "train.init_checkpoint_scope must be 'full' or 'backbone_only', "
                f"got {init_scope!r}"
            )

    if resume_from_output:
        # The checkpoint stores the training iteration that just finished, thus we start
        # at the next iteration
        start_iter = trainer.iter + 1
        logger.info(f"Resuming training from iteration {start_iter}")
    else:
        start_iter = 0
        logger.info("Starting training from iteration 0")
    
    # Add error handling for training loop
    try:
        trainer.train(start_iter, cfg.train.max_iter)
    except Exception as e:
        logger.error(f"Training interrupted with error: {e}")
        logger.info("Training can be resumed using --resume flag")
        raise


def main(args):
    cfg = LazyConfig.load(args.config_file)
    cfg = LazyConfig.apply_overrides(cfg, args.opts)
    if args.ddebug:
        cfg.train.max_iter = 8
        cfg.train.eval_period = 8
        cfg.train.log_period = 4
        cfg.train.checkpointer.period = 8
        cfg.dataloader.train.num_workers = 0
        cfg.dataloader.train.total_batch_size = 1
        cfg.train.output_dir = 'output/debug'
        cfg.dataloader.evaluator.output_dir = 'output/debug'
        if cfg.model.save_dir:
            cfg.model.save_dir = cfg.model.save_dir + '_debug'
        cfg.DDEBUG = True
    else:
        cfg.DDEBUG = False
    default_setup(cfg, args)

    if args.eval_only:
        model = instantiate(cfg.model)
        model.to(cfg.train.device)

        # ==== Added by ChatGPT ====
        # Keep eval path consistent: convert SyncBN & broadcast buffers pre-DDP as well.
        syncbn_flag = getattr(cfg.train, "sync_batchnorm", True)
        model = _maybe_convert_syncbn(model, enable=syncbn_flag)
        _broadcast_tpa_buffers(model)
        # ==== End ====

        model = create_ddp_model(model)
        DetectionCheckpointer(model).load(cfg.train.init_checkpoint)
        print(do_test(cfg, model))
    else:
        do_train(args, cfg)


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
