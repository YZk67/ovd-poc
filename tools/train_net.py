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

# ==== Added by ChatGPT ====
import torch.distributed as dist
from torch import nn
# ==== End ====

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

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
        
        # Flag to print TPA gradient info only once
        self.tpa_grad_printed = False

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

        if self.amp:
            self.grad_scaler.scale(losses).backward()
            if self.clip_grad_params is not None:
                self.grad_scaler.unscale_(self.optimizer)
                self.clip_grads(self.model.parameters())
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            losses.backward()
            if self.clip_grad_params is not None:
                self.clip_grads(self.model.parameters())
            self.optimizer.step()

        # === 只打印一次 TPA 梯度信息 ===
        if not self.tpa_grad_printed:
            # Get the underlying model (unwrap DDP if needed)
            model = self.model.module if hasattr(self.model, 'module') else self.model
            logger.info(f"use_soft_attention: {model.transformer.use_soft_attention}")
            logger.info(f"soft_attention_tau: {model.transformer.soft_attention_tau}")
            
            # 检查 TPA 参数的梯度
            if hasattr(model.transformer.decoder.class_embed[0], 'tpa'):
                logger.info("TPA parameters gradient status:")
                for n, p in model.transformer.decoder.class_embed[0].tpa.named_parameters():
                    logger.info(f"  {n}: grad={'Yes' if p.grad is not None else 'No'}")
            else:
                logger.info("TPA not found in class_embed[0]")
            
            self.tpa_grad_printed = True  # ✅ 确保只打印一次
        
        # === 定期监控TPA指标和梯度 ===
        if self.iter % 100 == 0:  # 每100次iteration监控一次
            try:
                from examples.monitor_tpa_during_training import monitor_tpa_metrics, print_tpa_metrics
                model_for_monitor = self.model.module if hasattr(self.model, 'module') else self.model
                metrics = monitor_tpa_metrics(model_for_monitor, self.iter, log_interval=1)
                if metrics:
                    print_tpa_metrics(metrics)
            except Exception as e:
                # 如果导入失败或出错，静默忽略（不影响训练）
                pass
            
            # === 监控prototype_queries的梯度 ===
            try:
                model_for_grad = self.model.module if hasattr(self.model, 'module') else self.model
                if hasattr(model_for_grad, 'transformer') and hasattr(model_for_grad.transformer, 'decoder'):
                    if hasattr(model_for_grad.transformer.decoder, 'class_embed'):
                        if len(model_for_grad.transformer.decoder.class_embed) > 0:
                            text_classifier = model_for_grad.transformer.decoder.class_embed[0]
                            if hasattr(text_classifier, 'tpa') and hasattr(text_classifier.tpa, 'prototype_queries'):
                                prototype_queries = text_classifier.tpa.prototype_queries
                                if prototype_queries.grad is not None:
                                    grad = prototype_queries.grad
                                    grad_norm = grad.norm().item()
                                    grad_mean = grad.abs().mean().item()
                                    grad_max = grad.abs().max().item()
                                    
                                    # 计算prototype_queries之间的相似度
                                    with torch.no_grad():
                                        queries_norm = torch.nn.functional.normalize(prototype_queries.data, p=2, dim=1)
                                        similarity_matrix = torch.mm(queries_norm, queries_norm.t())
                                        off_diag_mask = ~torch.eye(similarity_matrix.size(0), dtype=bool, device=similarity_matrix.device)
                                        max_similarity = similarity_matrix[off_diag_mask].max().item()
                                        mean_similarity = similarity_matrix[off_diag_mask].mean().item()
                                    
                                    logger.info(
                                        f"[Gradient Monitor] iter={self.iter} "
                                        f"grad_norm={grad_norm:.6f} "
                                        f"grad_mean={grad_mean:.6f} "
                                        f"grad_max={grad_max:.6f} "
                                        f"max_sim={max_similarity:.4f} "
                                        f"mean_sim={mean_similarity:.4f}"
                                    )
                                else:
                                    logger.warning(f"[Gradient Monitor] iter={self.iter} prototype_queries.grad is None!")
            except Exception as e:
                # 如果监控失败，静默忽略（不影响训练）
                pass
        
        if hasattr(self.model, "transformer") and hasattr(self.model.transformer, "text_proto_bank"):
            monitor_dict = self.model.transformer.text_proto_bank.aggregator.get_monitor_dict()
            loss_dict.update(monitor_dict)

        self._write_metrics(loss_dict, data_time)

    def clip_grads(self, params):
        params = list(filter(lambda p: p.requires_grad and p.grad is not None, params))
        if len(params) > 0:
            return torch.nn.utils.clip_grad_norm_(
                parameters=params,
                **self.clip_grad_params,
            )


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

    # Robust checkpoint resuming with automatic detection
    logger = logging.getLogger("detectron2")
    
    # Try to resume from checkpoint if available
    checkpointer.resume_or_load(cfg.train.init_checkpoint, resume=args.resume)
    
    if args.resume and checkpointer.has_checkpoint():
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
