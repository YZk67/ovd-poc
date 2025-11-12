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
        assert self.model.training, "[Trainer] model was changed to eval mode!"
        assert torch.cuda.is_available(), "[Trainer] CUDA is required for AMP training!"
        from torch.cuda.amp import autocast

        # 1) 取数据并计时
        start = time.perf_counter()
        data = next(self._data_loader_iter)
        data_time = time.perf_counter() - start

        # 2) 清梯度
        self.optimizer.zero_grad()

        # 3) 前向 + 汇总 loss（放进 autocast）
        with autocast(enabled=self.amp, dtype=torch.bfloat16):
            loss_dict = self.model(data)
            if isinstance(loss_dict, torch.Tensor):
                losses = loss_dict
                loss_dict = {"total_loss": loss_dict}
            else:
                losses = sum(loss_dict.values())

        # 4) 反向 +（可选）梯度裁剪 + 更新
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

        # 5) 只打印一次 TPA 梯度信息（健壮判断）
        if not self.tpa_grad_printed:
            model = self.model.module if hasattr(self.model, "module") else self.model
            if hasattr(model, "transformer"):
                if hasattr(model.transformer, "use_soft_attention"):
                    logger.info(f"use_soft_attention: {getattr(model.transformer, 'use_soft_attention', None)}")
                if hasattr(model.transformer, "soft_attention_tau"):
                    logger.info(f"soft_attention_tau: {getattr(model.transformer, 'soft_attention_tau', None)}")
                try:
                    head0 = getattr(model.transformer.decoder, "class_embed", [None])[0]
                    if head0 is not None and hasattr(head0, "tpa"):
                        logger.info("TPA parameters gradient status:")
                        for n, p in head0.tpa.named_parameters():
                            logger.info(f"  {n}: grad={'Yes' if p.grad is not None else 'No'}")
                    else:
                        logger.info("TPA not found in class_embed[0]")
                except Exception:
                    logger.info("TPA grad probe skipped.")
            self.tpa_grad_printed = True

        # 6) 额外监控项写入
        if hasattr(self.model, "transformer") and hasattr(self.model.transformer, "text_proto_bank"):
            try:
                monitor_dict = self.model.transformer.text_proto_bank.aggregator.get_monitor_dict()
                loss_dict.update(monitor_dict)
            except Exception:
                pass

        # 7) 写指标
        self._write_metrics(loss_dict, data_time)


    def clip_grads(self, params):
        params = list(filter(lambda p: p.requires_grad and p.grad is not None, params))
        if len(params) > 0:
            return torch.nn.utils.clip_grad_norm_(
                parameters=params,
                **self.clip_grad_params,
            )


# def do_test(cfg, model):
#     if "evaluator" in cfg.dataloader:
#         try:
#             # Temporarily reduce num_workers for evaluation to prevent BrokenPipeError
#             original_num_workers = cfg.dataloader.test.num_workers
#             cfg.dataloader.test.num_workers = 0  # Use single process for evaluation
            
#             ret = inference_on_dataset(
#                 model, instantiate(cfg.dataloader.test), instantiate(cfg.dataloader.evaluator), cfg.DDEBUG
#             )
#             print_csv_format(ret)
            
#             # Restore original num_workers
#             cfg.dataloader.test.num_workers = original_num_workers
#             return ret
#         except BrokenPipeError as e:
#             logger = logging.getLogger("detectron2")
#             logger.warning(f"BrokenPipeError during evaluation: {e}")
#             logger.warning("Skipping evaluation due to multiprocessing issue - training will continue")
#             # Restore original num_workers
#             cfg.dataloader.test.num_workers = original_num_workers
#             return {}
#         except Exception as e:
#             logger = logging.getLogger("detectron2")
#             logger.warning(f"Error during evaluation: {e}")
#             logger.warning("Skipping evaluation - training will continue")
#             # Restore original num_workers
#             cfg.dataloader.test.num_workers = original_num_workers
#             return {}

def do_test(cfg, model):
    """
    支持多个 test dataloader / evaluator，逐个评测并打印。
    返回一个 dict[name] = metrics。
    """
    if "evaluator" not in cfg.dataloader:
        return {}

    # 统一实例化
    test_obj = instantiate(cfg.dataloader.test)
    eval_obj = instantiate(cfg.dataloader.evaluator)

    # 归一化为列表
    tests = test_obj if isinstance(test_obj, (list, tuple)) else [test_obj]
    evals = eval_obj if isinstance(eval_obj, (list, tuple)) else [eval_obj]

    # evaluator 个数与 dataloader 不匹配时，重复最后一个
    if len(evals) < len(tests):
        evals = list(evals) + [evals[-1]] * (len(tests) - len(evals))

    results = {}
    logger = logging.getLogger("detectron2")

    # 记录并临时降 worker
    saved_workers = []
    try:
        for i, (td, ev) in enumerate(zip(tests, evals)):
            # 取名字（如果 LazyConfig 有 name 字段/属性）
            name = getattr(td, "dataset", None)
            try:
                name = getattr(td.dataset, "name", None) or getattr(td, "name", None)
            except Exception:
                pass
            name = name or f"test_{i}"

            # 兼容 num_workers 字段
            nw = getattr(td, "num_workers", None)
            saved_workers.append(nw)
            if nw is not None:
                setattr(td, "num_workers", 0)

            logger.info(f"Running evaluation on: {name}")
            try:
                ret = inference_on_dataset(model, td, ev, cfg.DDEBUG)
                results[name] = ret
                print_csv_format(ret)
            except BrokenPipeError as e:
                logger.warning(f"BrokenPipeError during evaluation: {e}")
                logger.warning("Skipping this evaluation due to multiprocessing issue")
                results[name] = {}
            except Exception as e:
                logger.warning(f"Error during evaluation on {name}: {e}")
                logger.warning("Skipping this evaluation")
                results[name] = {}

    finally:
        # 恢复 num_workers
        for td, nw in zip(tests, saved_workers):
            if nw is not None:
                setattr(td, "num_workers", nw)

    return results


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

def _inject_ovd_indices(model, cfg):
    """
    把 48->65 的映射以“运行期属性”的方式塞进模型，
    避免 __init__ 参数报错；训练时会在 TextClassifier.forward 里切列。
    """
    if not hasattr(cfg, "ovd"):
        return
    tr = cfg.ovd.get("train48_to_65", None)
    ev = cfg.ovd.get("eval_idx_65", None)
    if tr is None or ev is None:
        return

    device = next(model.parameters()).device
    import torch
    tr = torch.as_tensor(tr, dtype=torch.long, device=device)
    ev = torch.as_tensor(ev, dtype=torch.long, device=device)

    # 你分类头的挂载位置（按你的工程，通常是 model.classifier）
    head = getattr(model, "classifier", None)
    if head is None:
        # 兜底：尝试从 transformer.decoder.class_embed 找最后一个头
        try:
            head = model.transformer.decoder.class_embed[-1]
        except Exception:
            head = None
    if head is None:
        raise RuntimeError("未找到分类头以注入 OVD 索引（model.classifier 或 decoder.class_embed[-1]）。")

    head.train_class_indices = tr
    head.eval_class_indices  = ev

    assert tr.numel() == 48 and ev.numel() == 65, "OVD 索引长度异常（应为 48 和 65）"

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
    # 在 instantiate(cfg.model) 之后
    try:
        head = model.transformer.decoder.class_embed[-1]
        if hasattr(head, "train_text_feats"):
            print("train_text_feats:", tuple(head.train_text_feats.shape))
        if hasattr(head, "eval_text_feats"):
            print("eval_text_feats:", tuple(head.eval_text_feats.shape))
    except Exception as _:
        pass

    logger = logging.getLogger("detectron2")
    logger.info("Model:\n{}".format(model))
    model.to(cfg.train.device)

    _inject_ovd_indices(model, cfg)

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

        _inject_ovd_indices(model, cfg)

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
