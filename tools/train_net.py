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

# CHANGE: force CUDA sync error reporting in spawned processes
os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "1")
os.environ.setdefault("TORCH_SHOW_CPP_STACKTRACES", "1")
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
from detectron2.data import MetadataCatalog

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
                losses = sum(loss_dict.values())

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


def _apply_trainable_param_keywords(model, keywords):
    """
    Freeze all parameters except those whose names contain any of the given keywords.
    """
    if not keywords:
        return
    keywords = [k for k in keywords if isinstance(k, str) and k]
    if not keywords:
        return

    d2_logger = logging.getLogger("detectron2")
    total, trainable = 0, 0
    trainable_names = []
    for name, param in model.named_parameters():
        total += param.numel()
        keep = any(k in name for k in keywords)
        param.requires_grad = bool(keep)
        if keep:
            trainable += param.numel()
            trainable_names.append(name)

    pct = 100.0 * trainable / max(total, 1)
    d2_logger.info(
        "[FreezePolicy] trainable keywords=%s | trainable params=%d/%d (%.4f%%)",
        keywords,
        trainable,
        total,
        pct,
    )
    preview = trainable_names[:20]
    if preview:
        d2_logger.info("[FreezePolicy] first trainable params: %s", preview)


def do_test(cfg, model):
    if "evaluator" in cfg.dataloader:
        training_mode = model.training  # CHANGE: remember current mode to restore after eval
        try:
            # Temporarily reduce num_workers for evaluation to prevent BrokenPipeError
            original_num_workers = cfg.dataloader.test.num_workers
            cfg.dataloader.test.num_workers = 0  # Use single process for evaluation

            ret = inference_on_dataset(
                model, instantiate(cfg.dataloader.test), instantiate(cfg.dataloader.evaluator), cfg.DDEBUG
            )
            if comm.is_main_process():
                _maybe_add_ap50_seen_novel(cfg, ret)
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
        finally:
            model.train(training_mode)  # CHANGE: ensure training mode is restored


def _maybe_add_ap50_seen_novel(cfg, ret):
    try:
        seen_path = getattr(cfg.model, "seen_classes", None)
        all_path = getattr(cfg.model, "all_classes", None)
        if not seen_path or not all_path:
            return
        dataset_name = cfg.dataloader.test.dataset.names
        if isinstance(dataset_name, (list, tuple)):
            dataset_name = dataset_name[0]
        meta = MetadataCatalog.get(dataset_name)
        gt_json = getattr(meta, "json_file", None)
        if not gt_json:
            return
        output_dir = getattr(cfg.dataloader.evaluator, "output_dir", None)
        if not output_dir:
            return
        results_json = os.path.join(output_dir, "coco_instances_results.json")
        if not os.path.exists(results_json):
            return

        ap50_seen, ap50_novel = _compute_seen_novel_ap50(
            gt_json, results_json, seen_path, all_path, iou_type="bbox"
        )
        if "bbox" in ret:
            ret["bbox"]["AP50_base"] = ap50_seen
            ret["bbox"]["AP50_novel"] = ap50_novel
        else:
            ret["AP50_base"] = ap50_seen
            ret["AP50_novel"] = ap50_novel
        logger.info(f"[AP50 split] base: {ap50_seen:.3f}, novel: {ap50_novel:.3f}")
    except Exception as e:
        logger.warning(f"[AP50 split] failed to compute: {e}")


def _compute_seen_novel_ap50(gt_json, results_json, seen_path, all_path, iou_type="bbox"):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    import json
    import numpy as np

    def load_list(p):
        with open(p, "r") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError(f"Expected list in {p}")
        return data

    seen = load_list(seen_path)
    all_classes = load_list(all_path)
    novel = [c for c in all_classes if c not in seen]

    coco_gt = COCO(gt_json)
    coco_dt = coco_gt.loadRes(results_json)
    coco_eval = COCOeval(coco_gt, coco_dt, iou_type)
    coco_eval.evaluate()
    coco_eval.accumulate()

    cat_ids = coco_gt.getCatIds()
    cats = coco_gt.loadCats(cat_ids)
    cat_names = [c["name"] for c in cats]

    precision = coco_eval.eval["precision"]  # T x R x K x A x M
    iou_idx = 0  # IoU=0.50
    ap = {}
    for k, name in enumerate(cat_names):
        p = precision[iou_idx, :, k, 0, 2]  # area=all, maxDets=100
        p = p[p > -1]
        ap[name] = float(np.mean(p)) if p.size else 0.0

    seen_vals = [ap[c] for c in seen if c in ap]
    novel_vals = [ap[c] for c in novel if c in ap]
    ap50_seen = float(np.mean(seen_vals)) if seen_vals else 0.0
    ap50_novel = float(np.mean(novel_vals)) if novel_vals else 0.0
    return ap50_seen, ap50_novel


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
    # CHANGE: debug class counts / text embedding shapes for 65-class setup
    try:
        num_classes = getattr(model, "num_classes", None)
        text_embed = None
        if hasattr(model, "classifier"):
            text_embed = getattr(model.classifier, "text_embed", None)
        logger.info(f"[Debug] num_classes={num_classes}")
        if text_embed is not None:
            logger.info(f"[Debug] classifier.text_embed shape={tuple(text_embed.shape)}")
        else:
            logger.info("[Debug] classifier.text_embed not found")
        # Probe decoder text features (TPA / class_embed)
        if hasattr(model, "transformer") and hasattr(model.transformer, "decoder"):
            class_embed0 = model.transformer.decoder.class_embed[0]
            if hasattr(class_embed0, "_maybe_move_text_feats"):
                text_feats = class_embed0._maybe_move_text_feats(training=True)
                logger.info(f"[Debug] decoder text_feats shape={tuple(text_feats.shape)}")
            else:
                logger.info("[Debug] decoder class_embed[0] has no _maybe_move_text_feats")
    except Exception as e:
        logger.warning(f"[Debug] failed to report class/embed info: {e}")

    trainable_keywords = getattr(cfg.train, "trainable_param_keywords", [])
    _apply_trainable_param_keywords(model, trainable_keywords)

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
    sampler_obj = getattr(train_loader, "sampler", None)
    if sampler_obj is None and hasattr(train_loader, "dataset"):
        sampler_obj = getattr(train_loader.dataset, "sampler", None)
    if sampler_obj is not None:
        logger.info(f"TRAIN SAMPLER: {type(sampler_obj)} {sampler_obj}")
    else:
        logger.info(f"TRAIN SAMPLER: <unknown> (loader type {type(train_loader)})")

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
