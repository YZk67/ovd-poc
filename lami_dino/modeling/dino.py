# coding=utf-8
# Copyright 2022 The IDEA Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import copy
import math
import json
from typing import List
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from detrex.layers import MLP, box_cxcywh_to_xyxy, box_xyxy_to_cxcywh
from detrex.utils import (inverse_sigmoid, is_dist_avail_and_initialized,
                          load_class_freq, get_fed_loss_inds, get_cluster_fed_loss_inds)

from detectron2.modeling import detector_postprocess
from detectron2.structures import Boxes, ImageList, Instances
from detectron2.utils.logger import setup_logger
from detectron2.utils.events import get_event_storage
from lami_dino.checkpoint_init import load_trusted_torch_file
from lami_dino.inference_ops import select_query_class_topk
from lami_dino.prototype_ops import prototype_eval_mode_view, prototype_task_view
from lami_dino.models import teacher_routed_mode_alignment

logger_rpsa = setup_logger()  # 用于RPSA日志输出


class DINO(nn.Module):
    """Implement DAB-Deformable-DETR in `DAB-DETR: Dynamic Anchor Boxes are Better Queries for DETR
    <https://arxiv.org/abs/2203.03605>`_.

    Code is modified from the `official github repo
    <https://github.com/IDEA-Research/DINO>`_.

    Args:
        backbone (nn.Module): backbone module
        position_embedding (nn.Module): position embedding module
        neck (nn.Module): neck module to handle the intermediate outputs features
        transformer (nn.Module): transformer module
        embed_dim (int): dimension of embedding
        num_classes (int): Number of total categories.
        num_queries (int): Number of proposal dynamic anchor boxes in Transformer
        criterion (nn.Module): Criterion for calculating the total losses.
        pixel_mean (List[float]): Pixel mean value for image normalization.
            Default: [123.675, 116.280, 103.530].
        pixel_std (List[float]): Pixel std value for image normalization.
            Default: [58.395, 57.120, 57.375].
        aux_loss (bool): Whether to calculate auxiliary loss in criterion. Default: True.
        select_box_nums_for_evaluation (int): the number of topk candidates
            slected at postprocess for evaluation. Default: 300.
        inference_query_class_topk (int): maximum categories retained per query
            before the image-level top-k. Zero preserves the original global
            query/category selection. Default: 0.
        device (str): Training device. Default: "cuda".
    """

    def __init__(
        self,
        backbone: nn.Module,
        position_embedding: nn.Module,
        neck: nn.Module,
        transformer: nn.Module,
        embed_dim: int,
        num_classes: int,
        num_queries: int,
        criterion: nn.Module,
        classifier,
        query_path,
        eval_query_path,
        vlm_query_path,
        pixel_mean: List[float] = [123.675, 116.280, 103.530],
        pixel_std: List[float] = [58.395, 57.120, 57.375],
        aux_loss: bool = True,
        select_box_nums_for_evaluation: int = 300,
        inference_query_class_topk: int = 0,
        device="cuda",
        dn_number: int = 100,
        label_noise_ratio: float = 0.2,
        box_noise_scale: float = 1.0,
        use_fed_loss: bool = False,
        cluster_fed_loss: bool = False,
        cluster_label_path=None,
        fed_loss_num_cat: int = 50,
        cat_freq_path = None,
        fed_loss_freq_weight = 0.5,
        score_ensemble: bool = False,
        unseen_classes=None,
        seen_classes=None,
        all_classes=None,
        save_dir=None,
        vlm_temperature: float =100.0,
        alpha: float =0.3,
        beta: float =0.7,
        novel_scale: float =5.0,
        clip_head_path=None,
        use_soft_attention: bool = True,
        soft_attention_tau: float = 0.1,
        soft_category_topk: int = 3,
        soft_category_tau: float = 1.0,
        tpa_stabilization_steps: int = 0,
        tpa_task_gradient_scale: float = 1.0,
        tpa_eval_mode_scale: float = 1.0,
        teacher_rpsa: bool = False,
        teacher_rpsa_num_proposals: int = 64,
        teacher_rpsa_category_topk: int = 3,
        teacher_rpsa_confidence_threshold: float = 0.25,
        teacher_rpsa_margin_threshold: float = 0.05,
        teacher_rpsa_gt_iou_threshold: float = 0.5,
        teacher_rpsa_mode_temperature: float = 0.07,
        teacher_rpsa_novel_weight: float = 1.5,
        teacher_rpsa_novel_balanced: bool = False,
        teacher_rpsa_warmup_start: int = 7100,
        teacher_rpsa_warmup_iters: int = 7100,
    ):
        super().__init__()
        self.vlm_temperature = vlm_temperature
        self.alpha = alpha
        self.beta = beta
        self.novel_scale = novel_scale
        if inference_query_class_topk < 0:
            raise ValueError("inference_query_class_topk must be non-negative")
        self.inference_query_class_topk = int(inference_query_class_topk)
        self.use_soft_attention = use_soft_attention
        self.soft_attention_tau = soft_attention_tau
        if soft_category_topk < 1:
            raise ValueError(f"soft_category_topk must be >= 1, got {soft_category_topk}")
        if soft_category_tau <= 0:
            raise ValueError(f"soft_category_tau must be positive, got {soft_category_tau}")
        self.soft_category_topk = int(soft_category_topk)
        self.soft_category_tau = float(soft_category_tau)
        if tpa_stabilization_steps < 0:
            raise ValueError("tpa_stabilization_steps must be non-negative")
        if not 0.0 <= tpa_task_gradient_scale <= 1.0:
            raise ValueError("tpa_task_gradient_scale must be within [0, 1]")
        self.tpa_stabilization_steps = int(tpa_stabilization_steps)
        self.tpa_task_gradient_scale = float(tpa_task_gradient_scale)
        if not 0.0 <= tpa_eval_mode_scale <= 1.0:
            raise ValueError("tpa_eval_mode_scale must be within [0, 1]")
        self.tpa_eval_mode_scale = float(tpa_eval_mode_scale)
        self.tpa_stabilizing = False
        self.tpa_active_task_gradient_scale = 1.0
        self.teacher_rpsa = bool(teacher_rpsa)
        self.teacher_rpsa_num_proposals = int(teacher_rpsa_num_proposals)
        self.teacher_rpsa_category_topk = int(teacher_rpsa_category_topk)
        self.teacher_rpsa_confidence_threshold = float(
            teacher_rpsa_confidence_threshold
        )
        self.teacher_rpsa_margin_threshold = float(teacher_rpsa_margin_threshold)
        self.teacher_rpsa_gt_iou_threshold = float(teacher_rpsa_gt_iou_threshold)
        self.teacher_rpsa_mode_temperature = float(teacher_rpsa_mode_temperature)
        self.teacher_rpsa_novel_weight = float(teacher_rpsa_novel_weight)
        self.teacher_rpsa_novel_balanced = bool(teacher_rpsa_novel_balanced)
        self.teacher_rpsa_warmup_start = int(teacher_rpsa_warmup_start)
        self.teacher_rpsa_warmup_iters = int(teacher_rpsa_warmup_iters)
        if self.teacher_rpsa_novel_balanced and not self.teacher_rpsa:
            raise ValueError(
                "teacher_rpsa_novel_balanced requires teacher_rpsa=True"
            )
        if self.teacher_rpsa_num_proposals < 1:
            raise ValueError("teacher_rpsa_num_proposals must be positive")
        if self.teacher_rpsa_category_topk < 1:
            raise ValueError("teacher_rpsa_category_topk must be positive")
        if not 0.0 <= self.teacher_rpsa_confidence_threshold <= 1.0:
            raise ValueError("teacher_rpsa_confidence_threshold must be within [0, 1]")
        if not 0.0 <= self.teacher_rpsa_margin_threshold <= 1.0:
            raise ValueError("teacher_rpsa_margin_threshold must be within [0, 1]")
        if not 0.0 <= self.teacher_rpsa_gt_iou_threshold <= 1.0:
            raise ValueError("teacher_rpsa_gt_iou_threshold must be within [0, 1]")
        if self.teacher_rpsa_mode_temperature <= 0.0:
            raise ValueError("teacher_rpsa_mode_temperature must be positive")
        if self.teacher_rpsa_novel_weight <= 0.0:
            raise ValueError("teacher_rpsa_novel_weight must be positive")
        if self.teacher_rpsa_warmup_start < 0 or self.teacher_rpsa_warmup_iters < 0:
            raise ValueError("teacher RPSA warmup values must be non-negative")
        if self.teacher_rpsa and getattr(transformer, "use_rpsa", False):
            raise ValueError(
                "teacher_rpsa and the legacy transformer RPSA cannot be enabled together"
            )
        # define backbone and position embedding module
        self.backbone = backbone
        self.position_embedding = position_embedding

        # define neck module
        self.neck = neck

        # number of dynamic anchor boxes and embedding dimension
        self.num_queries = num_queries
        self.embed_dim = embed_dim

        # define transformer module
        self.transformer = transformer

        # define classification head and box head
        # self.class_embed = nn.Linear(embed_dim, num_classes)
        self.class_embed = classifier
        self.bbox_embed = MLP(embed_dim, embed_dim, 4, 3)
        self.num_classes = num_classes

        # where to calculate auxiliary loss in criterion
        self.aux_loss = aux_loss
        self.criterion = criterion

        # denoising
        # self.label_enc = nn.Embedding(num_classes, embed_dim)
        self.dn_number = dn_number
        self.label_noise_ratio = label_noise_ratio
        self.box_noise_scale = box_noise_scale

        # normalizer for input raw images
        self.device = device
        pixel_mean = torch.Tensor(pixel_mean).to(self.device).view(3, 1, 1)
        pixel_std = torch.Tensor(pixel_std).to(self.device).view(3, 1, 1)
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std

        # initialize weights
        # prior_prob = 0.01
        # bias_value = -math.log((1 - prior_prob) / prior_prob)
        # self.class_embed.bias.data = torch.ones(num_classes) * bias_value
        nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)
        for _, neck_layer in self.neck.named_modules():
            if isinstance(neck_layer, nn.Conv2d):
                nn.init.xavier_uniform_(neck_layer.weight, gain=1)
                nn.init.constant_(neck_layer.bias, 0)

        # if two-stage, the last class_embed and bbox_embed is for region proposal generation
        num_pred = transformer.decoder.num_layers + 1
        self.class_embed = nn.ModuleList([copy.deepcopy(self.class_embed) for i in range(num_pred)])
        self.bbox_embed = nn.ModuleList([copy.deepcopy(self.bbox_embed) for i in range(num_pred)])
        nn.init.constant_(self.bbox_embed[0].layers[-1].bias.data[2:], -2.0)

        # Share TPA across every class_embed copy so the deepcopy above does not
        # produce num_pred independent sets of TPA parameters / text buffers.
        if getattr(self.class_embed[0], "use_tpa", False):
            shared_tpa = self.class_embed[0].tpa
            shared_train_feats = self.class_embed[0].train_text_feats
            shared_eval_feats = self.class_embed[0].eval_text_feats
            for i in range(1, len(self.class_embed)):
                self.class_embed[i].tpa = shared_tpa
                self.class_embed[i].train_text_feats = shared_train_feats
                self.class_embed[i].eval_text_feats = shared_eval_feats
                self.class_embed[i].tpa.log_owner = (i == 0)

        # two-stage
        self.transformer.decoder.class_embed = self.class_embed
        self.transformer.decoder.bbox_embed = self.bbox_embed

        # hack implementation for two-stage
        for bbox_embed_layer in self.bbox_embed:
            nn.init.constant_(bbox_embed_layer.layers[-1].bias.data[2:], 0.0)

        # set topk boxes selected for inference
        self.select_box_nums_for_evaluation = select_box_nums_for_evaluation

        content_query_embedding = torch.tensor(np.load(query_path), dtype=torch.float32, device=device).contiguous()
        
        # Handle multi-prototype embeddings: support both 2D [C, D] and 3D [C, K, D] formats
        if content_query_embedding.ndim == 3:
            # Multi-prototype mode: [C, K, D] where C=num_classes, K=num_prompts (e.g., 8 prompts)
            num_classes_from_embed, num_prompts, feat_dim = content_query_embedding.shape
            print(f"[Multi-Prompt Mode] Loaded {num_classes_from_embed} classes × {num_prompts} prompts × {feat_dim}D")
            
            # Note: TPA will handle the prompts directly from text_classifier, not from here
            # For compatibility with existing code, create aggregated version using simple mean
            # This is only used for dimension compatibility and Fed Loss sampling
            content_query_embedding_agg = self._aggregate_prototypes(content_query_embedding, method='mean')
            # Note: num_prototypes is determined by TPA configuration, not stored here
            # Use aggregated version for compatibility
            content_query_embedding = content_query_embedding_agg
        else:
            # Standard mode: [C, D]
            feat_dim = content_query_embedding.shape[1]
        
        self.content_query_embedding = F.normalize(content_query_embedding, p=2, dim=1)

        eval_content_query_embedding = torch.tensor(np.load(eval_query_path), dtype=torch.float32, device=device).contiguous()
        if eval_content_query_embedding.ndim == 3:
            # Average eval embeddings if multi-prototype format
            eval_content_query_embedding = eval_content_query_embedding.mean(dim=1)
        self.eval_content_query_embedding = F.normalize(eval_content_query_embedding, p=2, dim=1)
        
        # self.eval_content_id = torch.tensor(np.load(eval_id_path), dtype=torch.int64, device=device)
        if vlm_query_path:
            vlm_content_query_embedding = torch.tensor(np.load(vlm_query_path), dtype=torch.float32, device=device).contiguous()# [1203, 768]
            if vlm_content_query_embedding.ndim == 3:
                vlm_content_query_embedding = vlm_content_query_embedding.mean(dim=1)  # VLM queries use average
            self.vlm_content_query_embedding = F.normalize(vlm_content_query_embedding, p=2, dim=1)
        
        _, feat_dim = self.content_query_embedding.shape
        self.content_layer = nn.Linear(feat_dim, embed_dim)

        self.use_fed_loss = use_fed_loss
        self.cluster_fed_loss = cluster_fed_loss
        self.fed_loss_num_cat = fed_loss_num_cat
        if self.use_fed_loss:
            freq_weight = load_class_freq(cat_freq_path, fed_loss_freq_weight)
            self.register_buffer('freq_weight', freq_weight)
        if self.cluster_fed_loss:
            self.cluster_label = np.load(cluster_label_path)

        self.score_ensemble = score_ensemble
        if self.score_ensemble:
            # This file contains serialized nn.Modules rather than only tensor
            # weights. PyTorch >=2.6 defaults to weights_only=True, which rejects
            # that trusted legacy format unless the intent is explicit.
            clip_head = load_trusted_torch_file(clip_head_path)
            self.identical, self.thead = clip_head[0]
            self.head = clip_head[1]

            # These modules define the frozen region teacher. They are used for
            # score ensembling at evaluation and, optionally, detached proposal
            # supervision during training; neither path may update the teacher.
            for teacher_module in (self.identical, self.thead, self.head):
                for parameter in teacher_module.parameters():
                    parameter.requires_grad_(False)

            self.seen_classes = json.load(open(seen_classes))
            self.all_classes = json.load(open(all_classes))
            idx = [self.all_classes.index(seen) for seen in self.seen_classes]
            self.base_idx = torch.zeros(len(self.all_classes), dtype=bool)
            self.base_idx[idx] = True
            if unseen_classes:
                self.unseen_classes = json.load(open(unseen_classes))
                idx_novel = [self.all_classes.index(unseen) for unseen in self.unseen_classes]
                self.novel_idx = torch.zeros(len(self.all_classes), dtype=bool)
                self.novel_idx[idx_novel] = True
            else:
                self.novel_idx = self.base_idx == False
        elif self.teacher_rpsa:
            raise ValueError("teacher_rpsa requires score_ensemble and its CLIP ROI head")
        if self.teacher_rpsa and not hasattr(self, "vlm_content_query_embedding"):
            raise ValueError("teacher_rpsa requires a full-vocabulary vlm_query_path")
        if (
            self.teacher_rpsa
            and self.vlm_content_query_embedding.shape[0] != self.num_classes
        ):
            raise ValueError(
                "teacher_rpsa requires one frozen CLIP text vector per detector "
                f"class, got {self.vlm_content_query_embedding.shape[0]} vectors "
                f"for {self.num_classes} classes"
            )
        self.save_dir = save_dir
        if self.save_dir:
            os.makedirs(self.save_dir, exist_ok=True)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.teacher_rpsa and hasattr(self, "identical"):
            # DINO.train() recursively switches every child to train mode. Put
            # the frozen CLIP region teacher back into deterministic eval mode.
            self.identical.eval()
            self.thead.eval()
            self.head.eval()
        return self
    
    def _aggregate_prototypes(self, embeddings, method='mean', region_feats=None, tau=0.1):
        """
        Aggregate multiple prototypes into a single embedding.
        
        Args:
            embeddings: [C, K, D] multi-prototype embeddings
            method: aggregation method - 'mean', 'max', 'soft_attention'
            region_feats: [B, N, D] region features for soft-attention (required for 'soft_attention')
            tau: temperature parameter for soft-attention (default: 0.1)
        
        Returns:
            aggregated: [C, D] aggregated embeddings for 'mean'/'max'
            OR similarity scores: [B, C, N] for 'soft_attention'
        
        Methods:
            - 'mean': Simple averaging (current baseline)
            - 'max': Max pooling across prototypes  
            - 'soft_attention': Soft-attention aggregation preserving semantic granularity
        """
        if method == 'mean':
            # Simple averaging: mathematically equivalent to pre-averaging before indexing
            return embeddings.mean(dim=1)
        
        elif method == 'max':
            # Max pooling: select strongest feature per dimension
            return embeddings.max(dim=1)[0]
        
        elif method == 'soft_attention':
            # Soft-attention aggregation: preserves semantic granularity
            # Formula: s_i,c = sum_k α_i,c,k * cos(f_i, t_c,k)
            # where α_i,c,k = softmax(cos(f_i, t_c,k) / τ)
            if region_feats is None:
                raise ValueError("region_feats required for soft_attention method")
            
            # Normalize embeddings and region features
            embeddings_norm = F.normalize(embeddings, p=2, dim=-1)  # [C, K, D]
            region_feats_norm = F.normalize(region_feats, p=2, dim=-1)  # [B, N, D]
            
            # Compute similarity: [B, N, D] @ [C, K, D]^T -> [B, C, N, K]
            sim = torch.einsum("bnd,ckd->bcnk", region_feats_norm, embeddings_norm)
            
            # Soft-attention weights: α_i,c,k = softmax(cos(f_i, t_c,k) / τ)
            alpha = F.softmax(sim / tau, dim=-1)  # [B, C, N, K]
            
            # Weighted aggregation: s_i,c = sum_k α_i,c,k * cos(f_i, t_c,k)
            sim_aggregated = (alpha * sim).sum(dim=-1)  # [B, C, N]
            
            return sim_aggregated
        
        else:
            raise ValueError(f"Unknown aggregation method: {method}")

    def filter_content_info(self, batched_inputs):
        """
        Make FedLoss class subset 'content_inds' consistent across GPUs:
        1) all_gather GT classes from all ranks
        2) sample on rank 0 (include GTs + negatives)
        3) broadcast 'content_inds' to all ranks
        4) remap per-image gt_classes to [0..len(content_inds)-1] using the same mapping
        """
        device = self.device
        # 频率权重（保持与原逻辑一致）
        freq_weight = self.freq_weight if self.freq_weight is not None else torch.ones(self.num_classes, device=device)

        # 本 rank 的 GT 类
        local_gt = []
        for target in batched_inputs:
            local_gt.append(target["instances"].gt_classes.to(device))
        if len(local_gt) > 0:
            local_gt = torch.unique(torch.cat(local_gt))
        else:
            local_gt = torch.empty(0, dtype=torch.long, device=device)

        # 跨卡收集所有 GT 类（去重）
        if is_dist_avail_and_initialized():
            world_size = dist.get_world_size()
            # 先收集长度，再收集内容（避免不同长度 all_gather 失败）
            local_len = torch.tensor([local_gt.numel()], device=device, dtype=torch.long)
            lens = [torch.zeros_like(local_len) for _ in range(world_size)]
            dist.all_gather(lens, local_len)
            max_len = int(torch.stack(lens).max().item())
            pad = max_len - local_gt.numel()
            padded = torch.cat([local_gt, torch.full((pad,), -1, device=device, dtype=torch.long)]) if pad > 0 else local_gt
            gathered = [torch.empty_like(padded) for _ in range(world_size)]
            dist.all_gather(gathered, padded)
            all_gt = torch.unique(torch.cat(gathered))
            all_gt = all_gt[all_gt >= 0]  # 去掉 padding 的 -1
        else:
            all_gt = local_gt

        # 仅在 rank 0 进行采样；其它 rank 准备占位
        need_sample = (not is_dist_avail_and_initialized()) or (dist.get_rank() == 0)

        if need_sample:
            if self.cluster_fed_loss:
                content_inds = get_cluster_fed_loss_inds(
                    all_gt,
                    num_sample_cats=self.fed_loss_num_cat,
                    C=self.num_classes,
                    weight=freq_weight,
                    cluster_label=self.cluster_label,
                )
            else:
                content_inds = get_fed_loss_inds(
                    all_gt,
                    num_sample_cats=self.fed_loss_num_cat,
                    C=self.num_classes,
                    weight=freq_weight,
                )
            # 保证类型与设备
            content_inds = content_inds.to(device=device, dtype=torch.long)
        else:
            # 用固定长度占位，等会儿接收广播
            content_inds = torch.zeros(self.fed_loss_num_cat, device=device, dtype=torch.long)

        # 广播到所有 GPU（若未分布式则跳过）
        if is_dist_avail_and_initialized():
            dist.broadcast(content_inds, src=0)

        # === 之后保持你原有的映射逻辑：将 gt_classes 映射到 [0..M-1] ===
        convert_map = torch.ones(self.num_classes, dtype=torch.int64, device=device) * -1
        # content_inds[i] -> i
        convert_map[content_inds] = torch.arange(content_inds.numel(), device=device, dtype=torch.int64)

        for idx, target in enumerate(batched_inputs):
            cats = target["instances"].gt_classes.to(device)
            batched_inputs[idx]["instances"].gt_classes = convert_map[cats]
        
        # DEBUG（可选）：多卡一致性哈希（仅前200 iter打印）
        if is_dist_avail_and_initialized():
            import hashlib
            global_step = getattr(self, "_debug_step", 0)
            if global_step < 200 and global_step % 10 == 0:
                h_local = hashlib.md5(content_inds.detach().cpu().numpy().tobytes()).hexdigest()[:8]
                hashes = [None for _ in range(dist.get_world_size())]
                dist.all_gather_object(hashes, h_local)
                if dist.get_rank() == 0:
                    print(f"[DDP-OK] step={global_step:06d} content_inds hashes: {hashes}")
                self._debug_step = global_step + 1

        return content_inds, batched_inputs

    def _teacher_rpsa_scale(self, iteration: int) -> float:
        if iteration < self.teacher_rpsa_warmup_start:
            return 0.0
        if self.teacher_rpsa_warmup_iters <= 0:
            return 1.0
        progress = (
            iteration - self.teacher_rpsa_warmup_start
        ) / float(self.teacher_rpsa_warmup_iters)
        return min(max(progress, 0.0), 1.0)

    def _compute_teacher_rpsa(
        self,
        *,
        encoder_regions,
        proposal_boxes,
        clip_features,
        target_boxes,
        global_gt_classes,
    ):
        """Build detached full-vocabulary CLIP targets for proposal modes."""
        proposal_count = min(
            self.teacher_rpsa_num_proposals,
            encoder_regions.shape[1],
            proposal_boxes.shape[1],
        )
        encoder_regions = encoder_regions[:, :proposal_count]
        proposal_boxes = proposal_boxes[:, :proposal_count]

        # The CLIP branch is a frozen teacher. Only the detector projection and
        # live TPA prototypes below receive gradients.
        with torch.no_grad():
            teacher_regions = self.extract_region_feature(
                clip_features,
                proposal_boxes.detach(),
                "p3",
            ).float()
            teacher_regions = F.normalize(teacher_regions, dim=-1)
            teacher_category_logits = torch.einsum(
                "bmd,cd->bmc",
                teacher_regions,
                self.vlm_content_query_embedding.to(
                    device=teacher_regions.device,
                    dtype=teacher_regions.dtype,
                ),
            ) * self.vlm_temperature
            teacher_category_probs = teacher_category_logits.softmax(dim=-1)
            candidate_probs, candidate_ids = teacher_category_probs.topk(
                min(self.teacher_rpsa_category_topk, self.num_classes),
                dim=-1,
            )
            confidence = candidate_probs[..., 0]
            if candidate_probs.shape[-1] > 1:
                margin = candidate_probs[..., 0] - candidate_probs[..., 1]
            else:
                margin = candidate_probs[..., 0]

        batch_size = encoder_regions.shape[0]
        matched_gt = torch.zeros(
            batch_size,
            proposal_count,
            dtype=torch.bool,
            device=encoder_regions.device,
        )
        allowed_categories = torch.ones_like(candidate_ids, dtype=torch.bool)
        proposal_xyxy = box_cxcywh_to_xyxy(proposal_boxes.detach())

        for batch_index in range(batch_size):
            boxes = target_boxes[batch_index]
            labels = global_gt_classes[batch_index]
            if boxes.numel() == 0:
                continue
            ious = torchvision.ops.box_iou(
                proposal_xyxy[batch_index].float(),
                box_cxcywh_to_xyxy(boxes.detach()).float(),
            )
            if self.teacher_rpsa_novel_balanced:
                # Use at most one proposal per GT and at most one GT per
                # proposal. Processing the strongest pairs first resolves the
                # rare case where two GT boxes select the same proposal.
                best_iou, best_proposal = ious.max(dim=0)
                used_proposals = set()
                for gt_index in torch.argsort(best_iou, descending=True).tolist():
                    if best_iou[gt_index] < self.teacher_rpsa_gt_iou_threshold:
                        break
                    proposal_index = int(best_proposal[gt_index])
                    if proposal_index in used_proposals:
                        continue
                    used_proposals.add(proposal_index)
                    matched_gt[batch_index, proposal_index] = True
                    candidate_ids[batch_index, proposal_index, 0] = labels[gt_index]
                    allowed_categories[batch_index, proposal_index] = False
                    allowed_categories[batch_index, proposal_index, 0] = True
            else:
                best_iou, best_gt = ious.max(dim=1)
                matched = best_iou >= self.teacher_rpsa_gt_iou_threshold
                if not matched.any():
                    continue
                matched_gt[batch_index, matched] = True
                candidate_ids[batch_index, matched, 0] = labels[best_gt[matched]]
                allowed_categories[batch_index, matched] = False
                allowed_categories[batch_index, matched, 0] = True

        pseudo_valid = (
            (confidence >= self.teacher_rpsa_confidence_threshold)
            & (margin >= self.teacher_rpsa_margin_threshold)
        )
        encoder_classifier = self.transformer.decoder.class_embed[
            self.transformer.decoder.num_layers
        ]
        student_regions = encoder_classifier.linear(encoder_regions).float()

        # Compute live TPA modes only for the union of teacher candidates in
        # this batch. This exposes novel categories to RPSA without expanding
        # the focal/FedLoss vocabulary or paying for all 1,203 classes in TPA.
        unique_ids, inverse = torch.unique(
            candidate_ids.reshape(-1),
            sorted=True,
            return_inverse=True,
        )
        full_text_feats = encoder_classifier._maybe_move_text_feats(training=True)
        candidate_bank, _ = encoder_classifier.tpa(
            full_text_feats[unique_ids],
            with_loss=False,
            advance_step=False,
            apply_dropout=False,
            update_monitor_state=False,
        )
        candidate_prototypes = candidate_bank[inverse].view(
            *candidate_ids.shape,
            candidate_bank.shape[-2],
            candidate_bank.shape[-1],
        ).float()

        candidate_novel = self.novel_idx.to(candidate_ids.device)[candidate_ids]
        pseudo_novel = pseudo_valid & candidate_novel[..., 0] & ~matched_gt

        if self.teacher_rpsa_novel_balanced:
            # Only novel pseudo categories may contribute. A top-1 novel
            # prediction guarantees at least one allowed category here.
            allowed_categories = torch.where(
                pseudo_novel.unsqueeze(-1),
                allowed_categories & candidate_novel,
                allowed_categories,
            )
            selected_pseudo = pseudo_novel
            valid_mask = matched_gt | selected_pseudo
            proposal_weights = None
            group_masks = torch.stack((matched_gt, selected_pseudo), dim=0)
            group_weights = encoder_regions.new_tensor(
                [1.0, self.teacher_rpsa_novel_weight]
            )
        else:
            selected_pseudo = pseudo_valid & ~matched_gt
            valid_mask = matched_gt | pseudo_valid
            proposal_weights = torch.where(
                pseudo_novel,
                encoder_regions.new_tensor(self.teacher_rpsa_novel_weight),
                encoder_regions.new_tensor(1.0),
            ).float()
            group_masks = None
            group_weights = None

        loss, stats = teacher_routed_mode_alignment(
            student_regions,
            teacher_regions,
            candidate_prototypes,
            candidate_category_probs=candidate_probs,
            valid_mask=valid_mask,
            allowed_category_mask=allowed_categories,
            proposal_weights=proposal_weights,
            group_masks=group_masks,
            group_weights=group_weights,
            temperature=self.teacher_rpsa_mode_temperature,
        )
        with torch.no_grad():
            group_losses = stats.pop("teacher_rpsa_group_losses", None)
            group_active = stats.pop("teacher_rpsa_group_active", None)
            valid_count = valid_mask.sum().clamp_min(1)
            stats.update(
                {
                    "teacher_rpsa_gt_ratio": matched_gt.float().mean(),
                    "teacher_rpsa_pseudo_ratio": selected_pseudo.float().mean(),
                    "teacher_rpsa_novel_ratio": (
                        (pseudo_novel & valid_mask).sum() / valid_count
                    ),
                    "teacher_rpsa_confidence": (
                        confidence * valid_mask.to(confidence.dtype)
                    ).sum() / valid_count,
                    "teacher_rpsa_margin": (
                        margin * valid_mask.to(margin.dtype)
                    ).sum() / valid_count,
                    "rpsa_tokens": encoder_regions.new_tensor(
                        float(proposal_count)
                    ),
                }
            )
            if group_losses is not None:
                stats.update(
                    {
                        "teacher_rpsa_active_groups": group_active.float().sum(),
                        "teacher_rpsa_gt_group_loss": group_losses[0],
                        "teacher_rpsa_novel_group_loss": group_losses[1],
                        "teacher_rpsa_gt_anchors": matched_gt.float().sum(dim=1).mean(),
                        "teacher_rpsa_novel_proposals": selected_pseudo.float().sum(dim=1).mean(),
                    }
                )
        return loss, stats


    def forward(self, batched_inputs):
        """Forward function of `DINO` which excepts a list of dict as inputs.

        Args:
            batched_inputs (List[dict]): A list of instance dict, and each instance dict must consists of:
                - dict["image"] (torch.Tensor): The unnormalized image tensor.
                - dict["height"] (int): The original image height.
                - dict["width"] (int): The original image width.
                - dict["instance"] (detectron2.structures.Instances):
                    Image meta informations and ground truth boxes and labels during training.
                    Please refer to
                    https://detectron2.readthedocs.io/en/latest/modules/structures.html#detectron2.structures.Instances
                    for the basic usage of Instances.

        Returns:
            dict: Returns a dict with the following elements:
                - dict["pred_logits"]: the classification logits for all queries (anchor boxes in DAB-DETR).
                            with shape ``[batch_size, num_queries, num_classes]``
                - dict["pred_boxes"]: The normalized boxes coordinates for all queries in format
                    ``(x, y, w, h)``. These values are normalized in [0, 1] relative to the size of
                    each individual image (disregarding possible padding). See PostProcess for information
                    on how to retrieve the unnormalized bounding box.
                - dict["aux_outputs"]: Optional, only returned when auxilary losses are activated. It is a list of
                            dictionnaries containing the two above keys for each decoder layer.
        """
        if self.save_dir:
            filename = batched_inputs[0]['file_name'].split('/')[-1].replace('jpg', 'pth')

        images = self.preprocess_image(batched_inputs)

        content_inds = None
        global_gt_classes = None
        if self.training:
            batch_size, _, H, W = images.tensor.shape
            img_masks = images.tensor.new_ones(batch_size, H, W)
            for img_id in range(batch_size):
                img_h, img_w = batched_inputs[img_id]["instances"].image_size
                img_masks[img_id, :img_h, :img_w] = 0
            global_gt_classes = [
                sample["instances"].gt_classes.to(self.device).clone()
                for sample in batched_inputs
            ]
            if self.use_fed_loss:
                content_inds, batched_inputs = self.filter_content_info(batched_inputs)
        else:
            batch_size, _, H, W = images.tensor.shape
            img_masks = images.tensor.new_zeros(batch_size, H, W)

        # original features
        if self.score_ensemble:
            features, features_wonorm = self.backbone(images.tensor)  # output feature dict
        else:
            features = self.backbone(images.tensor)  # output feature dict

        # project backbone features to the reuired dimension of transformer
        # we use multi-scale features in DINO
        multi_level_feats = self.neck(features)
        multi_level_masks = []
        multi_level_position_embeddings = []
        for feat in multi_level_feats:
            multi_level_masks.append(
                F.interpolate(img_masks[None], size=feat.shape[-2:]).to(torch.bool).squeeze(0)
            )
            multi_level_position_embeddings.append(self.position_embedding(multi_level_masks[-1]))
        
        # === Build content_query_embeds using TPA prototypes (replacing .npy embeddings) ===
        # Run TPA exactly once per forward and broadcast the same prototypes (same dropout
        # draw) to every class_embed layer + the query-init path. This ensures every
        # consumer sees an identical prototype tensor and APR is accumulated only once.
        if hasattr(self.transformer.decoder.class_embed[0], 'use_tpa') and self.transformer.decoder.class_embed[0].use_tpa:
            text_classifier = self.transformer.decoder.class_embed[0]
            text_feats = text_classifier._maybe_move_text_feats(training=self.training)
            if content_inds is not None:
                text_feats = text_feats[content_inds]
            # [C,K,D_text]. At inference Eq. (7) says the category bank is
            # computed once and cached; do not rerun TPA for every image.
            if self.training:
                shared_prototypes, shared_apr_loss = text_classifier.tpa(
                    text_feats,
                    with_loss=True,
                    advance_step=getattr(self, "tpa_advance_step", True),
                )
            else:
                cached = text_classifier._cached_eval
                cache_valid = (
                    cached is not None
                    and cached.device == text_feats.device
                    and cached.shape[0] == text_feats.shape[0]
                )
                if not cache_valid:
                    shared_prototypes, _ = text_classifier.tpa(
                        text_feats, with_loss=False
                    )
                    text_classifier._cached_eval = shared_prototypes.detach()
                else:
                    shared_prototypes = cached
                shared_apr_loss = None

                # Same-checkpoint counterfactual: scale zero makes every
                # class's K slots identical for both query fusion and final
                # classification; scale one is the untouched trained model.
                shared_prototypes = prototype_eval_mode_view(
                    shared_prototypes,
                    self.tpa_eval_mode_scale,
                )

            current_iter = 0
            if self.training:
                try:
                    current_iter = get_event_storage().iter
                except AssertionError:
                    pass
            self.tpa_stabilizing = bool(
                self.training and current_iter < self.tpa_stabilization_steps
            )
            if not self.training:
                self.tpa_active_task_gradient_scale = 1.0
            elif self.tpa_stabilizing:
                self.tpa_active_task_gradient_scale = 0.0
            else:
                self.tpa_active_task_gradient_scale = self.tpa_task_gradient_scale
            task_prototypes = prototype_task_view(
                shared_prototypes,
                iteration=current_iter,
                stabilization_steps=self.tpa_stabilization_steps,
                task_gradient_scale=self.tpa_task_gradient_scale,
                training=self.training,
            )
            # Broadcast to every class_embed copy so they reuse the same prototype tensor.
            for ce in self.transformer.decoder.class_embed:
                ce.set_external_prototypes(task_prototypes, shared_apr_loss)

            # Project to decoder dim for query init
            proto_ckd = self.content_layer(task_prototypes.view(-1, task_prototypes.size(-1))).view(
                task_prototypes.size(0), task_prototypes.size(1), -1
            )
            proto_ckd = F.normalize(proto_ckd, p=2, dim=-1)
            raw_content_query_embeds = proto_ckd  # [C,K,embed_dim]
        else:
            # Fallback to aggregated version
            content_query_embedding = self.content_layer(self.content_query_embedding)
            content_query_embedding = F.normalize(content_query_embedding, p=2, dim=1)
            raw_content_query_embeds = content_query_embedding.unsqueeze(1)

        # denoising preprocessing
        # prepare label query embedding
        if self.training:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
            targets = self.prepare_targets(gt_instances)
            cdn_num_classes = self.fed_loss_num_cat if self.use_fed_loss else self.num_classes
            input_query_label, input_query_bbox, attn_mask, dn_meta = self.prepare_for_cdn(
                targets,
                dn_number=self.dn_number,
                label_noise_ratio=self.label_noise_ratio,
                box_noise_scale=self.box_noise_scale,
                num_queries=self.num_queries,
                num_classes=cdn_num_classes,
                hidden_dim=self.embed_dim,
                # label_enc=self.label_enc,
                content_query_embeds=raw_content_query_embeds,
            )
        else:
            input_query_label, input_query_bbox, attn_mask, dn_meta = None, None, None, None
        query_embeds = (input_query_label, input_query_bbox)

        # Set soft-attention parameters for transformer if using multi-prototype mode
        if hasattr(self, 'use_soft_attention') and self.use_soft_attention and raw_content_query_embeds.ndim == 3:
            self.transformer.use_soft_attention = self.use_soft_attention
            self.transformer.soft_attention_tau = self.soft_attention_tau
            self.transformer.soft_category_topk = self.soft_category_topk
            self.transformer.soft_category_tau = self.soft_category_tau

        # feed into transformer
        (
            inter_states,
            init_reference,
            inter_references,
            enc_state,
            enc_reference,  # [0..1]
            apr_loss,
        ) = self.transformer(
            multi_level_feats,
            multi_level_masks,
            multi_level_position_embeddings,
            query_embeds,
            attn_masks=[attn_mask, None],
            content_query_embeds=raw_content_query_embeds,  # Pass raw multi-prototype embeddings
            content_inds=content_inds, 
        )

        teacher_rpsa_loss = None
        teacher_rpsa_scale = 0.0
        if self.training and self.teacher_rpsa:
            try:
                teacher_rpsa_iteration = get_event_storage().iter
            except AssertionError:
                teacher_rpsa_iteration = 0
            teacher_rpsa_scale = self._teacher_rpsa_scale(teacher_rpsa_iteration)
            if teacher_rpsa_scale > 0.0:
                teacher_rpsa_loss, teacher_rpsa_stats = self._compute_teacher_rpsa(
                    encoder_regions=enc_state,
                    proposal_boxes=enc_reference,
                    clip_features=features_wonorm,
                    target_boxes=[target["boxes"] for target in targets],
                    global_gt_classes=global_gt_classes,
                )
            else:
                teacher_rpsa_loss = enc_state.sum() * 0.0
                teacher_rpsa_stats = {
                    "teacher_rpsa_active": enc_state.new_tensor(0.0),
                    "teacher_rpsa_valid_proposals": enc_state.new_tensor(0.0),
                    "teacher_rpsa_valid_ratio": enc_state.new_tensor(0.0),
                    "rpsa_tokens": enc_state.new_tensor(0.0),
                }
            self.transformer.rpsa_last_loss = teacher_rpsa_loss.detach()
            self.transformer.rpsa_last_stats = teacher_rpsa_stats
        # hack implementation for distributed training
        # inter_states[0] += self.label_enc.weight[0, 0] * 0.0
        inter_states[0] += self.content_layer.weight[0, 0] * 0.0

        # Calculate output coordinates and classes.
        outputs_classes = []
        outputs_coords = []
        for lvl in range(inter_states.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            outputs_class = self.class_embed[lvl](inter_states[lvl], content_inds=content_inds)
            tmp = self.bbox_embed[lvl](inter_states[lvl])
            if reference.shape[-1] == 4:
                tmp += reference
            else:
                assert reference.shape[-1] == 2
                tmp[..., :2] += reference
            outputs_coord = tmp.sigmoid()
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)
        outputs_class = torch.stack(outputs_classes)
        # tensor shape: [num_decoder_layers, bs, num_query, num_classes]
        outputs_coord = torch.stack(outputs_coords)
        # tensor shape: [num_decoder_layers, bs, num_query, 4]

        # denoising postprocessing
        if dn_meta is not None:
            outputs_class, outputs_coord = self.dn_post_process(
                outputs_class, outputs_coord, dn_meta
            )

        # prepare for loss computation
        output = {"pred_logits": outputs_class[-1], "pred_boxes": outputs_coord[-1]}
        if self.aux_loss:
            output["aux_outputs"] = self._set_aux_loss(outputs_class, outputs_coord)

        # prepare two stage output
        interm_coord = enc_reference
        interm_class = self.transformer.decoder.class_embed[-1](enc_state, content_inds=content_inds)
        output["enc_outputs"] = {"pred_logits": interm_class, "pred_boxes": interm_coord}

        if self.training:
            loss_dict = self.criterion(output, targets, dn_meta)
            # === 1️⃣ 添加 APR 损失（保持原逻辑） ===
            if apr_loss is not None:
                loss_dict["loss_apr"] = apr_loss

            # === 2️⃣ 添加 RPSA 损失（Region–Prototype Semantic Alignment） ===
            # Formal runs must never continue silently without Eq. (6).
            if self.teacher_rpsa or getattr(self.transformer, "use_rpsa", False):
                dec_encoder = self.transformer.decoder.class_embed[self.transformer.decoder.num_layers]
                rpsa_loss_value = (
                    teacher_rpsa_loss
                    if self.teacher_rpsa
                    else getattr(dec_encoder, "rpsa_loss", None)
                )
                if rpsa_loss_value is None:
                    raise RuntimeError("RPSA is enabled but no loss was produced")
                if not torch.isfinite(rpsa_loss_value).all():
                    raise FloatingPointError(
                        f"RPSA produced a non-finite loss: {rpsa_loss_value.detach()}"
                    )

                loss_dict["loss_rpsa"] = rpsa_loss_value

                storage = None
                try:
                    storage = get_event_storage()
                    current_iter = storage.iter
                except AssertionError:
                    current_iter = 0

                if self.teacher_rpsa:
                    schedule_scale = teacher_rpsa_scale
                else:
                    warmup_iters = getattr(self.transformer, "rpsa_warmup_iters", 0)
                    warmup_start = getattr(self.transformer, "rpsa_warmup_start", 0)
                    warmup_init = getattr(self.transformer, "rpsa_warmup_init_scale", 0.0)
                    warmup_power = getattr(self.transformer, "rpsa_warmup_power", 1.0)
                    schedule_scale = 1.0
                    if warmup_iters > 0:
                        if current_iter < warmup_start:
                            schedule_scale = warmup_init
                        elif current_iter < warmup_start + warmup_iters:
                            progress = (current_iter - warmup_start) / float(max(warmup_iters, 1))
                            schedule_scale = warmup_init + (progress ** warmup_power) * (1.0 - warmup_init)

                loss_dict["loss_rpsa"] = loss_dict["loss_rpsa"] * schedule_scale

                if storage is not None:
                    storage.put_scalar("loss_rpsa_scale", float(schedule_scale), smoothing_hint=False)

            # === 3️⃣ FedLoss、主损失加权保持一致 ===
            weight_dict = self.criterion.weight_dict
            for k in loss_dict.keys():
                if k in weight_dict:
                    loss_dict[k] *= weight_dict[k]
            return loss_dict
        else:
            box_cls = output["pred_logits"]
            box_pred = output["pred_boxes"]
            if self.save_dir and not self.score_ensemble:
                save_output = {}
                save_output["pred_logits"] = copy.deepcopy(output["pred_logits"]).cpu()
                save_output["pred_boxes"] = copy.deepcopy(output["pred_boxes"]).cpu()
                torch.save(save_output, os.path.join(self.save_dir, filename))
            if self.score_ensemble:
                roi_features_ori = self.extract_region_feature(features_wonorm, box_pred, 'p3')

                if self.save_dir:
                    save_output = {}
                    save_output["pred_logits"] = copy.deepcopy(output["pred_logits"]).cpu()
                    save_output["roi_features_ori"] = copy.deepcopy(roi_features_ori).cpu()# [1, 900, 768]
                    save_output["pred_boxes"] = copy.deepcopy(output["pred_boxes"]).cpu()
                    torch.save(save_output, os.path.join(self.save_dir, filename))

                cls_score = box_cls.sigmoid()
                vlm_score = roi_features_ori @ self.vlm_content_query_embedding.t() * self.vlm_temperature
                vlm_score = vlm_score.softmax(dim=-1)
                cls_score[:, :, self.base_idx] = cls_score[:, :, self.base_idx] ** (
                        1 - self.alpha) * vlm_score[:, :, self.base_idx] ** self.alpha
                cls_score[:, :, self.novel_idx] = cls_score[:, :, self.novel_idx] ** (
                        1 - self.beta) * vlm_score[:, :, self.novel_idx] ** self.beta 
                cls_score[:, :, self.novel_idx] = cls_score[:, :, self.novel_idx] * self.novel_scale
                box_cls = cls_score
                results = self.inference(box_cls, box_pred, images.image_sizes, wo_sigmoid=True)
            else:
                results = self.inference(box_cls, box_pred, images.image_sizes)
            processed_results = []
            for results_per_image, input_per_image, image_size in zip(
                results, batched_inputs, images.image_sizes
            ):
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])
                r = detector_postprocess(results_per_image, height, width)
                processed_results.append({"instances": r})
            return processed_results
    
    def extract_region_feature(self, features, bbox, layer_name):
        if layer_name == 'p2':
            h, w = features['p2'].shape[-2:]# 50 75
        elif layer_name == 'p3':
            h, w = features['p3'].shape[-2:]# 50 75

        rpn_boxes = box_cxcywh_to_xyxy(bbox)
        rpn_boxes = torch.clamp(rpn_boxes, min=0, max=1)
        for i in range(len(rpn_boxes)):
            rpn_boxes[i][:,[0,2]] = rpn_boxes[i][:,[0,2]] * w
            rpn_boxes[i][:,[1,3]] = rpn_boxes[i][:,[1,3]] * h
        rpn_boxes = [rpn_box for rpn_box in rpn_boxes]
       
        bs = len(rpn_boxes)
        roi_features = torchvision.ops.roi_align(
            # hid,# [2, 768, 50, 66]
            features['p2'] if layer_name == 'p2' else features['p3'],
            rpn_boxes,
            output_size=(15, 15),
            spatial_scale=1.0,
            aligned=True)  # (bs * num_queries, c, 14, 14) [1800, 768, 30, 30]

        if layer_name == 'p2':
            roi_features = self.backbone.downsample_layers[3](roi_features)# [33, 768, 30, 30]->[33, 1536, 15, 15] 
            roi_features = self.backbone.stages[3](roi_features)# [33, 1536, 15, 15]->[33, 1536, 15, 15]
        roi_features = self.identical(roi_features)# [900, 1536, 15, 15]
        roi_features = self.thead(roi_features)# [900, 1536]
        roi_features = self.head(roi_features)# [900, 768] TODO:
        roi_features = roi_features.reshape(bs, -1, roi_features.shape[-1])
        roi_features = nn.functional.normalize(roi_features, dim=-1)# [1, 900, 768]
        return roi_features


    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [
            {"pred_logits": a, "pred_boxes": b}
            for a, b in zip(outputs_class[:-1], outputs_coord[:-1])
        ]

    def prepare_for_cdn(
        self,
        targets,
        dn_number,
        label_noise_ratio,
        box_noise_scale,
        num_queries,
        num_classes,
        hidden_dim,
        label_enc=None,
        content_query_embeds=None,
        convert_map=None,
    ):
        """
        A major difference of DINO from DN-DETR is that the author process pattern embedding pattern embedding
            in its detector
        forward function and use learnable tgt embedding, so we change this function a little bit.
        :param dn_args: targets, dn_number, label_noise_ratio, box_noise_scale
        :param training: if it is training or inference
        :param num_queries: number of queires
        :param num_classes: number of classes
        :param hidden_dim: transformer hidden dim
        :param label_enc: encode labels in dn
        :return:
        """
        if dn_number <= 0:
            return None, None, None, None
            # positive and negative dn queries
        dn_number = dn_number * 2
        known = [(torch.ones_like(t["labels"])).cuda() for t in targets]
        batch_size = len(known)
        known_num = [sum(k) for k in known]
        if int(max(known_num)) == 0:
            return None, None, None, None

        dn_number = dn_number // (int(max(known_num) * 2))

        if dn_number == 0:
            dn_number = 1
        unmask_bbox = unmask_label = torch.cat(known)
        labels = torch.cat([t["labels"] for t in targets])
        boxes = torch.cat([t["boxes"] for t in targets])
        batch_idx = torch.cat(
            [torch.full_like(t["labels"].long(), i) for i, t in enumerate(targets)]
        )

        known_indice = torch.nonzero(unmask_label + unmask_bbox)
        known_indice = known_indice.view(-1)

        known_indice = known_indice.repeat(2 * dn_number, 1).view(-1)
        known_labels = labels.repeat(2 * dn_number, 1).view(-1)
        known_bid = batch_idx.repeat(2 * dn_number, 1).view(-1)
        known_bboxs = boxes.repeat(2 * dn_number, 1)
        known_labels_expaned = known_labels.clone()
        known_bbox_expand = known_bboxs.clone()

        if label_noise_ratio > 0:
            p = torch.rand_like(known_labels_expaned.float())
            chosen_indice = torch.nonzero(p < (label_noise_ratio * 0.5)).view(
                -1
            )  # half of bbox prob
            new_label = torch.randint_like(
                chosen_indice, 0, num_classes
            )  # randomly put a new one here
            known_labels_expaned.scatter_(0, chosen_indice, new_label)
        single_padding = int(max(known_num))

        pad_size = int(single_padding * 2 * dn_number)
        positive_idx = (
            torch.tensor(range(len(boxes))).long().cuda().unsqueeze(0).repeat(dn_number, 1)
        )
        positive_idx += (torch.tensor(range(dn_number)) * len(boxes) * 2).long().cuda().unsqueeze(1)
        positive_idx = positive_idx.flatten()
        negative_idx = positive_idx + len(boxes)
        if box_noise_scale > 0:
            known_bbox_ = torch.zeros_like(known_bboxs)
            known_bbox_[:, :2] = known_bboxs[:, :2] - known_bboxs[:, 2:] / 2
            known_bbox_[:, 2:] = known_bboxs[:, :2] + known_bboxs[:, 2:] / 2

            diff = torch.zeros_like(known_bboxs)
            diff[:, :2] = known_bboxs[:, 2:] / 2
            diff[:, 2:] = known_bboxs[:, 2:] / 2

            rand_sign = (
                torch.randint_like(known_bboxs, low=0, high=2, dtype=torch.float32) * 2.0 - 1.0
            )
            rand_part = torch.rand_like(known_bboxs)
            rand_part[negative_idx] += 1.0
            rand_part *= rand_sign
            known_bbox_ = known_bbox_ + torch.mul(rand_part, diff).cuda() * box_noise_scale
            known_bbox_ = known_bbox_.clamp(min=0.0, max=1.0)
            known_bbox_expand[:, :2] = (known_bbox_[:, :2] + known_bbox_[:, 2:]) / 2
            known_bbox_expand[:, 2:] = known_bbox_[:, 2:] - known_bbox_[:, :2]

        m = known_labels_expaned.long().to("cuda")
        # input_label_embed = label_enc(m)
        
        if content_query_embeds is not None:
            if content_query_embeds.ndim == 3:
                # Multi-prototype mode: [C, K, embed_dim]
                # For CDN, we need to aggregate prototypes first
                content_query_embeds_agg = content_query_embeds.mean(dim=1)  # [C, embed_dim]
                input_label_content = content_query_embeds_agg[m]
            else:
                # Single-prototype mode: [C, embed_dim]
                input_label_content = content_query_embeds[m]
            input_label_embed = input_label_content

        input_bbox_embed = inverse_sigmoid(known_bbox_expand)

        padding_label = torch.zeros(pad_size, hidden_dim).cuda()
        padding_bbox = torch.zeros(pad_size, 4).cuda()

        input_query_label = padding_label.repeat(batch_size, 1, 1)
        input_query_bbox = padding_bbox.repeat(batch_size, 1, 1)

        map_known_indice = torch.tensor([]).to("cuda")
        if len(known_num):
            map_known_indice = torch.cat(
                [torch.tensor(range(num)) for num in known_num]
            )  # [1,2, 1,2,3]
            map_known_indice = torch.cat(
                [map_known_indice + single_padding * i for i in range(2 * dn_number)]
            ).long()
        if len(known_bid):
            input_query_label[(known_bid.long(), map_known_indice)] = input_label_embed
            input_query_bbox[(known_bid.long(), map_known_indice)] = input_bbox_embed

        tgt_size = pad_size + num_queries
        attn_mask = torch.ones(tgt_size, tgt_size).to("cuda") < 0
        # match query cannot see the reconstruct
        attn_mask[pad_size:, :pad_size] = True
        # reconstruct cannot see each other
        for i in range(dn_number):
            if i == 0:
                attn_mask[
                    single_padding * 2 * i : single_padding * 2 * (i + 1),
                    single_padding * 2 * (i + 1) : pad_size,
                ] = True
            if i == dn_number - 1:
                attn_mask[
                    single_padding * 2 * i : single_padding * 2 * (i + 1), : single_padding * i * 2
                ] = True
            else:
                attn_mask[
                    single_padding * 2 * i : single_padding * 2 * (i + 1),
                    single_padding * 2 * (i + 1) : pad_size,
                ] = True
                attn_mask[
                    single_padding * 2 * i : single_padding * 2 * (i + 1), : single_padding * 2 * i
                ] = True

        dn_meta = {
            "single_padding": single_padding * 2,
            "dn_num": dn_number,
        }

        return input_query_label, input_query_bbox, attn_mask, dn_meta

    def dn_post_process(self, outputs_class, outputs_coord, dn_metas):
        if dn_metas and dn_metas["single_padding"] > 0:
            padding_size = dn_metas["single_padding"] * dn_metas["dn_num"]
            output_known_class = outputs_class[:, :, :padding_size, :]
            output_known_coord = outputs_coord[:, :, :padding_size, :]
            outputs_class = outputs_class[:, :, padding_size:, :]
            outputs_coord = outputs_coord[:, :, padding_size:, :]

            out = {"pred_logits": output_known_class[-1], "pred_boxes": output_known_coord[-1]}
            if self.aux_loss:
                out["aux_outputs"] = self._set_aux_loss(output_known_class, output_known_coord)
            dn_metas["output_known_lbs_bboxes"] = out
        return outputs_class, outputs_coord

    def preprocess_image(self, batched_inputs):
        images = [self.normalizer(x["image"].to(self.device)) for x in batched_inputs]
        images = ImageList.from_tensors(images)
        return images

    def inference(self, box_cls, box_pred, image_sizes, wo_sigmoid=False):
        """
        Arguments:
            box_cls (Tensor): tensor of shape (batch_size, num_queries, K).
                The tensor predicts the classification probability for each query.
            box_pred (Tensor): tensors of shape (batch_size, num_queries, 4).
                The tensor predicts 4-vector (x,y,w,h) box
                regression values for every queryx
            image_sizes (List[torch.Size]): the input image sizes

        Returns:
            results (List[Instances]): a list of #images elements.
        """
        assert len(box_cls) == len(image_sizes)
        results = []

        # box_cls.shape: 1, 300, 80
        # box_pred.shape: 1, 300, 4
        if wo_sigmoid:
            prob = box_cls
        else:
            prob = box_cls.sigmoid()
        scores, topk_boxes, labels = select_query_class_topk(
            prob,
            max_detections=self.select_box_nums_for_evaluation,
            per_query_class_topk=self.inference_query_class_topk,
        )

        boxes = torch.gather(box_pred, 1, topk_boxes.unsqueeze(-1).repeat(1, 1, 4))

        # For each box we assign the best class or the second best if the best on is `no_object`.
        # scores, labels = F.softmax(box_cls, dim=-1)[:, :, :-1].max(-1)

        for i, (scores_per_image, labels_per_image, box_pred_per_image, image_size) in enumerate(
            zip(scores, labels, boxes, image_sizes)
        ):
            result = Instances(image_size)
            result.pred_boxes = Boxes(box_cxcywh_to_xyxy(box_pred_per_image))

            result.pred_boxes.scale(scale_x=image_size[1], scale_y=image_size[0])
            result.scores = scores_per_image
            result.pred_classes = labels_per_image
            results.append(result)
        return results

    def prepare_targets(self, targets):
        new_targets = []
        for targets_per_image in targets:
            h, w = targets_per_image.image_size
            image_size_xyxy = torch.as_tensor([w, h, w, h], dtype=torch.float, device=self.device)
            gt_classes = targets_per_image.gt_classes
            gt_scores = targets_per_image.gt_scores
            gt_boxes = targets_per_image.gt_boxes.tensor / image_size_xyxy
            gt_boxes = box_xyxy_to_cxcywh(gt_boxes)
            new_targets.append({"labels": gt_classes, "boxes": gt_boxes, "scores": gt_scores})
        return new_targets
