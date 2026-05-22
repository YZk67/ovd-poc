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
from typing import List, Optional, Tuple
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

logger_rpsa = setup_logger()  # 用于RPSA日志输出


class RegionDescriptionVerifier(nn.Module):
    """Small MLP verifier used for detector-internal region/description fusion."""

    def __init__(self, input_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        second_hidden = max(64, hidden_dim // 2)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, second_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(second_hidden, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features).squeeze(-1)


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
        device="cuda",
        dn_number: int = 100,
        label_noise_ratio: float = 0.2,
        box_noise_scale: float = 1.0,
        dn_label_embed_source: str = "query",
        dn_multi_prototype_sampling: str = "mean",
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
        save_roi_features_only: bool = False,
        save_roi_features_fp16: bool = False,
        vlm_temperature: float =100.0,
        alpha: float =0.3,
        beta: float =0.7,
        novel_scale: float =5.0,
        clip_head_path=None,
        use_soft_attention: bool = True,
        soft_attention_tau: float = 0.1,
        region_verifier_enabled: bool = False,
        region_verifier_checkpoint: Optional[str] = None,
        region_verifier_text_path: Optional[str] = None,
        region_verifier_text_index: Optional[int] = None,
        region_verifier_fusion: str = "logit_add",
        region_verifier_fusion_weight: float = 0.25,
        region_verifier_topk_per_image: int = 20,
        region_verifier_candidate_mode: str = "flat",
        region_verifier_num_boxes_per_image: int = 300,
        region_verifier_num_phrases_per_box: int = 50,
        region_verifier_eval_chunk_size: int = 4096,
        region_verifier_candidate_only: bool = False,
        region_verifier_train_enabled: bool = False,
        region_verifier_train_feature_mode: str = "no_detector_score",
        region_verifier_train_hidden_dim: int = 512,
        region_verifier_train_dropout: float = 0.1,
        region_verifier_same_phrase_neg_per_pos: int = 1,
        region_verifier_wrong_phrase_neg_per_pos: int = 2,
        region_verifier_neg_iou_thresh: float = 0.3,
        region_verifier_max_pairs: int = 256,
        region_verifier_train_detach_region_features: bool = True,
        region_verifier_train_loss_type: str = "bce",
        region_verifier_ranking_margin: float = 0.0,
    ):
        super().__init__()
        self.vlm_temperature = vlm_temperature
        self.alpha = alpha
        self.beta = beta
        self.novel_scale = novel_scale
        self.use_soft_attention = use_soft_attention
        self.soft_attention_tau = soft_attention_tau
        self.region_verifier_enabled = bool(region_verifier_enabled)
        self.region_verifier_checkpoint = region_verifier_checkpoint
        self.region_verifier_text_path = region_verifier_text_path
        self.region_verifier_text_index = region_verifier_text_index
        self.region_verifier_fusion = region_verifier_fusion
        self.region_verifier_fusion_weight = region_verifier_fusion_weight
        self.region_verifier_topk_per_image = region_verifier_topk_per_image
        if region_verifier_candidate_mode not in {"flat", "box_phrase"}:
            raise ValueError(
                "region_verifier_candidate_mode must be 'flat' or 'box_phrase', "
                f"got {region_verifier_candidate_mode!r}."
            )
        self.region_verifier_candidate_mode = str(region_verifier_candidate_mode)
        self.region_verifier_num_boxes_per_image = int(region_verifier_num_boxes_per_image)
        self.region_verifier_num_phrases_per_box = int(region_verifier_num_phrases_per_box)
        self.region_verifier_eval_chunk_size = int(region_verifier_eval_chunk_size)
        self.region_verifier_candidate_only = bool(region_verifier_candidate_only)
        self.region_verifier_train_enabled = bool(region_verifier_train_enabled)
        self.region_verifier_train_feature_mode = region_verifier_train_feature_mode
        self.region_verifier_train_hidden_dim = int(region_verifier_train_hidden_dim)
        self.region_verifier_train_dropout = float(region_verifier_train_dropout)
        self.region_verifier_same_phrase_neg_per_pos = int(region_verifier_same_phrase_neg_per_pos)
        self.region_verifier_wrong_phrase_neg_per_pos = int(region_verifier_wrong_phrase_neg_per_pos)
        self.region_verifier_neg_iou_thresh = float(region_verifier_neg_iou_thresh)
        self.region_verifier_max_pairs = int(region_verifier_max_pairs)
        self.region_verifier_train_detach_region_features = bool(region_verifier_train_detach_region_features)
        self.region_verifier_train_loss_type = str(region_verifier_train_loss_type)
        self.region_verifier_ranking_margin = float(region_verifier_ranking_margin)
        self.region_verifier = None
        self.region_verifier_feature_mode = str(region_verifier_train_feature_mode)
        self.region_verifier_text_embedding = None
        if dn_label_embed_source not in {"query", "classifier"}:
            raise ValueError(
                "dn_label_embed_source must be 'query' or 'classifier', "
                f"got {dn_label_embed_source!r}."
            )
        if dn_multi_prototype_sampling not in {"mean", "random"}:
            raise ValueError(
                "dn_multi_prototype_sampling must be 'mean' or 'random', "
                f"got {dn_multi_prototype_sampling!r}."
            )
        self.dn_label_embed_source = str(dn_label_embed_source)
        self.dn_multi_prototype_sampling = str(dn_multi_prototype_sampling)
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
        if self.score_ensemble or self.region_verifier_enabled or self.region_verifier_train_enabled:
            clip_head = torch.load(clip_head_path)
            self.identical, self.thead = clip_head[0]
            self.head = clip_head[1]

        if self.region_verifier_enabled or self.region_verifier_train_enabled:
            if self.region_verifier_checkpoint:
                self.region_verifier, self.region_verifier_feature_mode = self._load_region_verifier(
                    self.region_verifier_checkpoint,
                    device=self.device,
                    freeze=not self.region_verifier_train_enabled,
                )
            elif self.region_verifier_train_enabled:
                if self.region_verifier_feature_mode not in {"no_detector_score", "full", "no_text"}:
                    raise ValueError(
                        f"Unsupported region verifier train feature mode: {self.region_verifier_feature_mode!r}."
                    )
                input_dim = self._region_verifier_input_dim(self.region_verifier_feature_mode, feat_dim)
                self.region_verifier = RegionDescriptionVerifier(
                    input_dim=input_dim,
                    hidden_dim=self.region_verifier_train_hidden_dim,
                    dropout=self.region_verifier_train_dropout,
                ).to(self.device)
            else:
                raise ValueError("region_verifier_enabled=True requires region_verifier_checkpoint.")
            if self.region_verifier_feature_mode not in {"no_detector_score", "full", "no_text"}:
                raise ValueError(
                    f"Unsupported region verifier feature_mode={self.region_verifier_feature_mode!r}."
                )
            if self.region_verifier_feature_mode == "full":
                logger_rpsa.warning(
                    "[RegionVerifier] Loaded a full verifier that uses detector score. "
                    "Use a no_detector_score checkpoint for the cleaner internal method."
                )

        if self.score_ensemble:
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
        self.save_dir = save_dir
        self.save_roi_features_only = bool(save_roi_features_only)
        self.save_roi_features_fp16 = bool(save_roi_features_fp16)
        if self.save_dir:
            os.makedirs(self.save_dir, exist_ok=True)

        if self.region_verifier_enabled or self.region_verifier_train_enabled:
            verifier_text_path = self.region_verifier_text_path or eval_query_path
            verifier_text = self._load_region_verifier_text_embedding(
                verifier_text_path,
                text_index=self.region_verifier_text_index,
                device=self.device,
            )
            if verifier_text.shape[0] != self.num_classes:
                raise ValueError(
                    "Region verifier text embedding class count must match num_classes, "
                    f"got {verifier_text.shape[0]} vs {self.num_classes}."
                )
            self.region_verifier_text_embedding = F.normalize(verifier_text, p=2, dim=-1)
    
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

    @staticmethod
    def _select_dn_label_embeddings(
        content_query_embeds: torch.Tensor,
        labels: torch.Tensor,
        sampling: str = "mean",
    ) -> torch.Tensor:
        if content_query_embeds.ndim == 2:
            return content_query_embeds[labels]
        if content_query_embeds.ndim != 3:
            raise ValueError(
                "DN label embeddings must be 2D [C, D] or 3D [C, K, D], "
                f"got shape {tuple(content_query_embeds.shape)}."
            )
        if sampling == "mean":
            return content_query_embeds.mean(dim=1)[labels]
        if sampling == "random":
            candidates = content_query_embeds[labels]
            alias_idx = torch.randint(
                candidates.shape[1],
                (labels.numel(),),
                device=labels.device,
            )
            row_idx = torch.arange(labels.numel(), device=labels.device)
            return candidates[row_idx, alias_idx]
        raise ValueError(f"Unsupported dn_multi_prototype_sampling={sampling!r}.")

    def _build_dn_label_query_embeds(self, raw_content_query_embeds, content_inds=None):
        if self.dn_label_embed_source == "query":
            return raw_content_query_embeds

        classifier = self.transformer.decoder.class_embed[0]
        if not getattr(classifier, "static_multi_prototype", False):
            raise ValueError(
                "dn_label_embed_source='classifier' requires a static multi-prototype "
                "classifier weight bank."
            )

        embeddings = classifier.zs_weight
        if content_inds is not None:
            embeddings = embeddings[content_inds]
        embeddings = embeddings.to(
            device=self.content_layer.weight.device,
            dtype=self.content_layer.weight.dtype,
        )
        if embeddings.ndim != 3:
            raise ValueError(
                "Classifier DN label embeddings must be 3D [C, K, D], "
                f"got shape {tuple(embeddings.shape)}."
            )
        if embeddings.shape[-1] != self.content_layer.in_features:
            raise ValueError(
                "Classifier DN label embedding dim must match content_layer input dim, "
                f"got {embeddings.shape[-1]} vs {self.content_layer.in_features}."
            )
        projected = self.content_layer(embeddings.reshape(-1, embeddings.shape[-1])).reshape(
            embeddings.shape[0],
            embeddings.shape[1],
            -1,
        )
        return F.normalize(projected, p=2, dim=-1)

    @staticmethod
    def _torch_load_checkpoint(path: str, device: str):
        try:
            return torch.load(path, map_location=device, weights_only=False)
        except TypeError:
            return torch.load(path, map_location=device)

    @staticmethod
    def _region_verifier_input_dim(feature_mode: str, feature_dim: int) -> int:
        if feature_mode == "full":
            return feature_dim * 4 + 1
        if feature_mode == "no_text":
            return feature_dim + 1
        if feature_mode == "no_detector_score":
            return feature_dim * 4
        raise ValueError(f"Unsupported region verifier feature mode: {feature_mode}")

    def _load_region_verifier(
        self,
        path: str,
        device: str,
        *,
        freeze: bool = True,
    ) -> Tuple[RegionDescriptionVerifier, str]:
        checkpoint = self._torch_load_checkpoint(path, device)
        input_dim = int(checkpoint["input_dim"])
        hidden_dim = int(checkpoint.get("hidden_dim", 512))
        dropout = float(checkpoint.get("dropout", 0.0))
        feature_mode = str(checkpoint.get("feature_mode", "full"))
        verifier = RegionDescriptionVerifier(input_dim=input_dim, hidden_dim=hidden_dim, dropout=dropout)
        verifier.load_state_dict(checkpoint["model_state"])
        verifier.to(device)
        verifier.eval()
        if freeze:
            for param in verifier.parameters():
                param.requires_grad_(False)
        else:
            verifier.train()
        logger_rpsa.info(
            "[RegionVerifier] loaded %s (epoch=%s, score=%s, feature_mode=%s)",
            path,
            checkpoint.get("epoch"),
            checkpoint.get("score"),
            feature_mode,
        )
        return verifier, feature_mode

    @staticmethod
    def _load_region_verifier_text_embedding(
        path: str,
        *,
        text_index: Optional[int],
        device: str,
    ) -> torch.Tensor:
        arr = np.load(path)
        tensor = torch.tensor(arr, dtype=torch.float32, device=device).contiguous()
        if tensor.ndim == 3:
            if text_index is None:
                tensor = tensor.mean(dim=1)
            else:
                tensor = tensor[:, text_index, :]
        if tensor.ndim != 2:
            raise ValueError(
                f"Expected region verifier text embedding with shape [C,D] or [C,K,D], got {tuple(tensor.shape)}."
            )
        return tensor

    @staticmethod
    def _logit_tensor(score: torch.Tensor) -> torch.Tensor:
        score = score.clamp(min=1e-6, max=1.0 - 1e-6)
        return torch.log(score / (1.0 - score))

    def _build_region_verifier_features(
        self,
        region_feats: torch.Tensor,
        text_feats: torch.Tensor,
        detector_scores: torch.Tensor,
    ) -> torch.Tensor:
        if self.region_verifier_feature_mode == "full":
            score = detector_scores.to(dtype=region_feats.dtype).view(-1, 1)
            return torch.cat(
                [
                    region_feats,
                    text_feats,
                    region_feats * text_feats,
                    torch.abs(region_feats - text_feats),
                    score,
                ],
                dim=-1,
            )
        if self.region_verifier_feature_mode == "no_text":
            score = detector_scores.to(dtype=region_feats.dtype).view(-1, 1)
            return torch.cat([region_feats, score], dim=-1)
        if self.region_verifier_feature_mode == "no_detector_score":
            return torch.cat(
                [
                    region_feats,
                    text_feats,
                    region_feats * text_feats,
                    torch.abs(region_feats - text_feats),
                ],
                dim=-1,
            )
        raise ValueError(f"Unsupported region verifier feature mode: {self.region_verifier_feature_mode}")

    @staticmethod
    def _pairwise_iou_cxcywh(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
        if boxes1.numel() == 0 or boxes2.numel() == 0:
            return boxes1.new_zeros((boxes1.shape[0], boxes2.shape[0]))
        boxes1_xyxy = box_cxcywh_to_xyxy(boxes1)
        boxes2_xyxy = box_cxcywh_to_xyxy(boxes2)

        lt = torch.maximum(boxes1_xyxy[:, None, :2], boxes2_xyxy[None, :, :2])
        rb = torch.minimum(boxes1_xyxy[:, None, 2:], boxes2_xyxy[None, :, 2:])
        wh = (rb - lt).clamp(min=0)
        inter = wh[:, :, 0] * wh[:, :, 1]

        area1 = (boxes1_xyxy[:, 2] - boxes1_xyxy[:, 0]).clamp(min=0) * (
            boxes1_xyxy[:, 3] - boxes1_xyxy[:, 1]
        ).clamp(min=0)
        area2 = (boxes2_xyxy[:, 2] - boxes2_xyxy[:, 0]).clamp(min=0) * (
            boxes2_xyxy[:, 3] - boxes2_xyxy[:, 1]
        ).clamp(min=0)
        union = area1[:, None] + area2[None, :] - inter
        return torch.where(union > 0, inter / union.clamp(min=1e-6), torch.zeros_like(inter))

    def _sample_wrong_phrase_labels(
        self,
        *,
        correct_label: torch.Tensor,
        present_labels: torch.Tensor,
        num_classes: int,
        count: int,
    ) -> List[torch.Tensor]:
        if count <= 0 or num_classes <= 1:
            return []

        wrong_labels: List[torch.Tensor] = []
        alternatives = present_labels[present_labels != correct_label]
        for idx in range(count):
            if alternatives.numel() > 0:
                wrong_labels.append(alternatives[idx % alternatives.numel()])
            else:
                sampled = torch.randint(
                    low=0,
                    high=num_classes - 1,
                    size=(),
                    device=correct_label.device,
                    dtype=correct_label.dtype,
                )
                wrong_labels.append(sampled + (sampled >= correct_label).to(dtype=sampled.dtype))
        return wrong_labels

    def _append_region_verifier_pair(
        self,
        *,
        region_feats: List[torch.Tensor],
        text_feats: List[torch.Tensor],
        detector_scores: List[torch.Tensor],
        labels: List[torch.Tensor],
        group_ids: List[int],
        region_feat: torch.Tensor,
        text_feat: torch.Tensor,
        detector_score: torch.Tensor,
        label: float,
        group_id: int,
    ) -> None:
        region_feats.append(region_feat)
        text_feats.append(text_feat)
        detector_scores.append(detector_score.reshape(()))
        labels.append(region_feat.new_tensor(label))
        group_ids.append(group_id)

    def compute_region_verifier_training_loss(
        self,
        output,
        targets,
        features_wonorm,
        content_inds=None,
    ) -> torch.Tensor:
        if not self.region_verifier_train_enabled:
            return output["pred_logits"].sum() * 0.0
        if self.region_verifier is None or self.region_verifier_text_embedding is None:
            raise RuntimeError("Region verifier training is enabled but verifier/text embedding is not initialized.")

        pred_logits = output["pred_logits"]
        pred_boxes = output["pred_boxes"]
        device = pred_logits.device
        num_classes = pred_logits.shape[-1]

        text_embedding = self.region_verifier_text_embedding
        if content_inds is not None:
            text_embedding = text_embedding[content_inds]
        text_embedding = text_embedding.to(device=device, dtype=pred_logits.dtype)

        with torch.no_grad():
            match_output = {"pred_logits": pred_logits.detach(), "pred_boxes": pred_boxes.detach()}
            indices = self.criterion.matcher(match_output, targets)

        roi_features = self.extract_region_feature(features_wonorm, pred_boxes, "p3")
        if self.region_verifier_train_detach_region_features:
            roi_features = roi_features.detach()

        region_feats: List[torch.Tensor] = []
        text_feats: List[torch.Tensor] = []
        detector_scores: List[torch.Tensor] = []
        labels: List[torch.Tensor] = []
        group_ids: List[int] = []
        next_group_id = 0

        detached_boxes = pred_boxes.detach()
        detached_scores = pred_logits.detach().sigmoid()
        same_neg_per_pos = max(0, self.region_verifier_same_phrase_neg_per_pos)
        wrong_neg_per_pos = max(0, self.region_verifier_wrong_phrase_neg_per_pos)

        for batch_idx, (src_idx, tgt_idx) in enumerate(indices):
            src_idx = src_idx.to(device=device)
            tgt_idx = tgt_idx.to(device=device)
            if src_idx.numel() == 0:
                continue

            target = targets[batch_idx]
            target_labels = target["labels"].to(device=device, dtype=torch.long)
            target_boxes = target["boxes"].to(device=device, dtype=detached_boxes.dtype)
            present_labels = torch.unique(target_labels)
            matched_query_mask = torch.zeros(pred_logits.shape[1], dtype=torch.bool, device=device)
            matched_query_mask[src_idx] = True

            for query_idx, target_idx in zip(src_idx.tolist(), tgt_idx.tolist()):
                label = target_labels[target_idx]
                label_int = int(label.item())
                if label_int < 0 or label_int >= num_classes:
                    continue

                group_id = next_group_id
                next_group_id += 1
                region_feat = roi_features[batch_idx, query_idx]
                text_feat = text_embedding[label_int]
                detector_score = detached_scores[batch_idx, query_idx, label_int]
                self._append_region_verifier_pair(
                    region_feats=region_feats,
                    text_feats=text_feats,
                    detector_scores=detector_scores,
                    labels=labels,
                    group_ids=group_ids,
                    region_feat=region_feat,
                    text_feat=text_feat,
                    detector_score=detector_score,
                    label=1.0,
                    group_id=group_id,
                )

                for wrong_label in self._sample_wrong_phrase_labels(
                    correct_label=label,
                    present_labels=present_labels,
                    num_classes=num_classes,
                    count=wrong_neg_per_pos,
                ):
                    wrong_label = wrong_label.to(device=device, dtype=torch.long)
                    wrong_label_int = int(wrong_label.item())
                    self._append_region_verifier_pair(
                        region_feats=region_feats,
                        text_feats=text_feats,
                        detector_scores=detector_scores,
                        labels=labels,
                        group_ids=group_ids,
                        region_feat=region_feat,
                        text_feat=text_embedding[wrong_label_int],
                        detector_score=detached_scores[batch_idx, query_idx, wrong_label_int],
                        label=0.0,
                        group_id=group_id,
                    )

                if same_neg_per_pos <= 0:
                    continue

                ious = self._pairwise_iou_cxcywh(
                    detached_boxes[batch_idx],
                    target_boxes[target_idx : target_idx + 1],
                ).squeeze(1)
                valid = ious <= self.region_verifier_neg_iou_thresh
                valid &= ~matched_query_mask
                if not valid.any():
                    continue

                same_scores = detached_scores[batch_idx, :, label_int].clone()
                same_scores[~valid] = -1.0
                neg_count = min(same_neg_per_pos, int(valid.sum().item()))
                neg_queries = torch.topk(same_scores, k=neg_count).indices
                for neg_query_idx in neg_queries.tolist():
                    self._append_region_verifier_pair(
                        region_feats=region_feats,
                        text_feats=text_feats,
                        detector_scores=detector_scores,
                        labels=labels,
                        group_ids=group_ids,
                        region_feat=roi_features[batch_idx, neg_query_idx],
                        text_feat=text_feat,
                        detector_score=detached_scores[batch_idx, neg_query_idx, label_int],
                        label=0.0,
                        group_id=group_id,
                    )

        if not labels:
            return pred_logits.sum() * 0.0

        region_tensor = torch.stack(region_feats, dim=0)
        text_tensor = torch.stack(text_feats, dim=0).to(dtype=region_tensor.dtype)
        score_tensor = torch.stack(detector_scores, dim=0).to(dtype=region_tensor.dtype)
        label_tensor = torch.stack(labels, dim=0).to(dtype=region_tensor.dtype)
        group_tensor = torch.as_tensor(group_ids, device=device, dtype=torch.long)

        max_pairs = self.region_verifier_max_pairs
        if max_pairs > 0 and label_tensor.numel() > max_pairs:
            keep = torch.randperm(label_tensor.numel(), device=device)[:max_pairs]
            region_tensor = region_tensor[keep]
            text_tensor = text_tensor[keep]
            score_tensor = score_tensor[keep]
            label_tensor = label_tensor[keep]
            group_tensor = group_tensor[keep]

        pair_features = self._build_region_verifier_features(region_tensor, text_tensor, score_tensor)
        logits = self.region_verifier(pair_features)

        loss_type = self.region_verifier_train_loss_type
        if loss_type == "bce":
            loss = F.binary_cross_entropy_with_logits(logits, label_tensor)
        elif loss_type in {"pairwise_rank", "ranking"}:
            rank_losses = []
            rank_pair_count = 0
            for group_id in torch.unique(group_tensor).tolist():
                in_group = group_tensor == int(group_id)
                pos_logits = logits[in_group & (label_tensor > 0.5)]
                neg_logits = logits[in_group & (label_tensor <= 0.5)]
                if pos_logits.numel() == 0 or neg_logits.numel() == 0:
                    continue
                logit_margin = pos_logits[:, None] - neg_logits[None, :]
                rank_losses.append(F.softplus(self.region_verifier_ranking_margin - logit_margin).mean())
                rank_pair_count += int(pos_logits.numel() * neg_logits.numel())
            if not rank_losses:
                loss = logits.sum() * 0.0
            else:
                loss = torch.stack(rank_losses).mean()
        else:
            raise ValueError(
                f"Unsupported region_verifier_train_loss_type={loss_type!r}; "
                "expected 'bce' or 'pairwise_rank'."
            )

        try:
            storage = get_event_storage()
            storage.put_scalar("region_verifier_num_pairs", float(label_tensor.numel()), smoothing_hint=False)
            storage.put_scalar("region_verifier_pos_rate", float(label_tensor.mean().detach()), smoothing_hint=False)
            if loss_type in {"pairwise_rank", "ranking"}:
                num_pos = int((label_tensor > 0.5).sum().item())
                num_neg = int((label_tensor <= 0.5).sum().item())
                storage.put_scalar("region_verifier_num_pos", float(num_pos), smoothing_hint=False)
                storage.put_scalar("region_verifier_num_neg", float(num_neg), smoothing_hint=False)
                storage.put_scalar("region_verifier_num_rank_pairs", float(rank_pair_count), smoothing_hint=False)
        except AssertionError:
            pass

        return loss

    def _fuse_region_verifier_scores(
        self,
        detector_scores: torch.Tensor,
        verifier_logits: torch.Tensor,
    ) -> torch.Tensor:
        if self.region_verifier_fusion == "logit_add":
            return torch.sigmoid(
                self._logit_tensor(detector_scores)
                + self.region_verifier_fusion_weight * verifier_logits
            )
        if self.region_verifier_fusion == "linear":
            verifier_prob = torch.sigmoid(verifier_logits)
            return (
                (1.0 - self.region_verifier_fusion_weight) * detector_scores
                + self.region_verifier_fusion_weight * verifier_prob
            )
        if self.region_verifier_fusion == "replace":
            return torch.sigmoid(verifier_logits)
        raise ValueError(f"Unsupported region verifier fusion: {self.region_verifier_fusion}")

    @torch.no_grad()
    def _apply_region_verifier_pairs(
        self,
        fused: torch.Tensor,
        cls_score: torch.Tensor,
        roi_features: torch.Tensor,
        text_embedding: torch.Tensor,
        batch_idx: int,
        query_indexes: torch.Tensor,
        class_indexes: torch.Tensor,
    ) -> None:
        if query_indexes.numel() == 0:
            return

        detector_scores = cls_score[batch_idx, query_indexes, class_indexes]
        chunk_size = int(self.region_verifier_eval_chunk_size)
        if chunk_size <= 0:
            chunk_size = int(query_indexes.numel())

        fused_chunks = []
        for start in range(0, int(query_indexes.numel()), chunk_size):
            end = min(start + chunk_size, int(query_indexes.numel()))
            chunk_query_indexes = query_indexes[start:end]
            chunk_class_indexes = class_indexes[start:end]
            region_feats = roi_features[batch_idx, chunk_query_indexes]
            text_feats = text_embedding[chunk_class_indexes]
            chunk_detector_scores = detector_scores[start:end]
            verifier_inputs = self._build_region_verifier_features(
                region_feats,
                text_feats,
                chunk_detector_scores,
            ).to(device=cls_score.device)
            verifier_logits = self.region_verifier(verifier_inputs).to(dtype=cls_score.dtype)
            fused_chunks.append(
                self._fuse_region_verifier_scores(chunk_detector_scores, verifier_logits)
            )

        fused_scores = torch.cat(fused_chunks, dim=0)
        fused[batch_idx, query_indexes, class_indexes] = fused_scores

    @torch.no_grad()
    def apply_region_verifier(self, cls_score: torch.Tensor, roi_features: torch.Tensor) -> torch.Tensor:
        if not self.region_verifier_enabled or self.region_verifier is None:
            return cls_score
        if self.region_verifier_text_embedding is None:
            raise RuntimeError("region_verifier_text_embedding is not initialized.")

        if self.region_verifier_candidate_only:
            fused = cls_score.new_zeros(cls_score.shape)
        else:
            fused = cls_score.clone()
        num_classes = cls_score.shape[-1]
        text_embedding = self.region_verifier_text_embedding.to(
            device=cls_score.device,
            dtype=roi_features.dtype,
        )
        text_embedding = F.normalize(text_embedding, p=2, dim=-1)

        if self.region_verifier_candidate_mode == "flat":
            topk = int(self.region_verifier_topk_per_image)
            if topk <= 0:
                topk = cls_score.shape[1] * cls_score.shape[2]
            topk = min(topk, cls_score.shape[1] * cls_score.shape[2])

            flat_scores = cls_score.view(cls_score.shape[0], -1)
            _, topk_indexes = torch.topk(flat_scores, topk, dim=1)
            for batch_idx in range(cls_score.shape[0]):
                pair_indexes = topk_indexes[batch_idx]
                query_indexes = torch.div(pair_indexes, num_classes, rounding_mode="floor")
                class_indexes = pair_indexes % num_classes
                self._apply_region_verifier_pairs(
                    fused,
                    cls_score,
                    roi_features,
                    text_embedding,
                    batch_idx,
                    query_indexes,
                    class_indexes,
                )
            return fused

        if self.region_verifier_candidate_mode == "box_phrase":
            num_boxes = int(self.region_verifier_num_boxes_per_image)
            if num_boxes <= 0:
                num_boxes = cls_score.shape[1]
            num_boxes = min(num_boxes, cls_score.shape[1])

            num_phrases = int(self.region_verifier_num_phrases_per_box)
            if num_phrases <= 0:
                num_phrases = num_classes
            num_phrases = min(num_phrases, num_classes)

            box_scores = cls_score.max(dim=-1).values
            _, top_box_indexes = torch.topk(box_scores, num_boxes, dim=1)
            for batch_idx in range(cls_score.shape[0]):
                query_indexes = top_box_indexes[batch_idx]
                phrase_scores = cls_score[batch_idx, query_indexes]
                _, class_indexes_2d = torch.topk(phrase_scores, num_phrases, dim=1)
                query_indexes = query_indexes[:, None].expand(-1, num_phrases).reshape(-1)
                class_indexes = class_indexes_2d.reshape(-1)
                self._apply_region_verifier_pairs(
                    fused,
                    cls_score,
                    roi_features,
                    text_embedding,
                    batch_idx,
                    query_indexes,
                    class_indexes,
                )
            return fused

        raise ValueError(f"Unsupported region_verifier_candidate_mode: {self.region_verifier_candidate_mode}")

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
        if self.training:
            batch_size, _, H, W = images.tensor.shape
            img_masks = images.tensor.new_ones(batch_size, H, W)
            for img_id in range(batch_size):
                img_h, img_w = batched_inputs[img_id]["instances"].image_size
                img_masks[img_id, :img_h, :img_w] = 0
            if self.use_fed_loss:
                content_inds, batched_inputs = self.filter_content_info(batched_inputs)
        else:
            batch_size, _, H, W = images.tensor.shape
            img_masks = images.tensor.new_zeros(batch_size, H, W)

        # original features
        needs_wonorm_features = self.score_ensemble or self.region_verifier_enabled or self.region_verifier_train_enabled
        backbone_outputs = self.backbone(images.tensor)
        if needs_wonorm_features:
            if not (isinstance(backbone_outputs, tuple) and len(backbone_outputs) == 2):
                raise RuntimeError(
                    "score_ensemble or region_verifier requires backbone.score_ensemble=True "
                    "so ConvNeXt returns both normalized and pre-normalized features."
                )
            features, features_wonorm = backbone_outputs  # output feature dict
        else:
            features = backbone_outputs  # output feature dict

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
            # [C,K,D_text]
            with_loss = self.training
            shared_prototypes, shared_apr_loss = text_classifier.tpa(text_feats, with_loss=with_loss)
            # Broadcast to every class_embed copy so they reuse the same prototype tensor.
            for ce in self.transformer.decoder.class_embed:
                ce.set_external_prototypes(shared_prototypes, shared_apr_loss)

            # Project to decoder dim for query init
            proto_ckd = self.content_layer(shared_prototypes.view(-1, shared_prototypes.size(-1))).view(
                shared_prototypes.size(0), shared_prototypes.size(1), -1
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
            dn_label_query_embeds = self._build_dn_label_query_embeds(
                raw_content_query_embeds,
                content_inds=content_inds,
            )
            if dn_label_query_embeds.ndim == 3 and (
                self.dn_label_embed_source != "query" or dn_label_query_embeds.shape[1] > 1
            ):
                storage = get_event_storage()
                storage.put_scalar(
                    "dn_num_aliases",
                    float(dn_label_query_embeds.shape[1]),
                    smoothing_hint=False,
                )
                storage.put_scalar(
                    "dn_alias_random_sampling",
                    float(self.dn_multi_prototype_sampling == "random"),
                    smoothing_hint=False,
                )
            input_query_label, input_query_bbox, attn_mask, dn_meta = self.prepare_for_cdn(
                targets,
                dn_number=self.dn_number,
                label_noise_ratio=self.label_noise_ratio,
                box_noise_scale=self.box_noise_scale,
                num_queries=self.num_queries,
                num_classes=cdn_num_classes,
                hidden_dim=self.embed_dim,
                # label_enc=self.label_enc,
                content_query_embeds=dn_label_query_embeds,
                multi_prototype_sampling=self.dn_multi_prototype_sampling,
            )
        else:
            input_query_label, input_query_bbox, attn_mask, dn_meta = None, None, None, None
        query_embeds = (input_query_label, input_query_bbox)

        # Set soft-attention parameters for transformer if using multi-prototype mode
        if hasattr(self, 'use_soft_attention') and self.use_soft_attention and raw_content_query_embeds.ndim == 3:
            self.transformer.use_soft_attention = self.use_soft_attention
            self.transformer.soft_attention_tau = self.soft_attention_tau

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
            if self.region_verifier_train_enabled:
                loss_dict["loss_region_verifier"] = self.compute_region_verifier_training_loss(
                    output,
                    targets,
                    features_wonorm,
                    content_inds=content_inds,
                )
            # === 1️⃣ 添加 APR 损失（保持原逻辑） ===
            if apr_loss is not None:
                loss_dict["loss_apr"] = apr_loss

            # === 2️⃣ 添加 RPSA 损失（Region–Prototype Semantic Alignment） ===
            # RPSA 模块的损失在 transformer 内部计算，并暂存在 encoder分类器中
            try:
                dec_encoder = self.transformer.decoder.class_embed[self.transformer.decoder.num_layers]
                if hasattr(dec_encoder, "rpsa_loss"):
                    rpsa_loss_value = dec_encoder.rpsa_loss
                    if rpsa_loss_value is not None:
                        loss_dict["loss_rpsa"] = rpsa_loss_value
                        stats = getattr(dec_encoder, "rpsa_stats", None)
                        if isinstance(stats, dict):
                            if "rpsa_center_orth_mse" in stats:
                                loss_dict["rpsa_center_orth_mse"] = stats["rpsa_center_orth_mse"]
                            if "rpsa_pi_entropy" in stats:
                                loss_dict["rpsa_pi_entropy"] = stats["rpsa_pi_entropy"]

                        storage = None
                        try:
                            storage = get_event_storage()
                            current_iter = storage.iter
                        except AssertionError:
                            current_iter = 0

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
                            if isinstance(stats, dict):
                                if "rpsa_bg_ratio" in stats:
                                    storage.put_scalar("loss_rpsa_bg_ratio", float(stats["rpsa_bg_ratio"]), smoothing_hint=False)
                                if "rpsa_valid_clusters" in stats:
                                    storage.put_scalar("loss_rpsa_valid_clusters", float(stats["rpsa_valid_clusters"]), smoothing_hint=False)
                                if "rpsa_tokens" in stats:
                                    storage.put_scalar("loss_rpsa_tokens", float(stats["rpsa_tokens"]), smoothing_hint=False)
                    # else:
                    #     logger_rpsa.warning("[RPSA] ⚠️ rpsa_loss is None - RPSA may not be computing loss")
                # else:
                #     logger_rpsa.warning(f"[RPSA] ⚠️ dec_encoder has no rpsa_loss attribute")
            except Exception as e:
                logger_rpsa.warning(f"[RPSA] ❌ Loss aggregation skipped: {e}")
                import traceback
                logger_rpsa.debug(f"[RPSA] Traceback: {traceback.format_exc()}")

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
            roi_features_ori = None
            if self.score_ensemble or self.region_verifier_enabled:
                roi_features_ori = self.extract_region_feature(features_wonorm, box_pred, 'p3')

            if self.score_ensemble:
                if self.save_dir:
                    save_output = {}
                    if not self.save_roi_features_only:
                        save_output["pred_logits"] = output["pred_logits"].detach().cpu()
                    roi_features_to_save = roi_features_ori.detach().cpu()
                    if self.save_roi_features_fp16:
                        roi_features_to_save = roi_features_to_save.half()
                    save_output["roi_features_ori"] = roi_features_to_save# [1, 900, 768]
                    save_output["pred_boxes"] = output["pred_boxes"].detach().cpu()
                    torch.save(save_output, os.path.join(self.save_dir, filename))

                cls_score = box_cls.sigmoid()
                vlm_score = roi_features_ori @ self.vlm_content_query_embedding.t() * self.vlm_temperature
                vlm_score = vlm_score.softmax(dim=-1)
                cls_score[:, :, self.base_idx] = cls_score[:, :, self.base_idx] ** (
                        1 - self.alpha) * vlm_score[:, :, self.base_idx] ** self.alpha
                cls_score[:, :, self.novel_idx] = cls_score[:, :, self.novel_idx] ** (
                        1 - self.beta) * vlm_score[:, :, self.novel_idx] ** self.beta 
                cls_score[:, :, self.novel_idx] = cls_score[:, :, self.novel_idx] * self.novel_scale
                if self.region_verifier_enabled:
                    cls_score = self.apply_region_verifier(cls_score, roi_features_ori)
                box_cls = cls_score
                results = self.inference(box_cls, box_pred, images.image_sizes, wo_sigmoid=True)
            else:
                if self.region_verifier_enabled:
                    cls_score = box_cls.sigmoid()
                    cls_score = self.apply_region_verifier(cls_score, roi_features_ori)
                    results = self.inference(cls_score, box_pred, images.image_sizes, wo_sigmoid=True)
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
        multi_prototype_sampling="mean",
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
            input_label_content = self._select_dn_label_embeddings(
                content_query_embeds,
                m,
                sampling=multi_prototype_sampling,
            )
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
        topk_values, topk_indexes = torch.topk(
            prob.view(box_cls.shape[0], -1), self.select_box_nums_for_evaluation, dim=1
        )
        scores = topk_values
        topk_boxes = torch.div(topk_indexes, box_cls.shape[2], rounding_mode="floor")
        labels = topk_indexes % box_cls.shape[2]

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
