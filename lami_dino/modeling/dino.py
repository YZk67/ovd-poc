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

import logging
import os
import copy
import math
import json
from typing import List
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from detrex.layers import MLP, box_cxcywh_to_xyxy, box_xyxy_to_cxcywh
from detrex.utils import (inverse_sigmoid, is_dist_avail_and_initialized,
                          load_class_freq, get_fed_loss_inds, get_cluster_fed_loss_inds)

from detectron2.modeling import detector_postprocess
from detectron2.structures import Boxes, ImageList, Instances

from .unknown_proposals import select_unknown_proposals

logger = logging.getLogger(__name__)
d2_logger = logging.getLogger("detectron2")


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
        save_predictions: bool = False,
        unknown_iou_thr: float = 0.5,
        unknown_iou_thr_start: float = 0.3,
        unknown_warmup_iters: int = 2000,
        unknown_match_mode: str = "soft",
        unknown_match_cost: float = 2.0,
        pseudo_score_thr: float = 0.95,
        vlm_temperature: float =100.0,
        alpha: float =0.3,
        beta: float =0.7,
        novel_scale: float =5.0,
        clip_head_path=None,
        freeze_backbone: bool = False,
        freeze_classifier: bool = False,
        text_query_path=None,
    ):
        super().__init__()
        self.vlm_temperature = vlm_temperature
        self.alpha = alpha
        self.beta = beta
        self.novel_scale = novel_scale
        self.unknown_iou_thr = float(unknown_iou_thr)
        self.unknown_iou_thr_start = float(unknown_iou_thr_start)
        self.unknown_warmup_iters = int(unknown_warmup_iters)
        self.unknown_match_mode = str(unknown_match_mode)
        self.unknown_match_cost = float(unknown_match_cost)
        self.pseudo_score_thr = float(pseudo_score_thr)
        self._train_forward_count = 0
        # define backbone and position embedding module
        self.backbone = backbone
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad_(False)
        self.freeze_classifier = freeze_classifier
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

        # two-stage
        self.transformer.decoder.class_embed = self.class_embed
        self.transformer.decoder.bbox_embed = self.bbox_embed

        # hack implementation for two-stage
        for bbox_embed_layer in self.bbox_embed:
            nn.init.constant_(bbox_embed_layer.layers[-1].bias.data[2:], 0.0)

        # Freeze classifier (ZeroShotClassifier.linear) to prevent novel class forgetting.
        # The linear layer projects decoder queries into CLIP space; if it drifts toward
        # base classes during training, novel class matching degrades at inference.
        if self.freeze_classifier:
            for cls_layer in self.class_embed:
                for param in cls_layer.parameters():
                    param.requires_grad_(False)

        # set topk boxes selected for inference
        self.select_box_nums_for_evaluation = select_box_nums_for_evaluation

        # OV-DQUO Eq. 6 — learnable wildcard query embedding q* (in decoder embed space).
        # Initialized to the projected mean of all class embeddings (a neutral "any object" prior).
        self.wildcard_query_embed = nn.Parameter(torch.zeros(embed_dim))

        content_query_embedding = torch.tensor(np.load(query_path), dtype=torch.float32, device=device).contiguous()
        self.content_query_embedding = F.normalize(content_query_embedding, p=2, dim=1)

        eval_content_query_embedding = torch.tensor(np.load(eval_query_path), dtype=torch.float32, device=device).contiguous()
        self.eval_content_query_embedding = F.normalize(eval_content_query_embedding, p=2, dim=1)
        # self.eval_content_id = torch.tensor(np.load(eval_id_path), dtype=torch.int64, device=device)
        if vlm_query_path:
            vlm_content_query_embedding = torch.tensor(np.load(vlm_query_path), dtype=torch.float32, device=device).contiguous()# [1203, 768]
            self.vlm_content_query_embedding = F.normalize(vlm_content_query_embedding, p=2, dim=1)
        
        _, feat_dim = self.content_query_embedding.shape
        self.content_layer = nn.Linear(feat_dim, embed_dim)

        # OV-DQUO DTQ: CLIP text embeddings for denoising groups.
        # Projected through the shared content_layer to keep the same embedding space.
        # register_buffer(None) first so PyTorch 2.x doesn't complain about duplicate attrs.
        self.register_buffer("text_query_embedding", None)
        if text_query_path:
            text_emb = torch.tensor(
                np.load(text_query_path), dtype=torch.float32
            ).contiguous()  # keep on CPU; content_layer is on CPU during __init__
            text_emb = F.normalize(text_emb, p=2, dim=1)  # [num_classes, feat_dim]
            with torch.no_grad():
                text_emb_proj = self.content_layer(text_emb)  # [num_classes, embed_dim]
                text_emb_proj = F.normalize(text_emb_proj, p=2, dim=1)
            self.text_query_embedding = text_emb_proj  # assign into the registered buffer slot

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
            clip_head = torch.load(clip_head_path, weights_only=False)
            self.identical, self.thead = clip_head[0]
            self.head = clip_head[1]

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

            # Class-specific WQI: store raw 768-dim novel class embeddings as a buffer so
            # they get moved to device with the model. We project through content_layer each
            # forward pass (weights change during training) to get 256-dim embeddings.
            novel_raw = self.eval_content_query_embedding[self.novel_idx.to(self.eval_content_query_embedding.device)]  # [n_novel, 768]
            self.register_buffer("novel_raw_embeds", novel_raw.clone())
        else:
            self.register_buffer("novel_raw_embeds", None)
        if isinstance(save_dir, str):
            normalized_save_dir = save_dir.strip()
            if normalized_save_dir.lower() in {"", "none", "null"}:
                save_dir = None
            else:
                save_dir = normalized_save_dir
        self.save_predictions = save_predictions
        self.save_dir = save_dir
        if self.save_predictions and self.save_dir:
            os.makedirs(self.save_dir, exist_ok=True)
        self.unknown_vis_dir = None
        self.unknown_vis_max_images = 8
        self.unknown_vis_saved = 0
        if self.save_predictions and self.save_dir:
            self.unknown_vis_dir = os.path.join(self.save_dir, "unknown_proposals")
            os.makedirs(self.unknown_vis_dir, exist_ok=True)

    def filter_content_info(self, batched_inputs):
        freq_weight = self.freq_weight if self.freq_weight is not None else torch.ones(self.num_classes, device=self.device)
        inner_gt = []
        for target in batched_inputs:
            target = target['instances'].gt_classes
            inner_gt.append(target)
        inner_gt = torch.cat(inner_gt)

        # Exclude pseudo-label class (-1) from federated class sampling.
        # Pseudo labels are class-agnostic and should not influence which base classes
        # are sampled for the federated loss. Including -1 corrupts content_inds via
        # PyTorch's negative-index wraparound (cluster_label[-1] = last element).
        inner_gt_real = inner_gt[inner_gt >= 0]
        if inner_gt_real.numel() == 0:
            # Edge case: all annotations in the batch are pseudo — fall back to zeros
            inner_gt_real = inner_gt.new_zeros(1)

        if self.cluster_fed_loss:
            content_inds = get_cluster_fed_loss_inds(
                inner_gt_real,
                num_sample_cats=self.fed_loss_num_cat,
                C=self.num_classes,
                weight=freq_weight,
                cluster_label=self.cluster_label)
        else:
            content_inds = get_fed_loss_inds(
                inner_gt_real,
                num_sample_cats=self.fed_loss_num_cat,
                C=self.num_classes,
                weight=freq_weight)

        convert_map = torch.ones(self.num_classes, dtype=torch.int64) * -1
        for idx, content_id in enumerate(content_inds):
            convert_map[content_id.item()] = idx
        for idx, target in enumerate(batched_inputs):
            orig_classes = batched_inputs[idx]['instances'].gt_classes
            # Clamp to [0, num_classes-1] before indexing to avoid negative-index wraparound.
            # Pseudo labels (class=-1) are restored to -1 after the mapping so they
            # remain distinguishable as class-agnostic in the criterion.
            mapped = convert_map[orig_classes.clamp(min=0)]
            mapped[orig_classes < 0] = -1
            batched_inputs[idx]['instances'].gt_classes = mapped

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
        if self.save_predictions and self.save_dir:
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

        if self.training:
            content_query_embeds = self.content_query_embedding[content_inds] if content_inds is not None else self.content_query_embedding
            content_query_embeds = self.content_layer(content_query_embeds)
            content_query_embeds = F.normalize(content_query_embeds, p=2, dim=1)
        else:
            content_query_embeds = self.content_layer(self.eval_content_query_embedding)
            content_query_embeds = F.normalize(content_query_embeds, p=2, dim=1)
 
        # denoising preprocessing
        # prepare label query embedding
        if self.training:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
            targets = self.prepare_targets(gt_instances)
            cdn_num_classes = self.fed_loss_num_cat if self.use_fed_loss else self.num_classes
            # OV-DQUO DTQ: select text embeddings for the currently sampled classes.
            text_query_embeds = None
            if self.text_query_embedding is not None:
                if content_inds is not None:
                    text_query_embeds = self.text_query_embedding[content_inds]
                else:
                    text_query_embeds = self.text_query_embedding
            input_query_label, input_query_bbox, attn_mask, dn_meta = self.prepare_for_cdn(
                targets,
                dn_number=self.dn_number,
                label_noise_ratio=self.label_noise_ratio,
                box_noise_scale=self.box_noise_scale,
                num_queries=self.num_queries,
                num_classes=cdn_num_classes,
                hidden_dim=self.embed_dim,
                # label_enc=self.label_enc,
                content_query_embeds=content_query_embeds,
                text_query_embeds=text_query_embeds,
            )
        else:
            targets = None
            input_query_label, input_query_bbox, attn_mask, dn_meta = None, None, None, None
        query_embeds = (input_query_label, input_query_bbox)

        # Class-specific WQI: project novel class raw embeddings through the current
        # content_layer (weights evolve during training) to get 256-dim representations.
        # Each unknown encoder proposal will be assigned its best-matching novel class
        # embedding instead of the generic wildcard.
        novel_content_embeds = None
        if self.training and self.novel_raw_embeds is not None:
            with torch.no_grad():
                novel_content_embeds = self.content_layer(self.novel_raw_embeds)  # [n_novel, 256]
                novel_content_embeds = F.normalize(novel_content_embeds, p=2, dim=1)

        # Compute warmup parameters before transformer so we can pass cur_iou_thr
        # into the transformer for wildcard query injection (OV-DQUO Eq. 6).
        if self.training:
            self._train_forward_count += 1
            warmup_scale = (
                min(1.0, self._train_forward_count / self.unknown_warmup_iters)
                if self.unknown_warmup_iters > 0
                else 1.0
            )
            cur_iou_thr = self.unknown_iou_thr_start + (
                self.unknown_iou_thr - self.unknown_iou_thr_start
            ) * warmup_scale
            cur_match_cost = self.unknown_match_cost * warmup_scale
            cur_match_mode = self.unknown_match_mode
            if cur_match_mode == "hard" and warmup_scale < 1.0:
                cur_match_mode = "soft"
        else:
            cur_iou_thr = self.unknown_iou_thr
            cur_match_cost = self.unknown_match_cost
            cur_match_mode = self.unknown_match_mode

        # feed into transformer
        (
            inter_states,
            init_reference,
            inter_references,
            enc_state,
            enc_reference,  # [0..1]
            proposal_unknown_mask,
            proposal_max_iou,
        ) = self.transformer(
            multi_level_feats,
            multi_level_masks,
            multi_level_position_embeddings,
            query_embeds,
            attn_masks=[attn_mask, None],
            content_query_embeds=content_query_embeds,
            content_inds=content_inds,
            wildcard_content=self.wildcard_query_embed if self.training else None,
            novel_content_embeds=novel_content_embeds,
            targets=targets,
            unknown_iou_thr=cur_iou_thr,
            pseudo_score_thr=self.pseudo_score_thr,
        )
        # hack implementation for distributed training
        # inter_states[0] += self.label_enc.weight[0, 0] * 0.0
        inter_states[0] += self.content_layer.weight[0, 0] * 0.0
        inter_states[0] += self.wildcard_query_embed[0] * 0.0

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
            unknown_encoder_proposals = [enc_reference[i][proposal_unknown_mask[i]] for i in range(enc_reference.shape[0])]
            output["proposal_unknown_mask"] = proposal_unknown_mask
            output["proposal_unknown_max_iou"] = proposal_max_iou
            output["unknown_encoder_proposals"] = unknown_encoder_proposals
            output["unknown_match_mode"] = cur_match_mode
            output["unknown_match_cost"] = cur_match_cost
            self._save_unknown_proposal_visuals(
                batched_inputs,
                targets,
                enc_reference,
                proposal_unknown_mask,
                proposal_max_iou,
            )
            if self.aux_loss:
                for aux_output in output["aux_outputs"]:
                    aux_output["proposal_unknown_mask"] = proposal_unknown_mask
                    aux_output["unknown_match_mode"] = cur_match_mode
                    aux_output["unknown_match_cost"] = cur_match_cost
            output["enc_outputs"]["proposal_unknown_mask"] = proposal_unknown_mask
            output["enc_outputs"]["unknown_match_mode"] = cur_match_mode
            output["enc_outputs"]["unknown_match_cost"] = cur_match_cost

        if self.training:
            loss_dict = self.criterion(output, targets, dn_meta)
            if "proposal_unknown_mask" in output:
                unknown_mask = output["proposal_unknown_mask"]
                unknown_counts = unknown_mask.sum(dim=1).float()
                pseudo_counts = torch.stack(
                    [t["pseudo_mask"].sum() for t in targets], dim=0
                ).to(unknown_counts.device).float()
                selected_iou = output["proposal_unknown_max_iou"][unknown_mask]
                loss_dict["debug_unknown_prop_count_mean"] = unknown_counts.mean().detach()
                loss_dict["debug_unknown_prop_ratio_mean"] = (
                    unknown_counts / unknown_mask.shape[1]
                ).mean().detach()
                loss_dict["debug_pseudo_gt_count_mean"] = pseudo_counts.mean().detach()
                loss_dict["debug_unknown_prop_iou_mean"] = (
                    selected_iou.mean().detach()
                    if selected_iou.numel() > 0
                    else unknown_counts.new_tensor(0.0)
                )
                loss_dict["debug_unknown_iou_thr"] = unknown_counts.new_tensor(cur_iou_thr)
                loss_dict["debug_unknown_match_cost"] = unknown_counts.new_tensor(cur_match_cost)
            weight_dict = self.criterion.weight_dict
            for k in loss_dict.keys():
                if k in weight_dict:
                    loss_dict[k] *= weight_dict[k]
            return loss_dict
        else:
            box_cls = output["pred_logits"]
            box_pred = output["pred_boxes"]
            if self.save_predictions and self.save_dir and not self.score_ensemble:
                save_output = {}
                save_output["pred_logits"] = copy.deepcopy(output["pred_logits"]).cpu()
                save_output["pred_boxes"] = copy.deepcopy(output["pred_boxes"]).cpu()
                save_path = os.path.join(self.save_dir, filename)
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                torch.save(save_output, save_path)
            if self.score_ensemble:
                roi_features_ori = self.extract_region_feature(features_wonorm, box_pred, 'p3')

                if self.save_predictions and self.save_dir:
                    save_output = {}
                    save_output["pred_logits"] = copy.deepcopy(output["pred_logits"]).cpu()
                    save_output["roi_features_ori"] = copy.deepcopy(roi_features_ori).cpu()# [1, 900, 768]
                    save_output["pred_boxes"] = copy.deepcopy(output["pred_boxes"]).cpu()
                    save_path = os.path.join(self.save_dir, filename)
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    torch.save(save_output, save_path)

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

    @torch.no_grad()
    def select_unknown_encoder_proposals(self, enc_reference, targets, thr=0.5):
        """Select encoder top-k proposals overlapping pseudo/unknown targets."""
        return select_unknown_proposals(
            enc_reference, targets, thr=thr, pseudo_score_thr=self.pseudo_score_thr
        )

    @torch.no_grad()
    def _save_unknown_proposal_visuals(
        self,
        batched_inputs,
        targets,
        enc_reference,
        proposal_unknown_mask,
        proposal_unknown_max_iou,
        max_draw_per_image=50,
    ):
        if not self.unknown_vis_dir or self.unknown_vis_saved >= self.unknown_vis_max_images:
            return

        for img_idx, (sample, target) in enumerate(zip(batched_inputs, targets)):
            if self.unknown_vis_saved >= self.unknown_vis_max_images:
                break

            img = sample["image"].detach().cpu()
            if img.dtype != torch.uint8:
                img = img.clamp(0, 255).to(torch.uint8)
            if img.dim() != 3:
                continue
            if img.shape[0] == 1:
                img = img.repeat(3, 1, 1)
            if img.shape[0] != 3:
                continue

            h, w = img.shape[-2:]
            pseudo_mask = target["pseudo_mask"].to(torch.bool)
            pseudo_boxes = target["boxes"][pseudo_mask]
            if pseudo_boxes.numel() > 0:
                pseudo_boxes = box_cxcywh_to_xyxy(pseudo_boxes)
                pseudo_boxes = pseudo_boxes * pseudo_boxes.new_tensor([w, h, w, h])
                pseudo_boxes = pseudo_boxes.clamp(min=0)
            else:
                pseudo_boxes = torch.zeros((0, 4), dtype=torch.float32)

            unk_mask = proposal_unknown_mask[img_idx]
            unk_boxes = enc_reference[img_idx][unk_mask]
            unk_ious = proposal_unknown_max_iou[img_idx][unk_mask]
            d2_logger.warning(
                "[unknown_vis] img_idx=%d pseudo=%d unk=%d",
                img_idx,
                int(pseudo_mask.sum().item()),
                int(unk_mask.sum().item()),
            )
            with open(os.path.join(self.unknown_vis_dir, "debug_counts.txt"), "a") as f:
                f.write(
                    f"img_idx={img_idx} pseudo={int(pseudo_mask.sum().item())} "
                    f"unk={int(unk_mask.sum().item())}\n"
                )
            if unk_boxes.numel() > 0:
                if unk_boxes.shape[0] > max_draw_per_image:
                    top_idx = torch.topk(unk_ious, k=max_draw_per_image).indices
                    unk_boxes = unk_boxes[top_idx]
                    unk_ious = unk_ious[top_idx]
                unk_boxes = box_cxcywh_to_xyxy(unk_boxes)
                unk_boxes = unk_boxes * unk_boxes.new_tensor([w, h, w, h])
                unk_boxes = unk_boxes.clamp(min=0)
                unk_labels = [f"u:{float(v):.2f}" for v in unk_ious.detach().cpu()]
            else:
                unk_boxes = torch.zeros((0, 4), dtype=torch.float32)
                unk_labels = None

            canvas = img
            if pseudo_boxes.numel() > 0:
                canvas = torchvision.utils.draw_bounding_boxes(
                    canvas,
                    pseudo_boxes.cpu(),
                    colors="red",
                    width=3,
                    labels=["pseudo"] * pseudo_boxes.shape[0],
                )
            if unk_boxes.numel() > 0:
                canvas = torchvision.utils.draw_bounding_boxes(
                    canvas,
                    unk_boxes.cpu(),
                    colors="cyan",
                    width=2,
                    labels=unk_labels,
                )

            stem = os.path.splitext(os.path.basename(sample.get("file_name", f"img_{img_idx}.jpg")))[0]
            out_path = os.path.join(self.unknown_vis_dir, f"{self.unknown_vis_saved:03d}_{stem}.png")
            torchvision.io.write_png(canvas.cpu(), out_path)
            self.unknown_vis_saved += 1
    
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
        text_query_embeds=None,
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

        # OV-DQUO DTQ: prefer CLIP text embeddings for DN groups (cross-modal denoising).
        # Fallback to visual embeddings if text embeddings are not available.
        # Clamp to avoid negative-index wraparound for pseudo labels (class=-1);
        # their DN loss is zeroed by pseudo_weight=0.0 so embedding choice is harmless.
        dn_embeds = text_query_embeds if text_query_embeds is not None else content_query_embeds
        if dn_embeds is not None:
            input_label_embed = dn_embeds[m.clamp(min=0)]

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
            if targets_per_image.has("gt_is_pseudo"):
                pseudo_mask = targets_per_image.gt_is_pseudo.to(torch.bool)
            elif targets_per_image.has("gt_pseudo"):
                pseudo_mask = targets_per_image.gt_pseudo.to(torch.bool)
            elif targets_per_image.has("gt_unknown"):
                pseudo_mask = targets_per_image.gt_unknown.to(torch.bool)
            else:
                pseudo_mask = gt_scores < self.pseudo_score_thr
            target_dict = {
                "labels": gt_classes,
                "boxes": gt_boxes,
                "scores": gt_scores,
                "pseudo_mask": pseudo_mask,
                "pseudo_score_thr": self.pseudo_score_thr,
            }
            if targets_per_image.has("gt_pseudo_weight"):
                target_dict["pseudo_weight"] = targets_per_image.gt_pseudo_weight.to(
                    device=self.device, dtype=torch.float32
                )
            new_targets.append(
                target_dict
            )
        return new_targets
