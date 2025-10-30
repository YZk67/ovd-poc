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

from typing import Dict, Optional
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from detrex.layers import (
    FFN,
    MLP,
    BaseTransformerLayer,
    MultiheadAttention,
    MultiScaleDeformableAttention,
    TransformerLayerSequence,
    get_sine_pos_embed,
)
from detrex.utils import inverse_sigmoid
from lami_dino.models.rpsa import RPSAModule, build_token_class_mask_from_logits
from detectron2.utils.logger import setup_logger

logger = setup_logger()  # 使用detectron2的logger，确保日志级别正确


class DINOTransformerEncoder(TransformerLayerSequence):
    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        feedforward_dim: int = 1024,
        attn_dropout: float = 0.1,
        ffn_dropout: float = 0.1,
        num_layers: int = 6,
        post_norm: bool = False,
        num_feature_levels: int = 4,
    ):
        super(DINOTransformerEncoder, self).__init__(
            transformer_layers=BaseTransformerLayer(
                attn=MultiScaleDeformableAttention(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    dropout=attn_dropout,
                    batch_first=True,
                    num_levels=num_feature_levels,
                ),
                ffn=FFN(
                    embed_dim=embed_dim,
                    feedforward_dim=feedforward_dim,
                    output_dim=embed_dim,
                    num_fcs=2,
                    ffn_drop=ffn_dropout,
                ),
                norm=nn.LayerNorm(embed_dim),
                operation_order=("self_attn", "norm", "ffn", "norm"),
            ),
            num_layers=num_layers,
        )
        self.embed_dim = self.layers[0].embed_dim
        self.pre_norm = self.layers[0].pre_norm

        if post_norm:
            self.post_norm_layer = nn.LayerNorm(self.embed_dim)
        else:
            self.post_norm_layer = None

    def forward(
        self,
        query,
        key,
        value,
        query_pos=None,
        key_pos=None,
        attn_masks=None,
        query_key_padding_mask=None,
        key_padding_mask=None,
        **kwargs,
    ):

        for layer in self.layers:
            query = layer(
                query,
                key,
                value,
                query_pos=query_pos,
                attn_masks=attn_masks,
                query_key_padding_mask=query_key_padding_mask,
                key_padding_mask=key_padding_mask,
                **kwargs,
            )

        if self.post_norm_layer is not None:
            query = self.post_norm_layer(query)
        return query


class DINOTransformerDecoder(TransformerLayerSequence):
    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        feedforward_dim: int = 1024,
        attn_dropout: float = 0.1,
        ffn_dropout: float = 0.1,
        num_layers: int = 6,
        return_intermediate: bool = True,
        num_feature_levels: int = 4,
        look_forward_twice=True,
    ):
        super(DINOTransformerDecoder, self).__init__(
            transformer_layers=BaseTransformerLayer(
                attn=[
                    MultiheadAttention(
                        embed_dim=embed_dim,
                        num_heads=num_heads,
                        attn_drop=attn_dropout,
                        batch_first=True,
                    ),
                    MultiScaleDeformableAttention(
                        embed_dim=embed_dim,
                        num_heads=num_heads,
                        dropout=attn_dropout,
                        batch_first=True,
                        num_levels=num_feature_levels,
                    ),
                ],
                ffn=FFN(
                    embed_dim=embed_dim,
                    feedforward_dim=feedforward_dim,
                    output_dim=embed_dim,
                    ffn_drop=ffn_dropout,
                ),
                norm=nn.LayerNorm(embed_dim),
                operation_order=("self_attn", "norm", "cross_attn", "norm", "ffn", "norm"),
            ),
            num_layers=num_layers,
        )
        self.return_intermediate = return_intermediate

        self.ref_point_head = MLP(2 * embed_dim, embed_dim, embed_dim, 2)

        self.bbox_embed = None
        self.class_embed = None
        self.look_forward_twice = look_forward_twice
        self.norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        query,
        key,
        value,
        query_pos=None,
        key_pos=None,
        attn_masks=None,
        query_key_padding_mask=None,
        key_padding_mask=None,
        reference_points=None,  # num_queries, 4. normalized.
        valid_ratios=None,
        **kwargs,
    ):
        output = query
        bs, num_queries, _ = output.size()
        if reference_points.dim() == 2:
            reference_points = reference_points.unsqueeze(0).repeat(bs, 1, 1)  # bs, num_queries, 4

        intermediate = []
        intermediate_reference_points = []
        for layer_idx, layer in enumerate(self.layers):
            if reference_points.shape[-1] == 4:
                reference_points_input = (
                    reference_points[:, :, None]
                    * torch.cat([valid_ratios, valid_ratios], -1)[:, None]
                )
            else:
                assert reference_points.shape[-1] == 2
                reference_points_input = reference_points[:, :, None] * valid_ratios[:, None]

            query_sine_embed = get_sine_pos_embed(reference_points_input[:, :, 0, :])
            query_pos = self.ref_point_head(query_sine_embed)

            output = layer(
                output,
                key,
                value,
                query_pos=query_pos,
                key_pos=key_pos,
                query_sine_embed=query_sine_embed,
                attn_masks=attn_masks,
                query_key_padding_mask=query_key_padding_mask,
                key_padding_mask=key_padding_mask,
                reference_points=reference_points_input,
                **kwargs,
            )

            if self.bbox_embed is not None:
                tmp = self.bbox_embed[layer_idx](output)
                if reference_points.shape[-1] == 4:
                    new_reference_points = tmp + inverse_sigmoid(reference_points)
                    new_reference_points = new_reference_points.sigmoid()
                else:
                    assert reference_points.shape[-1] == 2
                    new_reference_points = tmp
                    new_reference_points[..., :2] = tmp[..., :2] + inverse_sigmoid(reference_points)
                    new_reference_points = new_reference_points.sigmoid()
                reference_points = new_reference_points.detach()

            if self.return_intermediate:
                intermediate.append(self.norm(output))
                if self.look_forward_twice:
                    intermediate_reference_points.append(new_reference_points)
                else:
                    intermediate_reference_points.append(reference_points)

        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(intermediate_reference_points)

        return output, reference_points


class DINOTransformer(nn.Module):
    """Transformer module for DINO

    Args:
        encoder (nn.Module): encoder module.
        decoder (nn.Module): decoder module.
        as_two_stage (bool): whether to use two-stage transformer. Default False.
        num_feature_levels (int): number of feature levels. Default 4.
        two_stage_num_proposals (int): number of proposals in two-stage transformer. Default 900.
    """

    def __init__(
        self,
        encoder=None,
        decoder=None,
        num_feature_levels=4,
        two_stage_num_proposals=900,
        # learnt_init_query=False,
        # === RPSA kwargs (config-driven) ===
        use_rpsa: bool = True,
        rpsa_module: Optional[nn.Module] = None,
        rpsa_kwargs: Optional[Dict] = None,
    ):
        super(DINOTransformer, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.num_feature_levels = num_feature_levels
        self.two_stage_num_proposals = two_stage_num_proposals

        self.embed_dim = self.encoder.embed_dim

        self.level_embeds = nn.Parameter(torch.Tensor(self.num_feature_levels, self.embed_dim))
        # self.learnt_init_query = learnt_init_query
        # if self.learnt_init_query:
        #     self.tgt_embed = nn.Embedding(self.two_stage_num_proposals, self.embed_dim)
        self.enc_output = nn.Linear(self.embed_dim, self.embed_dim)
        self.enc_output_norm = nn.LayerNorm(self.embed_dim)

        self.init_weights()

        # === RPSA initialization (minimal intrusive) ===
        self.use_rpsa = use_rpsa
        self.rpsa_last_loss = None
        self.rpsa_last_stats = {}
        self.rpsa = None
        if self.use_rpsa:
            if rpsa_module is not None:
                self.rpsa = rpsa_module
                logger.info(f"[RPSA] ✅ Initialized with provided module: {type(self.rpsa)}")
                print(f"[RPSA] ✅ Initialized: {type(self.rpsa)}")  # 确保输出
            else:
                cfg = rpsa_kwargs or {}
                self.rpsa = RPSAModule(**cfg)
                logger.info(f"[RPSA] ✅ Initialized with kwargs: {cfg}")
                print(f"[RPSA] ✅ Initialized with kwargs: {cfg}")  # 确保输出
        else:
            logger.info("[RPSA] ⚠️ use_rpsa=False, RPSA module not initialized")
            print("[RPSA] ⚠️ use_rpsa=False, not initialized")  # 确保输出

    def init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MultiScaleDeformableAttention):
                m.init_weights()
        nn.init.normal_(self.level_embeds)

    def gen_encoder_output_proposals(self, memory, memory_padding_mask, spatial_shapes):
        N, S, C = memory.shape
        proposals = []
        _cur = 0
        for lvl, (H, W) in enumerate(spatial_shapes):
            mask_flatten_ = memory_padding_mask[:, _cur : (_cur + H * W)].view(N, H, W, 1)
            valid_H = torch.sum(~mask_flatten_[:, :, 0, 0], 1)
            valid_W = torch.sum(~mask_flatten_[:, 0, :, 0], 1)

            grid_y, grid_x = torch.meshgrid(
                torch.linspace(0, H - 1, H, dtype=torch.float32, device=memory.device),
                torch.linspace(0, W - 1, W, dtype=torch.float32, device=memory.device),
            )
            grid = torch.cat([grid_x.unsqueeze(-1), grid_y.unsqueeze(-1)], -1)

            scale = torch.cat([valid_W.unsqueeze(-1), valid_H.unsqueeze(-1)], 1).view(N, 1, 1, 2)
            grid = (grid.unsqueeze(0).expand(N, -1, -1, -1) + 0.5) / scale
            wh = torch.ones_like(grid) * 0.05 * (2.0**lvl)
            proposal = torch.cat((grid, wh), -1).view(N, -1, 4)
            proposals.append(proposal)
            _cur += H * W

        output_proposals = torch.cat(proposals, 1)
        output_proposals_valid = ((output_proposals > 0.01) & (output_proposals < 0.99)).all(
            -1, keepdim=True
        )
        output_proposals = torch.log(output_proposals / (1 - output_proposals))
        output_proposals = output_proposals.masked_fill(
            memory_padding_mask.unsqueeze(-1), float("inf")
        )
        output_proposals = output_proposals.masked_fill(~output_proposals_valid, float("inf"))

        output_memory = memory
        output_memory = output_memory.masked_fill(memory_padding_mask.unsqueeze(-1), float(0))
        output_memory = output_memory.masked_fill(~output_proposals_valid, float(0))
        output_memory = self.enc_output_norm(self.enc_output(output_memory))
        return output_memory, output_proposals

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        """Get the reference points used in decoder.

        Args:
            spatial_shapes (Tensor): The shape of all
                feature maps, has shape (num_level, 2).
            valid_ratios (Tensor): The ratios of valid
                points on the feature map, has shape
                (bs, num_levels, 2)
            device (obj:`device`): The device where
                reference_points should be.

        Returns:
            Tensor: reference points used in decoder, has \
                shape (bs, num_keys, num_levels, 2).
        """
        reference_points_list = []
        for lvl, (H, W) in enumerate(spatial_shapes):
            #  TODO  check this 0.5
            ref_y, ref_x = torch.meshgrid(
                torch.linspace(0.5, H - 0.5, H, dtype=torch.float32, device=device),
                torch.linspace(0.5, W - 0.5, W, dtype=torch.float32, device=device),
            )
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points

    def get_valid_ratio(self, mask):
        """Get the valid ratios of feature maps of all levels."""
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    def forward(
        self,
        multi_level_feats,
        multi_level_masks,
        multi_level_pos_embeds,
        query_embed,
        attn_masks,
        content_query_embeds,
        content_inds=None,
        **kwargs,
    ):
        feat_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        for lvl, (feat, mask, pos_embed) in enumerate(
            zip(multi_level_feats, multi_level_masks, multi_level_pos_embeds)
        ):
            bs, c, h, w = feat.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)

            feat = feat.flatten(2).transpose(1, 2)  # bs, hw, c
            mask = mask.flatten(1)
            pos_embed = pos_embed.flatten(2).transpose(1, 2)  # bs, hw, c
            lvl_pos_embed = pos_embed + self.level_embeds[lvl].view(1, 1, -1)
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            feat_flatten.append(feat)
            mask_flatten.append(mask)
        feat_flatten = torch.cat(feat_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        spatial_shapes = torch.as_tensor(
            spatial_shapes, dtype=torch.long, device=feat_flatten.device
        )
        level_start_index = torch.cat(
            (spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1])
        )
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in multi_level_masks], 1)

        reference_points = self.get_reference_points(
            spatial_shapes, valid_ratios, device=feat.device
        )

        memory = self.encoder(
            query=feat_flatten,
            key=None,
            value=None,
            query_pos=lvl_pos_embed_flatten,
            query_key_padding_mask=mask_flatten,
            spatial_shapes=spatial_shapes,
            reference_points=reference_points,  # bs, num_token, num_level, 2
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
            **kwargs,
        )

        output_memory, output_proposals = self.gen_encoder_output_proposals(
            memory, mask_flatten, spatial_shapes
        )
        # output_memory: bs, num_tokens, c
        # output_proposals: bs, num_tokens, 4. unsigmoided.

        # enc_outputs_class = self.decoder.class_embed[self.decoder.num_layers](output_memory, content_inds=content_inds)
        text_classifier = self.decoder.class_embed[self.decoder.num_layers]
        enc_outputs_class = text_classifier(output_memory, content_inds=content_inds)
        apr_loss = getattr(text_classifier, "apr_loss", None)

        # === RPSA: compute alignment loss (non-breaking: stash on classifier/transformer) ===
        use_rpsa_flag = getattr(self, "use_rpsa", False)
        is_training = self.training
        rpsa_module_exists = self.rpsa is not None
        if use_rpsa_flag and is_training and rpsa_module_exists:
            try:
                logger.info(f"[RPSA] Computing RPSA loss: use_rpsa={use_rpsa_flag}, training={is_training}, rpsa={rpsa_module_exists}")
                # 1) token->class 软掩码（从 encoder 分类 logits 构造）
                token_cls_mask = build_token_class_mask_from_logits(enc_outputs_class, topL=5).detach()  # [B,N,C]

                # 2) 文本原型，保证形状为 [C,Kp,D]
                #    这里沿用你当前变量 content_query_embeds（TPA/投影后的原型）
                if content_query_embeds.dim() == 2:  # [C,D] -> [C,1,D]
                    text_protos_for_rpsa = content_query_embeds.unsqueeze(1)
                elif content_query_embeds.dim() == 3:  # [C,Kp,D]
                    text_protos_for_rpsa = content_query_embeds
                else:
                    logger.warning(f"[RPSA] Unexpected content_query_embeds shape: {content_query_embeds.shape}")
                    raise ValueError(f"content_query_embeds must be [C,D] or [C,Kp,D], got {content_query_embeds.shape}")

                # 3) 计算 RPSA
                loss_rpsa_raw, rpsa_stats, _ = self.rpsa(
                    region_feats=output_memory,             # [B,N,D]
                    text_protos=text_protos_for_rpsa,       # [C,Kp,D]
                    token_cls_mask=token_cls_mask,          # [B,N,C]
                )
                loss_rpsa = loss_rpsa_raw

                # 4) 暂存，保持与你 APR 的用法一致（不改变返回签名）
                self.rpsa_last_loss = loss_rpsa.detach()
                self.rpsa_last_stats = rpsa_stats
                # RPSA计算使用encoder分类器(class_embed[num_layers])，
                # 损失存储也使用同一个分类器
                setattr(text_classifier, "rpsa_loss", loss_rpsa)
                setattr(text_classifier, "rpsa_stats", rpsa_stats)
                logger.info(f"[RPSA] ✅ Loss computed successfully: {loss_rpsa.item():.6f}, stored on text_classifier")
                print(f"[RPSA] ✅ Loss computed: {loss_rpsa.item():.6f}")  # 确保输出

            except (ValueError, RuntimeError) as e:
                logger.warning(f"[RPSA] ⚠️ skipped due to: {e}")
                print(f"[RPSA] ⚠️ Skipped: {e}")  # 确保输出
                import traceback
                logger.debug(f"[RPSA] Traceback: {traceback.format_exc()}")
            except Exception as e:
                logger.error(f"[RPSA] ❌ Unexpected error: {e}")
                print(f"[RPSA] ❌ Error: {e}")  # 确保输出
                import traceback
                logger.error(f"[RPSA] Traceback: {traceback.format_exc()}")
                raise
        else:
            # 使用info级别，这样即使debug关闭也能看到
            if not use_rpsa_flag:
                logger.info(f"[RPSA] ⚠️ Skipped: use_rpsa=False")
                print("[RPSA] ⚠️ Skipped: use_rpsa=False")  # 确保输出
            elif not is_training:
                logger.info(f"[RPSA] ⚠️ Skipped: not in training mode")
                print("[RPSA] ⚠️ Skipped: not in training mode")  # 确保输出
            elif not rpsa_module_exists:
                logger.warning(f"[RPSA] ❌ Skipped: rpsa module is None! Check initialization.")
                print("[RPSA] ❌ Skipped: rpsa module is None!")  # 确保输出


        enc_outputs_coord_unact = (
            self.decoder.bbox_embed[self.decoder.num_layers](output_memory) + output_proposals
        )  # unsigmoided.

        max_scores, max_labels = torch.max(enc_outputs_class, dim=-1)
        topk = self.two_stage_num_proposals
        
        # Check if we have multi-prototype embeddings and should use soft-attention
        if hasattr(self, 'use_soft_attention') and self.use_soft_attention and content_query_embeds.ndim == 3:
            # Multi-prototype mode with soft-attention aggregation
            # content_query_embeds: [C, K, embed_dim] where K is num_prototypes
            num_classes, num_prototypes, embed_dim = content_query_embeds.shape
            
            # Normalize embeddings and region features
            embeddings_norm = F.normalize(content_query_embeds, p=2, dim=-1)  # [C, K, embed_dim]
            region_feats_norm = F.normalize(output_memory.detach(), p=2, dim=-1)  # [B, N, embed_dim]
            
            # Compute similarity: [B, N, embed_dim] @ [C, K, embed_dim]^T -> [B, C, N, K]
            sim = torch.einsum("bnd,ckd->bcnk", region_feats_norm, embeddings_norm)
            
            # Soft-attention weights: α_i,c,k = softmax(cos(f_i, t_c,k) / τ)
            tau = getattr(self, 'soft_attention_tau', 0.1)
            alpha = F.softmax(sim / tau, dim=-1)  # [B, C, N, K]
            
            # Weighted aggregation: s_i,c = sum_k α_i,c,k * cos(f_i, t_c,k)
            sim_aggregated = (alpha * sim).sum(dim=-1)  # [B, C, N]
            
            # Get max scores and labels from soft-attention aggregated similarities
            max_scores_soft, max_labels_soft = torch.max(sim_aggregated, dim=1)  # [B, N]
            
            # Use soft-attention based selection for top-k proposals
            topk_proposals = torch.topk(max_scores_soft, topk, dim=1)[1]
            content_ids = torch.gather(max_labels_soft, 1, topk_proposals)
            
            # extract region proposal boxes
            topk_coords_unact = torch.gather(
                enc_outputs_coord_unact, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, 4)
            )  # unsigmoided.
            reference_points = topk_coords_unact.detach().sigmoid()
            if query_embed[1] is not None:
                reference_points = torch.cat([query_embed[1].sigmoid(), reference_points], 1)
            init_reference_out = reference_points

            # extract region features
            target_unact = torch.gather(
                output_memory, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, output_memory.shape[-1])
            )
            
            # For each selected region, compute its soft-attention aggregated embedding
            # Gather the attention weights for selected regions: [B, topk, C, K]
            selected_alpha = torch.gather(alpha, 2, topk_proposals.unsqueeze(1).unsqueeze(-1).repeat(1, num_classes, 1, num_prototypes))
            
            # For each selected region, compute weighted aggregation based on its class
            content_query_list = []
            for b in range(bs):
                batch_queries = []
                for k in range(topk):
                    class_id = content_ids[b, k]
                    # Get attention weights for this region and class: [K]
                    region_alpha = selected_alpha[b, class_id, k, :]
                    # Weighted aggregation: sum_k α_k * t_c,k
                    weighted_embed = torch.einsum("k,kd->d", region_alpha, content_query_embeds[class_id])
                    batch_queries.append(weighted_embed)
                content_query_list.append(torch.stack(batch_queries))
            content_query = torch.stack(content_query_list)  # [B, topk, embed_dim]
        else:
            # Standard single-prototype mode
            topk_proposals = torch.topk(max_scores, topk, dim=1)[1]
            
            # extract region proposal boxes
            topk_coords_unact = torch.gather(
                enc_outputs_coord_unact, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, 4)
            )  # unsigmoided.
            reference_points = topk_coords_unact.detach().sigmoid()
            if query_embed[1] is not None:
                reference_points = torch.cat([query_embed[1].sigmoid(), reference_points], 1)
            init_reference_out = reference_points

            # extract region features
            target_unact = torch.gather(
                output_memory, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, output_memory.shape[-1])
            )
            
            content_ids = torch.gather(max_labels, 1, topk_proposals)
            embed_dim = content_query_embeds.shape[-1]  # Get actual feature dimension
            content_query = torch.gather(
                content_query_embeds.unsqueeze(0).repeat(bs, 1, 1), 1,
                content_ids.unsqueeze(-1).repeat(1, 1, embed_dim)) 

        target = target_unact.detach() + content_query

        if query_embed[0] is not None:
            target = torch.cat([query_embed[0], target], 1)

        # decoder
        inter_states, inter_references = self.decoder(
            query=target,  # bs, num_queries, embed_dims
            key=memory,  # bs, num_tokens, embed_dims
            value=memory,  # bs, num_tokens, embed_dims
            query_pos=None,
            key_padding_mask=mask_flatten,  # bs, num_tokens
            reference_points=reference_points,  # num_queries, 4
            spatial_shapes=spatial_shapes,  # nlvl, 2
            level_start_index=level_start_index,  # nlvl
            valid_ratios=valid_ratios,  # bs, nlvl, 2
            attn_masks=attn_masks,
            **kwargs,
        )

        inter_references_out = inter_references
        return (
            inter_states,
            init_reference_out,
            inter_references_out,
            target_unact,
            topk_coords_unact.sigmoid(),
            apr_loss,
        )
