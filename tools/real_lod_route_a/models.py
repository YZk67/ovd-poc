"""Real-LOD adapter modules for D3 Route-A experiments."""

from __future__ import annotations

from typing import Sequence, Optional

import torch
from torch import Tensor, nn

from mmdet.registry import MODELS
from mmdet.structures import SampleList

try:
    from real_model.models.detectors.real_model import RealModel
except ImportError:  # pragma: no cover - imported only inside the Real-LOD repo.
    RealModel = None


class ResidualTextQueryAdapter(nn.Module):
    """Zero-initialized residual MLP for Real-Model text features."""

    def __init__(
        self,
        embed_dims: int,
        bottleneck_dim: int = 64,
        dropout: float = 0.0,
        init_scale: float = 1.0,
        learnable_scale: bool = True,
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(embed_dims)
        self.down = nn.Linear(embed_dims, bottleneck_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.up = nn.Linear(bottleneck_dim, embed_dims)
        if learnable_scale:
            self.scale = nn.Parameter(torch.tensor(float(init_scale)))
        else:
            self.register_buffer("scale", torch.tensor(float(init_scale)), persistent=False)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, text_features: Tensor) -> Tensor:
        residual = self.up(self.dropout(self.act(self.down(self.norm(text_features)))))
        return text_features + self.scale.to(dtype=text_features.dtype) * residual


def _bbox_xyxy_to_cxcywh(bboxes: Tensor) -> Tensor:
    x1, y1, x2, y2 = bboxes.unbind(dim=-1)
    return torch.stack(((x1 + x2) * 0.5, (y1 + y2) * 0.5, x2 - x1, y2 - y1), dim=-1)


def _inverse_sigmoid(x: Tensor, eps: float = 1e-3) -> Tensor:
    x = x.clamp(min=0.0, max=1.0)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)


class InImageWrongPhraseCdnQueryGenerator(nn.Module):
    """CDN generator that turns DN negatives into wrong-phrase same-region probes.

    The stock CDN negative half uses a far/noisy box with the original label
    embedding. For D3, the harder confusion is a nearly correct region paired
    with another description from the same image. Targets remain unchanged:
    the head's DN target builder already assigns all-zero token labels to the
    negative half.
    """

    def __init__(
        self,
        base_generator: nn.Module,
        same_region_box_noise_scale: float = 0.25,
        fallback_num_phrases: int | None = None,
    ) -> None:
        super().__init__()
        self.num_classes = int(base_generator.num_classes)
        self.embed_dims = int(base_generator.embed_dims)
        self.num_matching_queries = int(base_generator.num_matching_queries)
        self.label_noise_scale = float(base_generator.label_noise_scale)
        self.box_noise_scale = float(base_generator.box_noise_scale)
        self.dynamic_dn_groups = bool(base_generator.dynamic_dn_groups)
        self.same_region_box_noise_scale = float(same_region_box_noise_scale)
        self.fallback_num_phrases = fallback_num_phrases

        if self.dynamic_dn_groups:
            self.num_dn_queries = int(base_generator.num_dn_queries)
        else:
            self.num_groups = int(base_generator.num_groups)

        self.label_embedding = base_generator.label_embedding

    def forward(self, batch_data_samples: Sequence) -> tuple:
        gt_labels_list = []
        gt_bboxes_list = []
        num_phrase_list = []
        sent_group_ids_list = []
        for sample in batch_data_samples:
            img_h, img_w = sample.img_shape
            bboxes = sample.gt_instances.bboxes
            factor = bboxes.new_tensor([img_w, img_h, img_w, img_h]).unsqueeze(0)
            gt_bboxes_list.append(bboxes / factor)
            gt_labels_list.append(sample.gt_instances.labels)
            num_phrase_list.append(self._num_phrases(sample))
            sent_group_ids_list.append(self._sent_group_ids(sample))

        gt_labels = torch.cat(gt_labels_list)
        gt_bboxes = torch.cat(gt_bboxes_list)
        num_target_list = [len(bboxes) for bboxes in gt_bboxes_list]
        max_num_target = max(num_target_list)
        num_groups = self.get_num_groups(max_num_target)

        wrong_labels = self._build_wrong_labels(
            gt_labels_list,
            num_phrase_list,
            sent_group_ids_list,
        )
        dn_label_query = self.generate_dn_label_query(gt_labels, wrong_labels, num_groups)
        dn_bbox_query = self.generate_dn_bbox_query(gt_bboxes, num_groups)

        batch_idx = torch.cat(
            [
                torch.full_like(labels.long(), batch_id)
                for batch_id, labels in enumerate(gt_labels_list)
            ]
        )
        dn_label_query, dn_bbox_query = self.collate_dn_queries(
            dn_label_query,
            dn_bbox_query,
            batch_idx,
            len(batch_data_samples),
            num_groups,
        )
        attn_mask = self.generate_dn_mask(
            max_num_target,
            num_groups,
            device=dn_label_query.device,
        )
        dn_meta = dict(
            num_denoising_queries=int(max_num_target * 2 * num_groups),
            num_denoising_groups=num_groups,
        )
        return dn_label_query, dn_bbox_query, attn_mask, dn_meta

    def _num_phrases(self, sample) -> int:
        text = getattr(sample, "text", None)
        if isinstance(text, (list, tuple)) and len(text) > 0:
            return len(text)
        sent_ids = getattr(sample, "sent_ids", None)
        if sent_ids is not None and len(sent_ids) > 0:
            return len(sent_ids)
        if self.fallback_num_phrases is not None:
            return int(self.fallback_num_phrases)
        return self.num_classes

    def _sent_group_ids(self, sample) -> Tensor | None:
        sent_group_ids = getattr(sample, "sent_group_ids", None)
        if sent_group_ids is None:
            return None
        if isinstance(sent_group_ids, Tensor):
            return sent_group_ids
        return torch.as_tensor(sent_group_ids, dtype=torch.long)

    def _build_wrong_labels(
        self,
        gt_labels_list: Sequence[Tensor],
        num_phrase_list: Sequence[int],
        sent_group_ids_list: Sequence[Tensor | None],
    ) -> Tensor:
        wrong_labels = []
        for labels, num_phrases, sent_group_ids in zip(
            gt_labels_list,
            num_phrase_list,
            sent_group_ids_list,
        ):
            if len(labels) == 0:
                wrong_labels.append(labels)
                continue
            phrase_count = max(1, min(int(num_phrases), self.num_classes))
            wrong = self._different_group_labels(labels, phrase_count, sent_group_ids)
            if phrase_count == 1:
                wrong = (labels + 1) % self.num_classes
            wrong_labels.append(wrong)
        return torch.cat(wrong_labels)

    def _different_group_labels(
        self,
        labels: Tensor,
        phrase_count: int,
        sent_group_ids: Tensor | None,
    ) -> Tensor:
        if sent_group_ids is None or len(sent_group_ids) < phrase_count:
            return (labels + 1) % phrase_count

        sent_group_ids = sent_group_ids.to(device=labels.device)
        candidate_labels = torch.arange(phrase_count, device=labels.device)
        wrong = []
        for label in labels:
            label_idx = int(label)
            if label_idx >= phrase_count:
                wrong.append((label + 1) % phrase_count)
                continue
            group_id = sent_group_ids[label_idx]
            valid = candidate_labels[sent_group_ids[:phrase_count] != group_id]
            if len(valid) == 0:
                wrong.append((label + 1) % phrase_count)
                continue
            next_pos = torch.searchsorted(valid, label.clamp(max=phrase_count - 1))
            next_pos = next_pos % len(valid)
            wrong.append(valid[next_pos])
        return torch.stack(wrong).to(dtype=labels.dtype)

    def get_num_groups(self, max_num_target: int | None = None) -> int:
        if self.dynamic_dn_groups:
            assert max_num_target is not None
            if max_num_target == 0:
                num_groups = 1
            else:
                num_groups = self.num_dn_queries // max_num_target
        else:
            num_groups = self.num_groups
        return max(1, int(num_groups))

    def generate_dn_label_query(
        self,
        gt_labels: Tensor,
        wrong_labels: Tensor,
        num_groups: int,
    ) -> Tensor:
        labels_per_group = torch.cat([gt_labels, wrong_labels], dim=0)
        labels_expand = labels_per_group.repeat(num_groups)
        labels_expand = labels_expand.clamp(min=0, max=self.num_classes - 1)
        return self.label_embedding(labels_expand)

    def generate_dn_bbox_query(self, gt_bboxes: Tensor, num_groups: int) -> Tensor:
        assert self.box_noise_scale > 0
        device = gt_bboxes.device
        num_targets = len(gt_bboxes)
        gt_bboxes_expand = gt_bboxes.repeat(2 * num_groups, 1)

        positive_idx = torch.arange(num_targets, dtype=torch.long, device=device)
        positive_idx = positive_idx.unsqueeze(0).repeat(num_groups, 1)
        positive_idx += 2 * num_targets * torch.arange(
            num_groups, dtype=torch.long, device=device
        )[:, None]
        positive_idx = positive_idx.flatten()
        negative_idx = positive_idx + num_targets

        rand_sign = torch.randint_like(
            gt_bboxes_expand, low=0, high=2, dtype=torch.float32
        ) * 2.0 - 1.0
        rand_part = torch.rand_like(gt_bboxes_expand)
        rand_part *= rand_sign

        scale = gt_bboxes_expand.new_full((len(gt_bboxes_expand), 1), self.box_noise_scale)
        scale[negative_idx] = self.same_region_box_noise_scale
        bboxes_whwh = _bbox_xyxy_to_cxcywh(gt_bboxes_expand)[:, 2:].repeat(1, 2)
        noisy_bboxes_expand = gt_bboxes_expand + rand_part * bboxes_whwh * scale / 2
        noisy_bboxes_expand = noisy_bboxes_expand.clamp(min=0.0, max=1.0)
        noisy_bboxes_expand = _bbox_xyxy_to_cxcywh(noisy_bboxes_expand)
        return _inverse_sigmoid(noisy_bboxes_expand)

    def collate_dn_queries(
        self,
        input_label_query: Tensor,
        input_bbox_query: Tensor,
        batch_idx: Tensor,
        batch_size: int,
        num_groups: int,
    ) -> tuple[Tensor, Tensor]:
        device = input_label_query.device
        num_target_list = [
            torch.sum(batch_idx == batch_id) for batch_id in range(batch_size)
        ]
        max_num_target = max(num_target_list)
        num_denoising_queries = int(max_num_target * 2 * num_groups)

        map_query_index = torch.cat(
            [torch.arange(num_target, device=device) for num_target in num_target_list]
        )
        map_query_index = torch.cat(
            [
                map_query_index + max_num_target * group_id
                for group_id in range(2 * num_groups)
            ]
        ).long()
        batch_idx_expand = batch_idx.repeat(2 * num_groups, 1).view(-1)
        mapper = (batch_idx_expand, map_query_index)

        batched_label_query = torch.zeros(
            batch_size, num_denoising_queries, self.embed_dims, device=device
        )
        batched_bbox_query = torch.zeros(
            batch_size, num_denoising_queries, 4, device=device
        )
        batched_label_query[mapper] = input_label_query
        batched_bbox_query[mapper] = input_bbox_query
        return batched_label_query, batched_bbox_query

    def generate_dn_mask(
        self,
        max_num_target: int,
        num_groups: int,
        device: torch.device | str,
    ) -> Tensor:
        num_denoising_queries = int(max_num_target * 2 * num_groups)
        num_queries_total = num_denoising_queries + self.num_matching_queries
        attn_mask = torch.zeros(
            num_queries_total,
            num_queries_total,
            device=device,
            dtype=torch.bool,
        )
        attn_mask[num_denoising_queries:, :num_denoising_queries] = True
        for group_id in range(num_groups):
            row_scope = slice(
                max_num_target * 2 * group_id,
                max_num_target * 2 * (group_id + 1),
            )
            left_scope = slice(max_num_target * 2 * group_id)
            right_scope = slice(
                max_num_target * 2 * (group_id + 1),
                num_denoising_queries,
            )
            attn_mask[row_scope, right_scope] = True
            attn_mask[row_scope, left_scope] = True
        return attn_mask


if RealModel is not None:

    @MODELS.register_module()
    class RealModelD3PromptWrapper(RealModel):
        """Real-Model training wrapper for D3 list prompts without adapters."""

        @staticmethod
        def _prompt_key(text_prompt):
            if isinstance(text_prompt, list):
                return tuple(text_prompt)
            if isinstance(text_prompt, tuple):
                return text_prompt
            return text_prompt

        def loss(self, batch_inputs: Tensor, batch_data_samples: SampleList):
            """Compute GroundingDINO losses for D3 list prompts.

            Real-Model inherits the stock GroundingDINO training path, which
            assumes hashable string prompts. D3 supplies one image-local list of
            phrases, so this mirrors the MMDetection loss path with a list-safe
            prompt comparison.
            """
            text_prompts = [data_samples.text for data_samples in batch_data_samples]
            gt_labels = [
                data_samples.gt_instances.labels for data_samples in batch_data_samples
            ]

            positive_maps = []
            new_text_prompts = []
            first_prompt_key = self._prompt_key(text_prompts[0])
            same_prompt = all(
                self._prompt_key(prompt) == first_prompt_key for prompt in text_prompts
            )

            if same_prompt:
                tokenized, caption_string, tokens_positive, _ = (
                    self.get_tokens_and_prompts(text_prompts[0], True)
                )
                new_text_prompts = [caption_string] * len(batch_inputs)
                for gt_label in gt_labels:
                    new_tokens_positive = [
                        tokens_positive[int(label)] for label in gt_label
                    ]
                    _, positive_map = self.get_positive_map(
                        tokenized, new_tokens_positive
                    )
                    positive_maps.append(positive_map)
            else:
                for text_prompt, gt_label in zip(text_prompts, gt_labels):
                    tokenized, caption_string, tokens_positive, _ = (
                        self.get_tokens_and_prompts(text_prompt, True)
                    )
                    new_tokens_positive = [
                        tokens_positive[int(label)] for label in gt_label
                    ]
                    _, positive_map = self.get_positive_map(
                        tokenized, new_tokens_positive
                    )
                    positive_maps.append(positive_map)
                    new_text_prompts.append(caption_string)

            text_dict = self.language_model(new_text_prompts)
            if self.text_feat_map is not None:
                text_dict["embedded"] = self.text_feat_map(text_dict["embedded"])

            for i, data_samples in enumerate(batch_data_samples):
                positive_map = positive_maps[i].to(batch_inputs.device).bool().float()
                text_token_mask = text_dict["text_token_mask"][i]
                data_samples.gt_instances.positive_maps = positive_map
                data_samples.gt_instances.text_token_mask = (
                    text_token_mask.unsqueeze(0).repeat(len(positive_map), 1)
                )

            visual_features = self.extract_feat(batch_inputs)
            head_inputs_dict = self.forward_transformer(
                visual_features, text_dict, batch_data_samples
            )
            return self.bbox_head.loss(
                **head_inputs_dict, batch_data_samples=batch_data_samples
            )

    @MODELS.register_module()
    class RealModelTextQueryAdapter(RealModelD3PromptWrapper):
        """Real-Model with a small description-conditioned text/query adapter."""

        def __init__(
            self, *args, text_query_adapter: Optional[dict] = None, **kwargs
        ) -> None:
            self.text_query_adapter_cfg = dict(text_query_adapter or {})
            super().__init__(*args, **kwargs)

        def _init_layers(self) -> None:
            super()._init_layers()
            cfg = {
                "embed_dims": self.embed_dims,
                **self.text_query_adapter_cfg,
            }
            self.text_query_adapter = ResidualTextQueryAdapter(**cfg)

        def forward_encoder(
            self,
            feat: Tensor,
            feat_mask: Tensor,
            feat_pos: Tensor,
            spatial_shapes: Tensor,
            level_start_index: Tensor,
            valid_ratios: Tensor,
            text_dict: dict,
        ) -> dict:
            text_dict = dict(text_dict)
            text_dict["embedded"] = self.text_query_adapter(text_dict["embedded"])
            return super().forward_encoder(
                feat=feat,
                feat_mask=feat_mask,
                feat_pos=feat_pos,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                valid_ratios=valid_ratios,
                text_dict=text_dict,
            )

    @MODELS.register_module()
    class RealModelTextQueryAdapterNegDN(RealModelTextQueryAdapter):
        """Text adapter with in-image wrong-phrase denoising negatives."""

        def __init__(self, *args, negative_dn: Optional[dict] = None, **kwargs) -> None:
            self.negative_dn_cfg = dict(negative_dn or {})
            super().__init__(*args, **kwargs)

        def _init_layers(self) -> None:
            super()._init_layers()
            self.dn_query_generator = InImageWrongPhraseCdnQueryGenerator(
                self.dn_query_generator,
                **self.negative_dn_cfg,
            )

else:  # pragma: no cover - keeps local imports usable without Real-LOD installed.
    RealModelD3PromptWrapper = None
    RealModelTextQueryAdapter = None
    RealModelTextQueryAdapterNegDN = None
