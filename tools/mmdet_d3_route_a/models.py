"""GroundingDINO adapter modules for D3 Route-A experiments."""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor, nn

from mmdet.models.detectors.grounding_dino import GroundingDINO
from mmdet.registry import MODELS
from mmdet.structures import SampleList


class ResidualTextQueryAdapter(nn.Module):
    """Zero-initialized residual MLP for text features used by query selection."""

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


@MODELS.register_module()
class GroundingDINOTextQueryAdapter(GroundingDINO):
    """GroundingDINO with a small description-conditioned text/query adapter.

    The adapter is applied after BERT features are projected to detector hidden
    size and before the multimodal encoder, so it affects language-guided query
    selection and the cross-modal decoder without changing the visual backbone.
    """

    def __init__(self, *args, text_query_adapter: Optional[dict] = None, **kwargs) -> None:
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

    @staticmethod
    def _prompt_key(text_prompt):
        if isinstance(text_prompt, list):
            return tuple(text_prompt)
        if isinstance(text_prompt, tuple):
            return text_prompt
        return text_prompt

    def loss(self, batch_inputs: Tensor, batch_data_samples: SampleList):
        """Compute losses for D3 list prompts.

        MMDetection's stock GroundingDINO loss calls ``set(text_prompts)`` to
        detect shared prompts in a batch. D3 supplies each image prompt as a
        list of descriptions, which is unhashable, so we reproduce the stock
        loss path with a list-safe prompt key.
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
            tokenized, caption_string, tokens_positive, _ = self.get_tokens_and_prompts(
                text_prompts[0], True
            )
            new_text_prompts = [caption_string] * len(batch_inputs)
            for gt_label in gt_labels:
                new_tokens_positive = [
                    tokens_positive[int(label)] for label in gt_label
                ]
                _, positive_map = self.get_positive_map(tokenized, new_tokens_positive)
                positive_maps.append(positive_map)
        else:
            for text_prompt, gt_label in zip(text_prompts, gt_labels):
                tokenized, caption_string, tokens_positive, _ = self.get_tokens_and_prompts(
                    text_prompt, True
                )
                new_tokens_positive = [
                    tokens_positive[int(label)] for label in gt_label
                ]
                _, positive_map = self.get_positive_map(tokenized, new_tokens_positive)
                positive_maps.append(positive_map)
                new_text_prompts.append(caption_string)

        text_dict = self.language_model(new_text_prompts)
        if self.text_feat_map is not None:
            text_dict["embedded"] = self.text_feat_map(text_dict["embedded"])

        for i, data_samples in enumerate(batch_data_samples):
            positive_map = positive_maps[i].to(batch_inputs.device).bool().float()
            text_token_mask = text_dict["text_token_mask"][i]
            data_samples.gt_instances.positive_maps = positive_map
            data_samples.gt_instances.text_token_mask = text_token_mask.unsqueeze(0).repeat(
                len(positive_map), 1
            )

        visual_features = self.extract_feat(batch_inputs)
        head_inputs_dict = self.forward_transformer(
            visual_features, text_dict, batch_data_samples
        )
        return self.bbox_head.loss(
            **head_inputs_dict, batch_data_samples=batch_data_samples
        )
