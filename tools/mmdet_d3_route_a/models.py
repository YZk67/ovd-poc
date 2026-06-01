"""GroundingDINO adapter modules for D3 Route-A experiments."""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor, nn

from mmdet.models.detectors.grounding_dino import GroundingDINO
from mmdet.registry import MODELS


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

