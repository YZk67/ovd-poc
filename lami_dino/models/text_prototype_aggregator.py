from __future__ import annotations

import math
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class TextPrototypeAggregator(nn.Module):
    """
    Learnable semantic aggregator that maps per-class phrase embeddings
    (shape: [num_classes, num_phrases, dim]) to K prototypes per class.
    """

    def __init__(
        self,
        dim: int,
        num_prototypes: int = 4,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        tau: float = 0.1,
    ) -> None:
        super().__init__()
        if num_prototypes <= 0:
            raise ValueError(f"num_prototypes must be > 0, got {num_prototypes}")
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be > 0, got {hidden_dim}")

        self.num_prototypes = num_prototypes
        self.tau = max(float(tau), 1e-6)

        self.key_proj = nn.Linear(dim, hidden_dim)
        self.value_proj = nn.Linear(dim, dim)
        self.prototype_queries = nn.Parameter(torch.empty(num_prototypes, hidden_dim))
        self.dropout = nn.Dropout(dropout)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.key_proj.weight)
        nn.init.constant_(self.key_proj.bias, 0.0)
        nn.init.xavier_uniform_(self.value_proj.weight)
        nn.init.constant_(self.value_proj.bias, 0.0)
        nn.init.xavier_uniform_(self.prototype_queries)

    @property
    def hidden_dim(self) -> int:
        return self.key_proj.out_features

    def forward(self, text_feats: torch.Tensor) -> torch.Tensor:
        """
        Args:
            text_feats: Tensor with shape [num_classes, num_phrases, dim].

        Returns:
            Tensor with shape [num_classes, num_prototypes, dim].
        """
        if text_feats.ndim != 3:
            raise ValueError(f"text_feats must be [C, N, D], got shape {tuple(text_feats.shape)}")

        class_count, phrase_count, feat_dim = text_feats.shape
        if phrase_count == 0:
            raise ValueError("num_phrases should be > 0.")

        keys = self.key_proj(text_feats)  # [C, N, H]
        values = self.value_proj(text_feats)  # [C, N, D]

        logits = torch.einsum("kh,cnh->ckn", self.prototype_queries, keys)
        logits = logits / math.sqrt(self.hidden_dim)
        attn = F.softmax(logits / self.tau, dim=-1)  # [C, K, N]

        prototypes = torch.einsum("ckn,cnd->ckd", attn, values)  # [C, K, D]
        prototypes = self.dropout(prototypes)
        return prototypes


class TextPrototypeBank(nn.Module):
    """
    Manage raw phrase embeddings and expose TPA outputs on demand.

    Args:
        embedding_path: Path to numpy array containing text embeddings.
                        Shape expected to be [C, N, D] or [C, D].
        aggregator: Optional TextPrototypeAggregator. If None, a default
                    aggregator is created with ``num_prototypes``.
        num_prototypes: Number of semantic prototypes per class.
        dtype: Target dtype for stored text embeddings.
    """

    def __init__(
        self,
        embedding_path: str,
        aggregator: Optional[TextPrototypeAggregator] = None,
        *,
        num_prototypes: int = 4,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()

        raw = np.load(embedding_path)
        if raw.ndim == 2:
            raw = raw[:, None, :]
        if raw.ndim != 3:
            raise ValueError(
                f"Expected text embedding array with 2 or 3 dims, got shape {raw.shape}."
            )

        text_feats = torch.from_numpy(raw).to(dtype=dtype)
        self.register_buffer("text_feats", text_feats, persistent=False)

        self.num_classes, self.num_phrases, self.feat_dim = text_feats.shape
        self.num_prototypes = num_prototypes

        if aggregator is None:
            aggregator = TextPrototypeAggregator(
                dim=self.feat_dim,
                num_prototypes=num_prototypes,
            )
        self.aggregator = aggregator

        self._cached_step = -1
        self._cached_prototypes: Optional[torch.Tensor] = None

    def _ensure_device(self) -> torch.Tensor:
        """
        Make sure text embeddings live on the same device as the aggregator.
        """
        target_device = next(self.aggregator.parameters()).device
        if self.text_feats.device != target_device:
            self.text_feats = self.text_feats.to(device=target_device)
        return self.text_feats

    def reset_cache(self) -> None:
        self._cached_prototypes = None
        self._cached_step = -1

    def forward(self, step: int = -1, *, force_recompute: bool = False) -> torch.Tensor:
        """
        Returns prototypes with shape [num_classes, num_prototypes, feat_dim].

        During training we always recompute so gradients propagate through
        the aggregator. Under eval mode the output is cached per ``step``.
        """
        text_feats = self._ensure_device()

        if self.training or force_recompute:
            return self.aggregator(text_feats)

        if (
            self._cached_prototypes is None
            or force_recompute
            or step != self._cached_step
            or self._cached_prototypes.device != text_feats.device
        ):
            with torch.no_grad():
                prototypes = self.aggregator(text_feats)
            self._cached_prototypes = prototypes
            self._cached_step = step
        return self._cached_prototypes
