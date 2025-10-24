from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class TextPrototypeAggregator(nn.Module):
    """
    Learnable Semantic Aggregator with Adaptive Prototype Regularization (APR)
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
        assert num_prototypes > 0 and hidden_dim > 0

        self.num_prototypes = num_prototypes
        self.tau = max(float(tau), 1e-6)

        self.key_proj = nn.Linear(dim, hidden_dim)
        self.value_proj = nn.Linear(dim, dim)
        self.prototype_queries = nn.Parameter(torch.empty(num_prototypes, hidden_dim))
        self.dropout = nn.Dropout(dropout)

        self._reset_parameters()
        self.register_buffer("_eye_buffer", torch.eye(num_prototypes), persistent=False)

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.key_proj.weight)
        nn.init.constant_(self.key_proj.bias, 0.0)
        nn.init.xavier_uniform_(self.value_proj.weight)
        nn.init.constant_(self.value_proj.bias, 0.0)
        nn.init.xavier_uniform_(self.prototype_queries)

    def forward(self, text_feats: torch.Tensor, with_loss: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        text_feats: [C, N, D]
        Returns:
            prototypes: [C, K, D]
            apr_loss: scalar
        """
        assert text_feats.ndim == 3, f"Expected [C,N,D], got {text_feats.shape}"
        C, N, D = text_feats.shape

        keys = self.key_proj(text_feats)      # [C, N, H]
        values = self.value_proj(text_feats)  # [C, N, D]

        # === Step 1: Attention aggregation ===
        logits = torch.einsum("kh,cnh->ckn", self.prototype_queries, keys)  # [C,K,N]
        logits = logits / math.sqrt(self.key_proj.out_features)
        attn = F.softmax(logits / self.tau, dim=-1)  # [C,K,N]

        prototypes = torch.einsum("ckn,cnd->ckd", attn, values)  # [C,K,D]
        prototypes = self.dropout(prototypes)

        # === Step 2: Optional APR loss ===
        apr_loss = None
        if with_loss:
            apr_loss = self.compute_apr_loss(prototypes, attn)

        return prototypes, apr_loss

    def compute_apr_loss(self, prototypes: torch.Tensor, attn: torch.Tensor) -> torch.Tensor:
        """
        Compute Adaptive Prototype Regularization (APR) loss
        Includes orthogonal and diversity regularization.
        """
        C, K, D = prototypes.shape
        loss_orth, loss_div = 0.0, 0.0

        # === 1️⃣ Orthogonal Loss ===
        P_norm = F.normalize(prototypes, dim=-1)
        gram = torch.einsum("ckd,cmd->ckm", P_norm, P_norm)  # [C,K,K]
        I = self._eye_buffer[:K, :K].to(prototypes.device)
        loss_orth = ((gram - I) ** 2).mean()

        # === 2️⃣ Diversity (Normalized Entropy) Loss ===
        p = attn.clamp(min=1e-8, max=1.0)  # More stable clamping
        log_p = torch.log(p)
        entropy = -(p * log_p).sum(dim=-1) / math.log(p.size(-1))  # normalized [0,1]
        loss_div = entropy.mean()

        # === 3️⃣ Combine ===
        lambda_orth, lambda_div = 0.05, 0.01  # static version

        # TODO (Stage 3): curriculum-style adaptive weighting
        # lambda_orth = 0.05 * (1 - torch.exp(-0.001 * global_step))
        # lambda_div  = 0.01 * (1 - torch.exp(-0.001 * global_step))

        apr_loss = lambda_orth * loss_orth + lambda_div * loss_div

        # Optional: lightweight debug print (only occasionally)
        # if torch.rand(1) < 0.001:
        #     print(f"[APR] Orth={loss_orth.item():.4f} Div={loss_div.item():.4f}")

        return apr_loss


class TextPrototypeBank(nn.Module):
    """
    Wrapper for managing and caching text prototypes
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
        assert raw.ndim == 3, f"Expected [C,N,D], got {raw.shape}"

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
        self._cached_prototypes = None
        self._cached_apr_loss = None

    def forward(self, step: int = -1, *, force_recompute: bool = False):
        text_feats = self.text_feats.to(next(self.aggregator.parameters()).device)

        if self.training or force_recompute:
            prototypes, apr_loss = self.aggregator(text_feats)
            return prototypes, apr_loss

        if (
            self._cached_prototypes is None
            or force_recompute
            or step != self._cached_step
        ):
            with torch.no_grad():
                prototypes, apr_loss = self.aggregator(text_feats)
            self._cached_prototypes = prototypes
            self._cached_apr_loss = apr_loss
            self._cached_step = step

        return self._cached_prototypes, self._cached_apr_loss
