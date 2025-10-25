from __future__ import annotations

import math
import logging
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
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
        *,
        log_interval: int = 200,
    ) -> None:
        super().__init__()
        assert num_prototypes > 0 and hidden_dim > 0
        self.lambda_orth = 0.05
        self.lambda_div = 0.01

        self.num_prototypes = num_prototypes
        self.tau = max(float(tau), 1e-6)

        self.key_proj = nn.Linear(dim, hidden_dim)
        self.value_proj = nn.Linear(dim, dim)
        self.prototype_queries = nn.Parameter(torch.empty(num_prototypes, hidden_dim))
        self.dropout = nn.Dropout(dropout)

        self._reset_parameters()
        self.register_buffer("_eye_buffer", torch.eye(num_prototypes), persistent=False)
        self._logger = logging.getLogger("lami_dino.tpa")
        self.log_interval = log_interval
        self._step = 0
        self.last_loss_terms: Dict[str, float] = {}
        self._last_attention: Optional[torch.Tensor] = None
        self._last_prototypes: Optional[torch.Tensor] = None
        self.last_attention: Optional[torch.Tensor] = None
        self.last_prototypes: Optional[torch.Tensor] = None
        self.last_monitor_terms: Dict[str, float] = {}

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
        detached_attn = attn.detach()
        detached_proto = prototypes.detach()
        self._last_attention = detached_attn
        self._last_prototypes = detached_proto
        self.last_attention = detached_attn
        self.last_prototypes = detached_proto

        # === Step 2: Optional APR loss ===
        apr_loss = None
        if with_loss:
            apr_loss = self.compute_apr_loss(prototypes, attn)
            apr_value = apr_loss.detach()
        else:
            apr_value = self._update_metrics_no_grad(prototypes.detach(), attn.detach())

        self._maybe_log(apr_value)
        return prototypes, apr_loss

    def compute_apr_loss(self, prototypes: torch.Tensor, attn: torch.Tensor) -> torch.Tensor:
        """
        Compute Adaptive Prototype Regularization (APR) loss
        Includes orthogonal and diversity regularization.
        """
        loss_orth = self._orthogonality_term(prototypes)
        loss_div = self._diversity_term(attn)
        apr_loss = self.lambda_orth * loss_orth + self.lambda_div * loss_div
        self._store_loss_terms(loss_orth.detach(), loss_div.detach(), apr_loss.detach())
        return apr_loss

    def _orthogonality_term(self, prototypes: torch.Tensor) -> torch.Tensor:
        P_norm = F.normalize(prototypes, dim=-1)
        gram = torch.einsum("ckd,cmd->ckm", P_norm, P_norm)
        K = gram.size(-1)
        I = self._eye_buffer[:K, :K]
        if I.device != prototypes.device:
            I = I.to(prototypes.device)
        return ((gram - I) ** 2).mean()

    def _diversity_term(self, attn: torch.Tensor) -> torch.Tensor:
        p = attn.clamp(min=1e-6, max=1.0)
        entropy = -(p * torch.log(p)).sum(dim=-1) / math.log(p.size(-1))
        return entropy.mean()

    def _store_loss_terms(self, loss_orth: torch.Tensor, loss_div: torch.Tensor, apr_loss: torch.Tensor) -> None:
        self.last_loss_terms = {
            "loss_orth": float(loss_orth.item()),
            "loss_div": float(loss_div.item()),
            "loss_apr": float(apr_loss.item()),
        }

    def _update_metrics_no_grad(self, prototypes: torch.Tensor, attn: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            loss_orth = self._orthogonality_term(prototypes)
            loss_div = self._diversity_term(attn)
            apr_loss = self.lambda_orth * loss_orth + self.lambda_div * loss_div
        self._store_loss_terms(loss_orth, loss_div, apr_loss)
        return apr_loss.detach()

    def _maybe_log(self, apr_value: torch.Tensor) -> None:
        if self._logger is None:
            return
        if dist.is_available() and dist.is_initialized() and dist.get_rank() != 0:
            return
        loss_orth = self.last_loss_terms.get("loss_orth")
        loss_div = self.last_loss_terms.get("loss_div")
        apr_scalar = self.last_loss_terms.get("loss_apr")
        if loss_orth is None or loss_div is None:
            return
        if isinstance(apr_value, torch.Tensor):
            apr_scalar = float(apr_value.detach().item())
        if self.training:
            self._step += 1
            if self._step % self.log_interval != 0:
                return
            display_step = self._step
        else:
            display_step = self._step
        msg = f"[TPA] step={display_step} orth={loss_orth:.4f} usage={loss_div:.4f}"
        if apr_scalar is not None:
            msg += f" apr={apr_scalar:.4f}"
        monitor = monitor_prototype_metrics(self._last_prototypes, self._last_attention, display_step, prefix="[TPA]")
        if monitor:
            self.last_monitor_terms.update(monitor)
            msg += f" | orth_norm={monitor['orthogonality']:.4f} usage_entropy={monitor['usage_entropy']:.4f}"
        logging.getLogger("detectron2").info(msg)
        self._logger.info(msg)


@torch.no_grad()
def compute_prototype_orthogonality(prototypes: Optional[torch.Tensor]) -> float:
    if prototypes is None:
        return float("nan")
    P = F.normalize(prototypes, dim=-1)
    gram = torch.einsum("ckd,cmd->ckm", P, P)
    I = torch.eye(P.size(1), device=P.device)
    return ((gram - I) ** 2).mean().item()


@torch.no_grad()
def compute_usage_entropy(attn: Optional[torch.Tensor]) -> float:
    if attn is None:
        return float("nan")
    usage = attn.mean(dim=-1)
    p = usage / (usage.sum(dim=-1, keepdim=True) + 1e-8)
    entropy = -(p * (p + 1e-8).log()).sum(dim=-1) / math.log(p.size(-1))
    return entropy.mean().item()


def monitor_prototype_metrics(
    prototypes: Optional[torch.Tensor],
    attn: Optional[torch.Tensor],
    step: int = 0,
    log_interval: int = 200,
    prefix: str = "[Monitor]",
) -> Dict[str, float]:
    orth_score = compute_prototype_orthogonality(prototypes)
    usage_entropy = compute_usage_entropy(attn)
    if step % log_interval == 0:
        print(
            f"{prefix} step={step:06d} | orth={orth_score:.4f} | usage_entropy={usage_entropy:.4f}"
        )
    return {"orthogonality": orth_score, "usage_entropy": usage_entropy}


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
