from __future__ import annotations

import math
import logging
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _is_main_process() -> bool:
    try:
        import torch.distributed as dist
        return not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0
    except Exception:
        return True


class TextPrototypeAggregator(nn.Module):
    """
    Learnable Semantic Aggregator with Adaptive Prototype Regularization (APR)
    - Attention聚合: [C, N, D] -> [C, K, D]
    - APR: orthogonality + diversity (based on logits softmax across K)
    - 健康的监控指标: orth(off-diag MSE), usage entropy
    """

    def __init__(
        self,
        dim: int,
        num_prototypes: int = 4,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        tau: float = 0.1,
        *,
        lambda_orth: float = 0.08,
        lambda_div: float = 0.02,
        log_interval: int = 200,
        entropy_mode: str = "soft",   # "soft" 或 "hard"
        warmup_steps: int = 0,        # >0 时启用线性warmup
    ) -> None:
        super().__init__()
        assert num_prototypes > 0 and hidden_dim > 0
        self.num_prototypes = num_prototypes
        self.tau = max(float(tau), 1e-6)

        self.lambda_orth_base = float(lambda_orth)
        self.lambda_div_base = float(lambda_div)
        self.warmup_steps = int(warmup_steps)

        self.key_proj = nn.Linear(dim, hidden_dim)
        self.value_proj = nn.Linear(dim, dim)
        self.prototype_queries = nn.Parameter(torch.empty(num_prototypes, hidden_dim))
        self.dropout = nn.Dropout(dropout)

        self.register_buffer("_eye_buffer", torch.eye(num_prototypes, dtype=torch.float32), persistent=False)
        self._logger = logging.getLogger("lami_dino.tpa")
        self.log_interval = int(log_interval)
        self.entropy_mode = entropy_mode
        self._step = 0

        self.last_loss_terms: Dict[str, float] = {}
        self._last_attention: Optional[torch.Tensor] = None
        self._last_logits: Optional[torch.Tensor] = None
        self._last_prototypes: Optional[torch.Tensor] = None
        self.last_monitor_terms: Dict[str, float] = {}

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.key_proj.weight)
        nn.init.constant_(self.key_proj.bias, 0.0)
        nn.init.xavier_uniform_(self.value_proj.weight)
        nn.init.constant_(self.value_proj.bias, 0.0)
        nn.init.xavier_uniform_(self.prototype_queries)

    # ------------------------ 前向 ------------------------
    def forward(self, text_feats: torch.Tensor, with_loss: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            text_feats: [C, N, D]
        Returns:
            prototypes: [C, K, D]
            apr_loss: scalar tensor or None
        """
        assert text_feats.ndim == 3, f"Expected [C,N,D], got {text_feats.shape}"
        C, N, D = text_feats.shape

        keys = self.key_proj(text_feats)      # [C, N, H]
        values = self.value_proj(text_feats)  # [C, N, D]

        logits = torch.einsum("kh,cnh->ckn", self.prototype_queries, keys)  # [C,K,N]
        logits = logits / math.sqrt(self.key_proj.out_features)
        attn = F.softmax(logits / self.tau, dim=-1)  # 每个prototype在短语上的分布

        prototypes_clean = torch.einsum("ckn,cnd->ckd", attn, values)  # [C,K,D]
        prototypes = self.dropout(prototypes_clean)

        # 缓存
        self._last_attention = attn.detach()
        self._last_logits = logits.detach()
        self._last_prototypes = prototypes_clean.detach()

        apr_loss = None
        if with_loss:
            apr_loss = self.compute_apr_loss(prototypes_clean, logits)
            apr_value = apr_loss.detach()
        else:
            apr_value = self._update_metrics_no_grad(prototypes_clean.detach(), logits.detach())

        self._maybe_log(apr_value)
        return prototypes, apr_loss

    # ------------------------ APR ------------------------
    def _effective_lambdas(self) -> Tuple[float, float]:
        """线性warmup（若启用）"""
        if self.warmup_steps <= 0:
            return self.lambda_orth_base, self.lambda_div_base
        factor = min(1.0, float(self._step + 1) / float(self.warmup_steps))
        return self.lambda_orth_base * factor, self.lambda_div_base * factor

    def compute_apr_loss(self, prototypes: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        loss_orth = self._orthogonality_term(prototypes)
        loss_div = self._diversity_term(logits, mode=self.entropy_mode)

        lam_orth, lam_div = self._effective_lambdas()
        apr_loss = lam_orth * loss_orth + lam_div * loss_div
        self._store_loss_terms(loss_orth.detach(), loss_div.detach(), apr_loss.detach(), lam_orth, lam_div)
        return apr_loss

    # ------------------------ 正则细项 ------------------------
    def _orthogonality_term(self, prototypes: torch.Tensor) -> torch.Tensor:
        P = F.normalize(prototypes, dim=-1)
        G = torch.einsum("ckd,cmd->ckm", P, P)
        K = G.size(-1)
        I = self._eye_buffer[:K, :K].to(G.device)
        off_mask = (1.0 - torch.eye(K, device=G.device))
        off = ((G - I) ** 2 * off_mask).sum(dim=(-2, -1)) / (K * K - K)
        loss_off = off.mean()
        return loss_off

    def _diversity_term(self, logits: torch.Tensor, mode: str = "soft") -> torch.Tensor:
        """
        新版: 基于logits, 在K维softmax, 衡量“短语分配给不同原型的多样性”
        """
        C, K, N = logits.shape
        if mode == "hard":
            winners = logits.argmax(dim=1)                 # [C, N]
            votes = torch.zeros(C, K, device=logits.device, dtype=logits.dtype)
            votes.scatter_add_(1, winners, torch.ones_like(winners, dtype=logits.dtype))
        else:
            w = torch.softmax(logits, dim=1)               # [C, K, N] 在K维softmax
            votes = w.sum(dim=-1)                          # [C, K]

        p = votes / (votes.sum(dim=1, keepdim=True) + 1e-8)
        entropy = -(p * (p.clamp_min(1e-8)).log()).sum(dim=1) / math.log(K)
        return entropy.mean()

    # ------------------------ 监控与日志 ------------------------
    def _store_loss_terms(
        self,
        loss_orth: torch.Tensor,
        loss_div: torch.Tensor,
        apr_loss: torch.Tensor,
        lam_orth: float,
        lam_div: float,
    ) -> None:
        self.last_loss_terms = {
            "loss_orth": float(loss_orth.item()),
            "loss_div": float(loss_div.item()),
            "loss_apr": float(apr_loss.item()),
            "lambda_orth": float(lam_orth),
            "lambda_div": float(lam_div),
        }

    def _update_metrics_no_grad(self, prototypes: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            loss_orth = self._orthogonality_term(prototypes)
            loss_div = self._diversity_term(logits, mode=self.entropy_mode)
            lam_orth, lam_div = self._effective_lambdas()
            apr_loss = lam_orth * loss_orth + lam_div * loss_div
        self._store_loss_terms(loss_orth, loss_div, apr_loss, lam_orth, lam_div)
        return apr_loss.detach()

    def _maybe_log(self, apr_value: torch.Tensor) -> None:
        self._step += 1
        if self._logger is None or (self.training and self._step % self.log_interval != 0):
            return
        if not _is_main_process():
            return

        monitor = monitor_prototype_metrics(
            self._last_prototypes, self._last_logits,
            step=self._step, log_interval=self.log_interval, prefix="[TPA]", entropy_mode=self.entropy_mode
        )
        if monitor:
            self.last_monitor_terms.update(monitor)

        msg = (
            f"[TPA] step={self._step:06d} "
            f"orth_off={monitor.get('orth_off_mse', float('nan')):.4f} "
            f"diag_mse={monitor.get('diag_mse', float('nan')):.4f} "
            f"usage_entropy={monitor.get('usage_entropy', float('nan')):.4f} "
            f"apr={float(apr_value):.5f} "
            f"(lam_orth={self.last_loss_terms.get('lambda_orth', 0):.3f}, "
            f"lam_div={self.last_loss_terms.get('lambda_div', 0):.3f})"
        )
        self._logger.info(msg)

    def get_monitor_dict(self) -> Dict[str, float]:
        return {**self.last_loss_terms, **self.last_monitor_terms}


# ======================= 度量函数 =======================
@torch.no_grad()
def compute_prototype_orthogonality(prototypes: Optional[torch.Tensor]) -> Tuple[float, float]:
    if prototypes is None:
        return float("nan"), float("nan")
    P = F.normalize(prototypes, dim=-1)
    G = torch.einsum("ckd,cmd->ckm", P, P)
    C, K, _ = G.shape
    I = torch.eye(K, device=G.device)
    diag_mse = (G.diagonal(dim1=-2, dim2=-1) - 1.0).pow(2).mean()
    off_mask = (1.0 - torch.eye(K, device=G.device))
    off_mse = ((G - I) ** 2 * off_mask).sum(dim=(-2, -1)) / (K * K - K)
    off_mse = off_mse.mean()
    return float(off_mse.item()), float(diag_mse.item())


@torch.no_grad()
def compute_usage_entropy(logits: Optional[torch.Tensor], mode: str = "soft") -> float:
    if logits is None:
        return float("nan")
    C, K, N = logits.shape
    if mode == "hard":
        winners = logits.argmax(dim=1)
        votes = torch.zeros(C, K, device=logits.device, dtype=logits.dtype)
        votes.scatter_add_(1, winners, torch.ones_like(winners, dtype=logits.dtype))
    else:
        w = torch.softmax(logits, dim=1)
        votes = w.sum(dim=-1)
    p = votes / (votes.sum(dim=1, keepdim=True) + 1e-8)
    entropy = -(p * (p.clamp_min(1e-8)).log()).sum(dim=1) / math.log(K)
    return float(entropy.mean().item())


def monitor_prototype_metrics(
    prototypes: Optional[torch.Tensor],
    logits: Optional[torch.Tensor],
    step: int = 0,
    log_interval: int = 200,
    prefix: str = "[Monitor]",
    *,
    entropy_mode: str = "soft",
) -> Dict[str, float]:
    off_mse, diag_mse = compute_prototype_orthogonality(prototypes)
    usage_entropy = compute_usage_entropy(logits, mode=entropy_mode)
    if step % log_interval == 0 and _is_main_process():
        print(f"{prefix} step={step:06d} | orth_off={off_mse:.4f} | diag_mse={diag_mse:.4f} | usage_entropy={usage_entropy:.4f}")
    return {
        "orthogonality": off_mse,
        "orth_off_mse": off_mse,
        "diag_mse": diag_mse,
        "usage_entropy": usage_entropy,
    }


# ======================= PrototypeBank =======================
class TextPrototypeBank(nn.Module):
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
            or self._cached_prototypes.device != text_feats.device
        ):
            with torch.no_grad():
                prototypes, apr_loss = self.aggregator(text_feats, with_loss=False)
            self._cached_prototypes = prototypes
            self._cached_apr_loss = apr_loss
            self._cached_step = step
        return self._cached_prototypes, self._cached_apr_loss
