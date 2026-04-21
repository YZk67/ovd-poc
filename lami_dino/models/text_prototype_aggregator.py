from __future__ import annotations
import math
import logging
from typing import Dict, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# 假设 total_iters = cfg.train.max_iter = 85200
warmup_ratio = 0.05  # 5% 标准档
warmup_steps = int(85200 * warmup_ratio)  # ≈ 4300


def _is_main_process() -> bool:
    try:
        import torch.distributed as dist
        return not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0
    except Exception:
        return True


class TextPrototypeAggregator(nn.Module):
    """
    Learnable Semantic Aggregator with Adaptive Prototype Regularization (APR)
    Enhanced version (warmup + stronger regularization + logging)
    """

    def __init__(
        self,
        dim: int,
        num_prototypes: int = 5,       # ↑ from 4 → 5
        hidden_dim: int = 256,
        dropout: float = 0.05,         # ↓ from 0.1 → 0.05
        tau: float = 0.07,             # ↓ from 0.1 → 0.07 (sharper attention)
        *,
        lambda_orth: float = 0.10,     # ↑ from 0.08 → 0.10
        lambda_div: float = 0.03,      # ↑ from 0.02 → 0.03
        apr_on_base_only: bool = False,
        novel_anchor_weight: float = 0.0,
        warmup_steps: int = warmup_steps,      # new: smooth ramp-up
        log_interval: int = 200,
    ) -> None:
        super().__init__()

        assert num_prototypes > 0 and hidden_dim > 0

        self.num_prototypes = num_prototypes
        self.tau = max(float(tau), 1e-6)
        self.lambda_orth_base = float(lambda_orth)
        self.lambda_div_base = float(lambda_div)
        self.apr_on_base_only = bool(apr_on_base_only)
        self.lambda_novel_anchor = float(novel_anchor_weight)
        self.warmup_steps = int(warmup_steps)

        # projections
        self.key_proj = nn.Linear(dim, hidden_dim)
        self.value_proj = nn.Linear(dim, dim)
        self.prototype_queries = nn.Parameter(torch.empty(num_prototypes, hidden_dim))
        self.dropout = nn.Dropout(dropout)

        # buffers
        self.register_buffer("_eye_buffer", torch.eye(num_prototypes), persistent=False)
        self.register_buffer("base_class_mask", torch.empty(0, dtype=torch.bool), persistent=False)
        self.register_buffer("novel_class_mask", torch.empty(0, dtype=torch.bool), persistent=False)
        self._logger = logging.getLogger("lami_dino.tpa")
        self.log_interval = int(log_interval)
        self._step = 0

        # caches
        self.last_loss_terms: Dict[str, float] = {}
        self.last_monitor_terms: Dict[str, float] = {}
        self._last_logits: Optional[torch.Tensor] = None
        self._last_prototypes: Optional[torch.Tensor] = None

        self._reset_parameters()

    # === parameter init ===
    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.key_proj.weight)
        nn.init.constant_(self.key_proj.bias, 0.0)
        nn.init.xavier_uniform_(self.value_proj.weight)
        nn.init.constant_(self.value_proj.bias, 0.0)
        nn.init.xavier_uniform_(self.prototype_queries)

    # === lambda warmup ===
    def _effective_lambdas(self) -> Tuple[float, float]:
        """
        Compute step-dependent regularization strengths (cosine warm-up).
        Smoothly ramps λ_orth and λ_div from 0 → base within warmup_steps.
        """
        if self.warmup_steps <= 0:
            return self.lambda_orth_base, self.lambda_div_base

        progress = min(1.0, (self._step + 1) / float(self.warmup_steps))
        # cosine warm-up: 0 → 1
        factor = 0.5 * (1.0 - math.cos(math.pi * progress))
        lam_orth = self.lambda_orth_base * factor
        lam_div = self.lambda_div_base * factor
        return lam_orth, lam_div

    def set_class_splits(
        self,
        base_mask: Optional[torch.Tensor] = None,
        novel_mask: Optional[torch.Tensor] = None,
    ) -> None:
        if base_mask is None or novel_mask is None:
            self.base_class_mask = torch.empty(0, dtype=torch.bool, device=self.prototype_queries.device)
            self.novel_class_mask = torch.empty(0, dtype=torch.bool, device=self.prototype_queries.device)
            return
        self.base_class_mask = torch.as_tensor(base_mask, dtype=torch.bool, device=self.prototype_queries.device).clone()
        self.novel_class_mask = torch.as_tensor(novel_mask, dtype=torch.bool, device=self.prototype_queries.device).clone()

    def _resolve_class_mask(
        self,
        mask: torch.Tensor,
        class_inds: Optional[torch.Tensor],
        *,
        num_classes: int,
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        if mask.numel() == 0:
            return None
        resolved = mask
        if class_inds is not None:
            class_inds = class_inds.to(device=resolved.device, dtype=torch.long)
            resolved = resolved.index_select(0, class_inds.reshape(-1))
        if resolved.numel() != num_classes:
            return None
        return resolved.to(device=device)

    # === forward ===
    def forward(
        self,
        text_feats: torch.Tensor,
        with_loss: bool = True,
        class_inds: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        assert text_feats.ndim == 3, f"Expected [C,N,D], got {text_feats.shape}"
        C, N, D = text_feats.shape

        keys = self.key_proj(text_feats)
        values = self.value_proj(text_feats)

        logits = torch.einsum("kh,cnh->ckn", self.prototype_queries, keys)
        logits = logits / math.sqrt(self.key_proj.out_features)
        attn = F.softmax(logits / self.tau, dim=-1)

        prototypes_clean = torch.einsum("ckn,cnd->ckd", attn, values)
        prototypes = self.dropout(prototypes_clean)

        self._last_logits = logits.detach()
        self._last_prototypes = prototypes_clean.detach()

        apr_loss = None
        if with_loss:
            apr_loss = self.compute_apr_loss(prototypes_clean, logits, text_feats, class_inds=class_inds)
            apr_value = apr_loss.detach()
        else:
            apr_value = self._update_metrics_no_grad(
                prototypes_clean.detach(),
                logits.detach(),
                text_feats.detach(),
                class_inds=class_inds,
            )

        self._maybe_log(apr_value)
        return prototypes, apr_loss

    # === APR loss ===
    def compute_apr_loss(
        self,
        prototypes: torch.Tensor,
        logits: torch.Tensor,
        text_feats: torch.Tensor,
        *,
        class_inds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        scope_prototypes = prototypes
        scope_logits = logits
        if self.apr_on_base_only:
            base_mask = self._resolve_class_mask(
                self.base_class_mask,
                class_inds,
                num_classes=prototypes.shape[0],
                device=prototypes.device,
            )
            if base_mask is not None:
                scope_prototypes = prototypes[base_mask]
                scope_logits = logits[base_mask]

        loss_orth = self._orthogonality_term(scope_prototypes)
        loss_div = self._diversity_term(scope_logits)
        loss_novel_anchor = self._novel_anchor_term(prototypes, text_feats, class_inds=class_inds)
        lam_orth, lam_div = self._effective_lambdas()
        apr_loss = lam_orth * loss_orth + lam_div * loss_div + self.lambda_novel_anchor * loss_novel_anchor
        self._store_loss_terms(loss_orth, loss_div, loss_novel_anchor, apr_loss, lam_orth, lam_div)
        return apr_loss

    # === orthogonality term ===
    def _orthogonality_term(self, prototypes: torch.Tensor) -> torch.Tensor:
        if prototypes.numel() == 0 or prototypes.shape[0] == 0 or prototypes.shape[1] <= 1:
            return prototypes.new_zeros(())
        P = F.normalize(prototypes, dim=-1)
        G = torch.einsum("ckd,cmd->ckm", P, P)
        K = G.size(-1)
        I = self._eye_buffer[:K, :K].to(G.device)
        off_mask = (1.0 - torch.eye(K, device=G.device))
        return ((G - I) ** 2 * off_mask).sum(dim=(-2, -1)).mean() / (K * K - K)

    # === diversity term ===
    def _diversity_term(self, logits: torch.Tensor) -> torch.Tensor:
        if logits.numel() == 0 or logits.shape[0] == 0 or logits.shape[1] <= 1:
            return logits.new_zeros(())
        C, K, N = logits.shape
        w = torch.softmax(logits, dim=1)
        votes = w.sum(dim=-1)
        p = votes / (votes.sum(dim=1, keepdim=True) + 1e-8)
        entropy = -(p * (p.clamp_min(1e-8)).log()).sum(dim=1) / math.log(K)
        return entropy.mean()

    def _novel_anchor_term(
        self,
        prototypes: torch.Tensor,
        text_feats: torch.Tensor,
        *,
        class_inds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.lambda_novel_anchor <= 0:
            return prototypes.new_zeros(())
        novel_mask = self._resolve_class_mask(
            self.novel_class_mask,
            class_inds,
            num_classes=prototypes.shape[0],
            device=prototypes.device,
        )
        if novel_mask is None or not bool(novel_mask.any()):
            return prototypes.new_zeros(())
        novel_proto = prototypes[novel_mask]
        novel_text = text_feats[novel_mask].to(device=prototypes.device, dtype=prototypes.dtype)
        proto_center = F.normalize(novel_proto.mean(dim=1), dim=-1)
        raw_center = F.normalize(novel_text.mean(dim=1), dim=-1)
        return F.mse_loss(proto_center, raw_center)

    # === bookkeeping ===
    def _store_loss_terms(self, loss_orth, loss_div, loss_novel_anchor, apr_loss, lam_orth, lam_div):
        self.last_loss_terms = {
            "loss_orth": float(loss_orth.item()),
            "loss_div": float(loss_div.item()),
            "loss_novel_anchor": float(loss_novel_anchor.item()),
            "loss_apr": float(apr_loss.item()),
            "lambda_orth": float(lam_orth),
            "lambda_div": float(lam_div),
            "lambda_novel_anchor": float(self.lambda_novel_anchor),
        }

    def _update_metrics_no_grad(self, prototypes, logits, text_feats, *, class_inds=None):
        with torch.no_grad():
            scope_prototypes = prototypes
            scope_logits = logits
            if self.apr_on_base_only:
                base_mask = self._resolve_class_mask(
                    self.base_class_mask,
                    class_inds,
                    num_classes=prototypes.shape[0],
                    device=prototypes.device,
                )
                if base_mask is not None:
                    scope_prototypes = prototypes[base_mask]
                    scope_logits = logits[base_mask]
            loss_orth = self._orthogonality_term(scope_prototypes)
            loss_div = self._diversity_term(scope_logits)
            loss_novel_anchor = self._novel_anchor_term(prototypes, text_feats, class_inds=class_inds)
            lam_orth, lam_div = self._effective_lambdas()
            apr_loss = lam_orth * loss_orth + lam_div * loss_div + self.lambda_novel_anchor * loss_novel_anchor
        self._store_loss_terms(loss_orth, loss_div, loss_novel_anchor, apr_loss, lam_orth, lam_div)
        return apr_loss.detach()

    # === logging ===
    def _maybe_log(self, apr_value):
        self._step += 1
        if (not self._logger) or (self.training and self._step % self.log_interval != 0):
            return
        if not _is_main_process():
            return
        monitor = monitor_prototype_metrics(self._last_prototypes, self._last_logits, step=self._step)
        if monitor:
            self.last_monitor_terms.update(monitor)
        msg = (
            f"[TPA] step={self._step:06d} "
            f"orth_off={monitor.get('orth_off_mse', 0):.4f} "
            f"diag_mse={monitor.get('diag_mse', 0):.4f} "
            f"usage_entropy={monitor.get('usage_entropy', 0):.4f} "
            f"apr={float(apr_value):.5f} "
            f"(λ_orth={self.last_loss_terms.get('lambda_orth', 0):.3f}, "
            f"λ_div={self.last_loss_terms.get('lambda_div', 0):.3f})"
        )
        self._logger.info(msg)

    def get_monitor_dict(self) -> Dict[str, float]:
        out = dict(self.last_loss_terms)
        if not self.last_monitor_terms and self._last_prototypes is not None:
            with torch.no_grad():
                off_mse, diag_mse = compute_prototype_orthogonality(self._last_prototypes)
                usage = compute_usage_entropy(self._last_logits)
            self.last_monitor_terms = {
                "orth_off_mse": off_mse,
                "diag_mse": diag_mse,
                "usage_entropy": usage,
            }
        out.update(self.last_monitor_terms)
        return out


@torch.no_grad()
def compute_prototype_orthogonality(prototypes):
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
def compute_usage_entropy(logits):
    if logits is None:
        return float("nan")
    C, K, N = logits.shape
    w = torch.softmax(logits, dim=1)
    votes = w.sum(dim=-1)
    p = votes / (votes.sum(dim=1, keepdim=True) + 1e-8)
    entropy = -(p * (p.clamp_min(1e-8)).log()).sum(dim=1) / math.log(K)
    return float(entropy.mean().item())


def monitor_prototype_metrics(prototypes, logits, step=0, prefix="[TPA]"):
    off_mse, diag_mse = compute_prototype_orthogonality(prototypes)
    usage_entropy = compute_usage_entropy(logits)
    if step % 200 == 0 and _is_main_process():
        print(f"{prefix} step={step:06d} | orth_off={off_mse:.4f} | diag_mse={diag_mse:.4f} | usage_entropy={usage_entropy:.4f}")
    return {"orth_off_mse": off_mse, "diag_mse": diag_mse, "usage_entropy": usage_entropy}


class TextPrototypeBank(nn.Module):
    """
    Wrapper for managing and caching text prototypes
    """

    def __init__(
        self,
        embedding_path: str,
        aggregator: Optional[TextPrototypeAggregator] = None,
        *,
        num_prototypes: int = 6,
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
