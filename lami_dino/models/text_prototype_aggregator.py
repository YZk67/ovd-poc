from __future__ import annotations
import math
import logging
from typing import Dict, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Default warmup configuration
# Note: warmup_steps should be set via config, not hardcoded
# Default warmup_ratio = 0.05 (5% of max_iter)
# For max_iter=92300, warmup_steps should be ~4615
warmup_ratio = 0.05  # 5% 标准档
# Default warmup_steps (will be overridden by config)
# This is a fallback value, should be set via tpa_warmup_steps in config
warmup_steps = int(85200 * warmup_ratio)  # ≈ 4260 (updated default for 85200 max_iter)


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
        warmup_steps: int = warmup_steps,      # new: smooth ramp-up
        log_interval: int = 200,
    ) -> None:
        super().__init__()

        assert num_prototypes > 0 and hidden_dim > 0

        self.num_prototypes = num_prototypes
        self.hidden_dim = hidden_dim  # Store hidden_dim for use in _reset_parameters
        self.tau = max(float(tau), 1e-6)
        self.lambda_orth_base = float(lambda_orth)
        self.lambda_div_base = float(lambda_div)
        self.warmup_steps = int(warmup_steps)

        # projections
        self.key_proj = nn.Linear(dim, hidden_dim)
        self.value_proj = nn.Linear(dim, dim)
        self.prototype_queries = nn.Parameter(torch.empty(num_prototypes, hidden_dim))
        self.dropout = nn.Dropout(dropout)

        # buffers
        self.register_buffer("_eye_buffer", torch.eye(num_prototypes), persistent=False)
        self.register_buffer("_step", torch.tensor(0, dtype=torch.long))  # Save step to checkpoint
        self._logger = logging.getLogger("lami_dino.tpa")
        self.log_interval = int(log_interval)

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
        
        # Improved initialization for prototype_queries to encourage diversity
        # Use orthogonal initialization to ensure initial queries are diverse
        if self.num_prototypes == 1:
            nn.init.xavier_uniform_(self.prototype_queries)
        else:
            # Use orthogonal initialization via QR decomposition
            # This ensures prototypes start with maximum diversity
            with torch.no_grad():
                # Initialize with random values
                init_vectors = torch.randn(self.num_prototypes, self.hidden_dim)
                # Use QR decomposition to get orthogonal vectors
                Q, R = torch.linalg.qr(init_vectors.t())
                # Use Q (orthogonal) and normalize
                orthogonal_queries = Q.t()[:self.num_prototypes]
                # Normalize and scale to small initial values
                self.prototype_queries.data = F.normalize(orthogonal_queries, p=2, dim=1) * 0.1

    # === lambda warmup ===
    def _effective_lambdas(self) -> Tuple[float, float]:
        """
        Compute step-dependent regularization strengths (cosine warm-up).
        Smoothly ramps λ_orth and λ_div from 0 → base within warmup_steps.
        """
        if self.warmup_steps <= 0:
            return self.lambda_orth_base, self.lambda_div_base

        step_value = int(self._step.item()) if isinstance(self._step, torch.Tensor) else self._step
        progress = min(1.0, (step_value + 1) / float(self.warmup_steps))
        # cosine warm-up: 0 → 1
        factor = 0.5 * (1.0 - math.cos(math.pi * progress))
        lam_orth = self.lambda_orth_base * factor
        lam_div = self.lambda_div_base * factor
        return lam_orth, lam_div


    # === forward ===
    def forward(self, text_feats: torch.Tensor, with_loss: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
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

        # Update _step before computing APR loss (so effective lambdas are correct)
        # Only update when computing loss (with_loss=True) to match iteration count
        if self.training and with_loss:
            self._step += 1
            # Direct orthogonalization: if prototypes are too similar, force orthogonality
            # This is necessary because gradient-based methods alone are not sufficient
            # We use it conservatively: only when really needed (high similarity) and infrequently
            step_value = int(self._step.item()) if isinstance(self._step, torch.Tensor) else self._step
            # Check periodically (every 100 steps) throughout training, not just warmup
            # This ensures diversity is maintained even when gradients are weak
            if step_value > 0 and step_value % 100 == 0:  # Every 100 steps
                with torch.no_grad():
                    queries_norm = F.normalize(self.prototype_queries, p=2, dim=1)
                    similarity_matrix = torch.mm(queries_norm, queries_norm.t())
                    off_diag_mask = ~torch.eye(self.num_prototypes, dtype=bool, device=similarity_matrix.device)
                    max_similarity = similarity_matrix[off_diag_mask].max()
                    # High threshold (0.95) - only trigger when prototypes are very similar
                    # This minimizes interference with normal gradient-based learning
                    if max_similarity > 0.95:
                        # Use QR decomposition for stable orthogonalization
                        Q, R = torch.linalg.qr(self.prototype_queries.t())
                        orthogonal_queries = Q.t()[:self.num_prototypes]
                        # Preserve magnitude to avoid disrupting training dynamics
                        orig_norms = self.prototype_queries.norm(p=2, dim=1, keepdim=True)
                        self.prototype_queries.data = F.normalize(orthogonal_queries, p=2, dim=1) * orig_norms

        apr_loss = None
        if with_loss:
            apr_loss = self.compute_apr_loss(prototypes_clean, logits)
            apr_value = apr_loss.detach()
        else:
            apr_value = self._update_metrics_no_grad(prototypes_clean.detach(), logits.detach())

        self._maybe_log(apr_value)
        return prototypes, apr_loss

    # === APR loss ===
    def compute_apr_loss(self, prototypes: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        loss_orth = self._orthogonality_term(prototypes)
        # Use both diversity terms: attention-based and direct prototype_queries-based
        # Direct term has stronger gradient, so give it more weight
        loss_div_attn = self._diversity_term(logits)
        loss_div_direct = self._diversity_term_direct()
        # Combine both terms with higher weight on direct term
        loss_div = loss_div_attn + 2.0 * loss_div_direct
        lam_orth, lam_div = self._effective_lambdas()
        apr_loss = lam_orth * loss_orth + lam_div * loss_div
        self._store_loss_terms(loss_orth, loss_div, apr_loss, lam_orth, lam_div)
        return apr_loss

    # === orthogonality term ===
    def _orthogonality_term(self, prototypes: torch.Tensor) -> torch.Tensor:
        P = F.normalize(prototypes, dim=-1)
        G = torch.einsum("ckd,cmd->ckm", P, P)
        K = G.size(-1)
        I = self._eye_buffer[:K, :K].to(G.device)
        off_mask = (1.0 - torch.eye(K, device=G.device))
        return ((G - I) ** 2 * off_mask).sum(dim=(-2, -1)).mean() / (K * K - K)

    # === diversity term ===
    def _diversity_term(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Encourage different prototypes to attend to different prompts.
        Minimize the similarity between attention distributions of different prototypes.
        
        Previous version encouraged uniform usage frequency but not diverse attention patterns.
        This version directly penalizes similarity between prototypes' attention distributions.
        """
        C, K, N = logits.shape
        w = torch.softmax(logits, dim=1)  # [C, K, N] - attention weights
        
        # For each class, compute similarity between different prototypes' attention distributions
        diversity_loss = 0.0
        for c in range(C):
            attn_c = w[c]  # [K, N] - attention distributions for all prototypes of this class
            
            # Normalize attention distributions
            attn_norm = F.normalize(attn_c, p=2, dim=1)  # [K, N]
            
            # Compute pairwise cosine similarity between prototypes' attention distributions
            similarity_matrix = torch.mm(attn_norm, attn_norm.t())  # [K, K]
            
            # Only consider off-diagonal elements (similarity between different prototypes)
            mask = ~torch.eye(K, dtype=bool, device=similarity_matrix.device)
            off_diag_similarities = similarity_matrix[mask]
            
            # Minimize similarity = maximize diversity
            # We want off-diagonal similarities to be close to 0 (different attention patterns)
            diversity_loss += off_diag_similarities.mean()
        
        return diversity_loss / C

    # === diversity term (direct on prototype_queries) ===
    def _diversity_term_direct(self) -> torch.Tensor:
        """
        Directly encourage diversity in prototype_queries.
        This provides stronger gradients when prototypes are similar.
        """
        # Normalize prototype_queries
        queries_norm = F.normalize(self.prototype_queries, p=2, dim=1)  # [K, D]
        
        # Compute pairwise cosine similarity
        similarity_matrix = torch.mm(queries_norm, queries_norm.t())  # [K, K]
        
        # Only consider off-diagonal elements
        mask = ~torch.eye(self.num_prototypes, dtype=bool, device=similarity_matrix.device)
        off_diag_similarities = similarity_matrix[mask]
        
        # Minimize similarity = maximize diversity
        return off_diag_similarities.mean()

    # === bookkeeping ===
    def _store_loss_terms(self, loss_orth, loss_div, apr_loss, lam_orth, lam_div):
        self.last_loss_terms = {
            "loss_orth": float(loss_orth.item()),
            "loss_div": float(loss_div.item()),
            "loss_apr": float(apr_loss.item()),
            "lambda_orth": float(lam_orth),
            "lambda_div": float(lam_div),
        }

    def _update_metrics_no_grad(self, prototypes, logits):
        with torch.no_grad():
            loss_orth = self._orthogonality_term(prototypes)
            loss_div_attn = self._diversity_term(logits)
            loss_div_direct = self._diversity_term_direct()
            loss_div = loss_div_attn + 2.0 * loss_div_direct
            lam_orth, lam_div = self._effective_lambdas()
            apr_loss = lam_orth * loss_orth + lam_div * loss_div
        self._store_loss_terms(loss_orth, loss_div, apr_loss, lam_orth, lam_div)
        return apr_loss.detach()

    # === logging ===
    def _maybe_log(self, apr_value):
        # Note: _step is now updated in forward() before computing APR loss
        step_value = int(self._step.item()) if isinstance(self._step, torch.Tensor) else self._step
        if (not self._logger) or (self.training and step_value % self.log_interval != 0):
            return
        if not _is_main_process():
            return
        monitor = monitor_prototype_metrics(self._last_prototypes, self._last_logits, step=step_value)
        if monitor:
            self.last_monitor_terms.update(monitor)
        msg = (
            f"[TPA] step={step_value:06d} "
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
