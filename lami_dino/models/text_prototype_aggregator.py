from __future__ import annotations
import math
import logging
from typing import Dict, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# APR ramps its regularizer strengths in over the first WARMUP_RATIO of training.
# The default below assumes the 12ep LVIS schedule (max_iter=85200); any run with
# a different max_iter should pass warmup_steps explicitly, otherwise a shorter
# schedule spends a disproportionate share of its training with APR suppressed
# (e.g. the 14200-iter quick ablations would keep it damped for the first 30%).
WARMUP_RATIO = 0.05
DEFAULT_MAX_ITER = 85200
DEFAULT_WARMUP_STEPS = int(DEFAULT_MAX_ITER * WARMUP_RATIO)  # ≈ 4260


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
        tau: float = 0.004375,         # Eq. (1) tau_p; gives 0.07 scale at d_h=256
        *,
        # Legacy parameter names are kept for checkpoint/config compatibility:
        # lambda_orth is Eq. (5)'s directional-diversity weight, while
        # lambda_div is Eq. (5)'s prototype-usage balance weight.
        lambda_orth: float = 0.10,
        lambda_div: float = 0.03,
        diversity_barrier_eps: float = 1e-4,
        slot_prior_strength: float = 0.0,
        prototype_mode_strength: float = 0.0,
        identity_value_init: bool = False,
        warmup_steps: int = DEFAULT_WARMUP_STEPS,  # set from train.max_iter * WARMUP_RATIO
        log_interval: int = 200,
    ) -> None:
        super().__init__()

        assert num_prototypes > 0 and hidden_dim > 0

        self.num_prototypes = num_prototypes
        self.tau = max(float(tau), 1e-6)
        # Paper Eq. (1): attention logits are divided by sqrt(d_h) * tau_p.
        # Keep the combined denominator explicit so prompt attention, APR usage,
        # and diagnostics all use exactly the same assignment distribution.
        self.attention_scale = math.sqrt(hidden_dim) * self.tau
        self.lambda_orth_base = float(lambda_orth)
        self.lambda_div_base = float(lambda_div)
        if not 0.0 < diversity_barrier_eps < 1.0:
            raise ValueError("diversity_barrier_eps must be within (0, 1)")
        self.diversity_barrier_eps = float(diversity_barrier_eps)
        if slot_prior_strength < 0.0:
            raise ValueError("slot_prior_strength must be non-negative")
        if prototype_mode_strength < 0.0:
            raise ValueError("prototype_mode_strength must be non-negative")
        self.register_buffer(
            "slot_prior_strength",
            torch.tensor(float(slot_prior_strength)),
            persistent=True,
        )
        self.register_buffer(
            "prototype_mode_strength",
            torch.tensor(float(prototype_mode_strength)),
            persistent=True,
        )
        self.identity_value_init = bool(identity_value_init)
        self.warmup_steps = int(warmup_steps)

        # projections
        self.key_proj = nn.Linear(dim, hidden_dim)
        self.value_proj = nn.Linear(dim, dim)
        self.prototype_queries = nn.Parameter(torch.empty(num_prototypes, hidden_dim))
        self.dropout = nn.Dropout(dropout)

        # buffers
        self._logger = logging.getLogger("lami_dino.tpa")
        self.log_interval = int(log_interval)
        # Persistent so that resuming from a checkpoint continues the APR warmup
        # instead of restarting it from zero and re-suppressing the regularizer.
        self.register_buffer("_step", torch.zeros((), dtype=torch.long), persistent=True)

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
        if self.identity_value_init:
            nn.init.eye_(self.value_proj.weight)
        else:
            nn.init.xavier_uniform_(self.value_proj.weight)
        nn.init.constant_(self.value_proj.bias, 0.0)
        nn.init.xavier_uniform_(self.prototype_queries)

    def _add_slot_prior(self, logits: torch.Tensor) -> torch.Tensor:
        """Seed prototype slots from evenly spaced semantic prompt roles."""
        num_prompts = logits.shape[-1]
        positions = torch.linspace(
            0,
            num_prompts - 1,
            steps=self.num_prototypes,
            device=logits.device,
        ).round().long()
        prior = logits.new_zeros((self.num_prototypes, num_prompts))
        prior.scatter_(
            1,
            positions[:, None],
            self.slot_prior_strength.to(dtype=logits.dtype).expand(
                self.num_prototypes, 1
            ),
        )
        return logits + prior.unsqueeze(0)

    def _centered_semantic_modes(
        self,
        prototypes: torch.Tensor,
        values: torch.Tensor,
    ) -> torch.Tensor:
        """Use fixed-radius prompt residuals so modes cannot shrink radially."""
        centroid = values.mean(dim=1, keepdim=True)
        residual = prototypes - centroid
        unit_residual = residual / residual.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        strength = self.prototype_mode_strength.to(dtype=prototypes.dtype)
        mode_radius = (
            strength
            * centroid.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        )
        fixed_radius_modes = centroid + mode_radius * unit_residual
        enabled = (strength > 0.0).to(dtype=prototypes.dtype)
        return prototypes + enabled * (fixed_radius_modes - prototypes)

    # === lambda warmup ===
    def _effective_lambdas(self) -> Tuple[float, float]:
        """
        Compute step-dependent regularization strengths (cosine warm-up).
        Smoothly ramps λ_orth and λ_div from 0 → base within warmup_steps.
        """
        if self.warmup_steps <= 0:
            return self.lambda_orth_base, self.lambda_div_base

        progress = min(1.0, (int(self._step) + 1) / float(self.warmup_steps))
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
        logits = self._add_slot_prior(logits)
        attn = F.softmax(logits / self.attention_scale, dim=-1)

        prototypes_clean = torch.einsum("ckn,cnd->ckd", attn, values)
        prototypes_clean = self._centered_semantic_modes(prototypes_clean, values)
        prototypes = self.dropout(prototypes_clean)

        self._last_logits = logits.detach()
        self._last_prototypes = prototypes_clean.detach()
        # Diagnostics must describe this forward. Keeping the previous dict made
        # metrics.json repeat stale rank/cosine values until log_interval happened
        # to refresh the cache.
        self.last_monitor_terms = {}

        apr_loss = None
        if with_loss:
            apr_loss = self.compute_apr_loss(prototypes_clean, logits)
            apr_value = apr_loss.detach()
        else:
            apr_value = self._update_metrics_no_grad(prototypes_clean.detach(), logits.detach())

        # Only training iterations advance the warmup clock. Counting eval forwards
        # too would fast-forward the schedule by however many validation passes have
        # run, which is not what "5% of training" is supposed to mean.
        if self.training:
            self._step += 1

        self._maybe_log(apr_value)
        return prototypes, apr_loss

    # === APR loss ===
    def compute_apr_loss(self, prototypes: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        loss_orth = self._orthogonality_term(prototypes)
        loss_balance = self._balance_term(logits)
        lam_orth, lam_balance = self._effective_lambdas()
        apr_loss = lam_orth * loss_orth + lam_balance * loss_balance
        self._store_loss_terms(loss_orth, loss_balance, apr_loss, lam_orth, lam_balance)
        return apr_loss

    # === adaptive directional-diversity barrier ===
    def _orthogonality_term(self, prototypes: torch.Tensor) -> torch.Tensor:
        """Log barrier against parallel or anti-parallel prototype directions.

        The previous mean squared cosine has the right minimum but a vanishing
        gradient as normalized vectors become parallel. Consequently, detector
        gradients could drive all slots into that bad stationary point. This
        barrier has the same optimum at zero cosine while its slope increases as
        |cosine| approaches one:

            -log(1 - (1-eps) * cosine^2).

        K=1 remains the no-op control and returns a graph-connected zero.
        """
        if prototypes.size(-2) < 2:
            return prototypes.sum() * 0.0
        P = F.normalize(prototypes, dim=-1)
        G = torch.einsum("ckd,cmd->ckm", P, P)
        K = G.size(-1)
        off_mask = ~torch.eye(K, device=G.device, dtype=torch.bool)
        cosine_sq = G[:, off_mask].square()
        return -torch.log1p(
            -(1.0 - self.diversity_barrier_eps) * cosine_sq
        ).mean()

    # === prototype usage balance term ===
    def _balance_term(self, logits: torch.Tensor) -> torch.Tensor:
        """Eq. (5)'s KL divergence from prototype usage to a uniform prior.

        ``logits[c,k,n]`` score how strongly slot k claims prompt n.  We first
        normalize over slots for each prompt, average those assignments over the
        prompt bank, and minimize KL(pi || Uniform).  Directional diversity is a
        separate term: balanced usage alone is blind to identical slots, which is
        why collapse monitoring must continue to use prototype cosine/effective
        rank rather than usage entropy.
        """
        C, K, N = logits.shape
        if K < 2:
            return logits.sum() * 0.0
        assignments = torch.softmax(logits / self.attention_scale, dim=1)
        usage = assignments.mean(dim=-1)
        usage = usage / usage.sum(dim=1, keepdim=True).clamp_min(1e-8)
        return (usage * (usage.clamp_min(1e-8).log() + math.log(K))).sum(dim=1).mean()

    # Backward-compatible private name used by older diagnostics/tests.
    def _diversity_term(self, logits: torch.Tensor) -> torch.Tensor:
        return self._balance_term(logits)

    # === bookkeeping ===
    def _store_loss_terms(self, loss_orth, loss_balance, apr_loss, lam_orth, lam_balance):
        self.last_loss_terms = {
            "loss_orth": float(loss_orth.item()),
            # Keep legacy keys for existing metrics readers and add paper names.
            "loss_div": float(loss_balance.item()),
            "loss_prototype_diversity": float(loss_orth.item()),
            "loss_balance": float(loss_balance.item()),
            "loss_apr": float(apr_loss.item()),
            "lambda_orth": float(lam_orth),
            "lambda_div": float(lam_balance),
            "lambda_balance": float(lam_balance),
        }

    def _update_metrics_no_grad(self, prototypes, logits):
        with torch.no_grad():
            loss_orth = self._orthogonality_term(prototypes)
            loss_balance = self._balance_term(logits)
            lam_orth, lam_balance = self._effective_lambdas()
            apr_loss = lam_orth * loss_orth + lam_balance * loss_balance
        self._store_loss_terms(loss_orth, loss_balance, apr_loss, lam_orth, lam_balance)
        return apr_loss.detach()

    # === logging ===
    def _maybe_log(self, apr_value):
        # Training-only. The old guard `self.training and step % interval` never
        # returned early in eval mode, so every eval forward emitted a [TPA] log
        # line -- thousands per validation pass. The monitors describe the same
        # weights in eval as in training, so nothing is lost.
        if not self.training:
            return
        step = int(self._step)
        if (not self._logger) or step % self.log_interval != 0:
            return
        if not _is_main_process():
            return
        monitor = monitor_prototype_metrics(
            self._last_prototypes,
            self._last_logits,
            step=step,
            attention_tau=self.attention_scale,
        )
        if monitor:
            self.last_monitor_terms.update(monitor)
        msg = (
            f"[TPA] step={step:06d} "
            # proto_cos/eff_rank first: usage_entropy reads healthy under collapse.
            f"proto_cos={monitor.get('proto_pairwise_cos', 0):.4f} "
            f"eff_rank={monitor.get('proto_effective_rank', 0):.3f} "
            f"orth_off={monitor.get('orth_off_mse', 0):.4f} "
            f"diag_mse={monitor.get('diag_mse', 0):.4f} "
            f"usage_entropy={monitor.get('usage_entropy', 0):.4f} "
            f"apr={float(apr_value):.5f} "
            f"(λ_proto_div={self.last_loss_terms.get('lambda_orth', 0):.3f}, "
            f"λ_bal={self.last_loss_terms.get('lambda_balance', 0):.3f})"
        )
        self._logger.info(msg)

    def get_monitor_dict(self) -> Dict[str, float]:
        out = dict(self.last_loss_terms)
        if not self.last_monitor_terms and self._last_prototypes is not None:
            with torch.no_grad():
                off_mse, diag_mse = compute_prototype_orthogonality(self._last_prototypes)
                usage = compute_usage_entropy(self._last_logits, tau=self.attention_scale)
                cos, rank = compute_prototype_similarity(self._last_prototypes)
            self.last_monitor_terms = {
                "orth_off_mse": off_mse,
                "diag_mse": diag_mse,
                "usage_entropy": usage,
                # The two that actually detect collapse. usage_entropy reads 0.9999
                # when every prototype is the same vector -- they are all used
                # equally precisely because they are indistinguishable -- so it must
                # not be read as a health signal on its own.
                "proto_pairwise_cos": cos,
                "proto_effective_rank": rank,
            }
        out.update(self.last_monitor_terms)
        return out


@torch.no_grad()
def compute_prototype_similarity(prototypes):
    """Return (mean pairwise cosine, mean effective rank) over the K prototypes.

    Effective rank is exp(entropy of the normalized singular values): 1.0 means the
    K prototypes span a single direction, K means they are mutually independent.
    Reference points measured on the real LVIS bank: a collapsed aggregator sits at
    cosine 0.99996 / rank 1.05, which is where the shipped checkpoints were.
    """
    if prototypes is None:
        return float("nan"), float("nan")
    P = F.normalize(prototypes, dim=-1)
    G = torch.einsum("ckd,cmd->ckm", P, P)
    K = G.size(-1)
    if K < 2:
        return float("nan"), float(K)
    off = (G.sum(dim=(-2, -1)) - G.diagonal(dim1=-2, dim2=-1).sum(-1)) / (K * K - K)
    # Singular values of P via eigvalsh of the K x K gram: sqrt(eig(P P^T)) equals
    # svdvals(P), but eigvalsh exists on the older torch of the training env
    # (torch.linalg.svdvals only appeared in 1.10) and works on a smaller matrix.
    evals = torch.linalg.eigvalsh(G.float()).clamp_min(0.0)
    svals = evals.sqrt()
    frac = svals / svals.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    eff_rank = torch.exp(-(frac * frac.clamp_min(1e-12).log()).sum(dim=-1))
    return float(off.mean().item()), float(eff_rank.mean().item())


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
def compute_usage_entropy(logits, tau=1.0):
    if logits is None:
        return float("nan")
    C, K, N = logits.shape
    if K < 2:
        return 0.0
    w = torch.softmax(logits / tau, dim=1)
    votes = w.sum(dim=-1)
    p = votes / (votes.sum(dim=1, keepdim=True) + 1e-8)
    entropy = -(p * (p.clamp_min(1e-8)).log()).sum(dim=1) / math.log(K)
    return float(entropy.mean().item())


def monitor_prototype_metrics(
    prototypes,
    logits,
    step=0,
    prefix="[TPA]",
    attention_tau=1.0,
):
    """Single source of truth for the diagnostics, so every caller sees the same set.

    get_monitor_dict only recomputes when its cache is empty, so any metric missing
    here silently never reaches metrics.json once _maybe_log has populated it.
    """
    off_mse, diag_mse = compute_prototype_orthogonality(prototypes)
    usage_entropy = compute_usage_entropy(logits, tau=attention_tau)
    proto_cos, proto_rank = compute_prototype_similarity(prototypes)
    if step % 200 == 0 and _is_main_process():
        print(f"{prefix} step={step:06d} | proto_cos={proto_cos:.4f} | eff_rank={proto_rank:.3f} "
              f"| orth_off={off_mse:.4f} | diag_mse={diag_mse:.4f} | usage_entropy={usage_entropy:.4f}")
    return {
        "orth_off_mse": off_mse,
        "diag_mse": diag_mse,
        "usage_entropy": usage_entropy,
        "proto_pairwise_cos": proto_cos,
        "proto_effective_rank": proto_rank,
    }


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
