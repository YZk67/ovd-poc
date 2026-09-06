
# rpsa.py
# Region–Prototype Semantic Alignment (RPSA) module
# --------------------------------------------------
# Provides:
#   - soft_kmeans_assign: differentiable soft clustering for region tokens
#   - compute_pi_weights: soft cluster->class weights using GT/pseudo masks
#   - weighted_infoNCE: weighted contrastive alignment between visual centers and text prototypes
#   - RPSAModule: nn.Module wrapping the above with convenient configuration
#
# Expected tensors:
#   region_feats: [B, N, D]            (encoder tokens)
#   text_protos:  [C, Kp, D]           (TPA prototypes, already projected to D)
#   token_cls_mask: [B, N, C]          (soft or hard masks; GT or pseudo)
#

from __future__ import annotations
import logging
import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger("lami_dino.rpsa")


# -------------------------------
# Utilities
# -------------------------------

def l2_normalize(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    return x / (x.norm(dim=dim, keepdim=True).clamp_min(eps))


def pairwise_sqdist(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Compute squared Euclidean distances between sets of vectors.
    a: [B, N, D]
    b: [B, K, D]
    return: [B, N, K]
    """
    # ||a-b||^2 = ||a||^2 + ||b||^2 - 2 a.b
    a2 = (a * a).sum(-1, keepdim=True)          # [B,N,1]
    b2 = (b * b).sum(-1)                        # [B,K]
    ab = torch.einsum('bnd,bkd->bnk', a, b)     # [B,N,K]
    # a2 is [B,N,1], b2 is [B,K], need to broadcast to [B,N,K]
    b2_expanded = b2.unsqueeze(1)               # [B,1,K]
    return (a2 + b2_expanded - 2 * ab).clamp_min(0.0)


# -------------------------------
# Soft K-Means (single EM step)
# -------------------------------

@torch.no_grad()
def _init_centers_fps(region_feats: torch.Tensor, K: int) -> torch.Tensor:
    """
    Farthest-Point Seeding (deterministic-ish) for initial centers.
    region_feats: [B, N, D]
    return: centers0 [B, K, D] (no grad)
    """
    B, N, D = region_feats.shape
    x = region_feats  # no copy
    # Start from the token with largest L2 norm, then farthest sequentially.
    norms = (x * x).sum(-1)                     # [B,N]
    first = norms.argmax(dim=1)                 # [B]
    centers = torch.empty(B, K, D, device=x.device, dtype=x.dtype)
    centers[:, 0] = x[torch.arange(B), first]

    # Precompute distances to first center
    centers_0 = centers[:, 0:1, :]  # [B, 1, D]
    dist = pairwise_sqdist(x, centers_0).squeeze(-1)  # [B,N]
    for k in range(1, K):
        idx = dist.argmax(dim=1)                               # [B]
        centers[:, k] = x[torch.arange(B), idx]
        # update min distance to any chosen center
        centers_k = centers[:, k:k+1, :]  # [B, 1, D]
        newdist = pairwise_sqdist(x, centers_k).squeeze(-1)  # [B,N]
        dist = torch.minimum(dist, newdist)
    return centers


def soft_kmeans_assign(region_feats: torch.Tensor,
                       K: int,
                       sigma: float = 1.0,
                       iters: int = 1,
                       init_centers: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Differentiable soft k-means (a few EM steps). Returns assignments r and centers mu.
    region_feats: [B, N, D]
    K: number of visual clusters
    sigma: temperature on squared distance (larger -> softer assignments)
    iters: EM iterations (1-3 is sufficient)
    init_centers: optional [B, K, D]
    returns:
        r:  [B, N, K]  soft assignment (rows sum to 1)
        mu: [B, K, D]  centers
    """
    B, N, D = region_feats.shape
    eps = 1e-6
    x = region_feats

    if init_centers is None:
        centers = _init_centers_fps(x.detach(), K)  # no grad seeding
    else:
        centers = init_centers.detach()

    for _ in range(iters):
        # E-step: responsibilities
        dist2 = pairwise_sqdist(x, centers)                     # [B,N,K]
        logits = -dist2 / (sigma * sigma)                       # [B,N,K]
        r = torch.softmax(logits, dim=-1)                       # [B,N,K]

        # M-step: centers
        # mass per cluster: [B,K,1] to broadcast along D, avoid mismatching D(256) vs K(8)
        mass = r.sum(dim=1).unsqueeze(-1).clamp_min(eps)        # [B,K,1]
        mu = torch.einsum('bnd,bnk->bkd', x, r) / mass          # [B,K,D]
        centers = mu.detach()                                   # use fresh centers for next E
    return r, mu


# -------------------------------
# Pi weights (cluster -> class)
# -------------------------------

def compute_pi_weights(assign_r: torch.Tensor,
                       token_cls_mask: torch.Tensor) -> torch.Tensor:
    """
    Compute soft cluster->class weights π using GT/pseudo masks.
    assign_r:      [B, N, K]
    token_cls_mask:[B, N, C]  (0/1 or soft probabilities)
    returns:
        pi: [B, K, C]  (rows over classes are not forced to sum to 1)
    """
    eps = 1e-6
    # numerator: sum_n r_{n,k} * M_{n,c}
    num = torch.einsum('bnk,bnc->bkc', assign_r, token_cls_mask)       # [B,K,C]
    den = assign_r.sum(dim=1, keepdim=True).transpose(1, 2).clamp_min(eps)  # [B,K,1]
    pi = num / den                                                      # [B,K,C]
    return pi.clamp_min(0.0)


def select_high_confidence_tokens(
    region_feats: torch.Tensor,
    token_cls_mask: torch.Tensor,
    *,
    valid_mask: Optional[torch.Tensor] = None,
    topk: int = 0,
    confidence_threshold: float = 0.0,
    min_tokens: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Select real encoder tokens without padding or duplicated fallbacks.

    A previous implementation replaced every below-threshold position with the
    same highest-confidence token in order to keep a fixed tensor shape. That
    could duplicate one token hundreds of times and collapse soft k-means. Here
    the common selection size is the minimum eligible count in the batch, so
    every returned index is a distinct, valid token.
    """

    if region_feats.ndim != 3 or token_cls_mask.ndim != 3:
        raise ValueError("region_feats and token_cls_mask must both be rank-3")
    if region_feats.shape[:2] != token_cls_mask.shape[:2]:
        raise ValueError("region_feats and token_cls_mask must share [B, N]")
    if confidence_threshold < 0:
        raise ValueError("confidence_threshold must be non-negative")
    if min_tokens < 1:
        raise ValueError("min_tokens must be at least one")

    batch_size, num_tokens, feat_dim = region_feats.shape
    if valid_mask is None:
        valid_mask = torch.ones(
            batch_size, num_tokens, dtype=torch.bool, device=region_feats.device
        )
    if valid_mask.shape != (batch_size, num_tokens):
        raise ValueError(
            f"valid_mask must have shape {(batch_size, num_tokens)}, got {valid_mask.shape}"
        )
    valid_mask = valid_mask.to(device=region_feats.device, dtype=torch.bool)

    confidence = token_cls_mask.max(dim=-1).values
    eligible = valid_mask & torch.isfinite(confidence)
    if confidence_threshold > 0:
        eligible = eligible & (confidence >= confidence_threshold)

    eligible_counts = eligible.sum(dim=1)
    common_count = int(eligible_counts.min().item())
    requested = common_count if topk <= 0 else min(int(topk), common_count)
    if requested < min_tokens:
        raise ValueError(
            "not enough distinct valid high-confidence tokens for RPSA: "
            f"minimum eligible count={common_count}, required={min_tokens}, "
            f"threshold={confidence_threshold}"
        )

    selection_scores = confidence.masked_fill(~eligible, float("-inf"))
    indices = selection_scores.topk(requested, dim=1).indices
    feat_indices = indices.unsqueeze(-1).expand(-1, -1, feat_dim)
    mask_indices = indices.unsqueeze(-1).expand(-1, -1, token_cls_mask.size(-1))
    selected_feats = torch.gather(region_feats, 1, feat_indices)
    selected_cls_mask = torch.gather(token_cls_mask, 1, mask_indices)
    return selected_feats, selected_cls_mask, indices


# -------------------------------
# Weighted InfoNCE
# -------------------------------

def weighted_infoNCE(mu: torch.Tensor,
                     text_protos: torch.Tensor,
                     pi: torch.Tensor,
                     tau: float = 0.07,
                     alpha_pi: float = 1.0,
                     bg_thresh: Optional[float] = 0.1,
                     adaptive_bg: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Weighted InfoNCE alignment between visual centers and text prototypes.
    mu:          [B, K, D]   (normalized or not, will normalize inside)
    text_protos: [C, Kp, D]  (will normalize)
    pi:          [B, K, C]   (soft positive weights)
    tau:         temperature
    alpha_pi:    sharpen/soften pi -> pi_tilde = pi^alpha / sum
    bg_thresh:   clusters with max_c pi < bg_thresh are treated as background (ignored)
    returns:
        loss: scalar
        stats: dict of diagnostic tensors
    """
    B, K, D = mu.shape
    C, Kp, D2 = text_protos.shape
    assert D == D2, "Dimension mismatch between mu and text_protos"

    mu_n = l2_normalize(mu, dim=-1)                                 # [B,K,D]
    P = l2_normalize(text_protos.view(C * Kp, D), dim=-1)           # [C*Kp, D]
    S = torch.einsum('bkd,md->bkm', mu_n, P).view(B, K, C, Kp)      # [B,K,C,Kp]

    # positive weights
    pi_clamped = pi.clamp_min(0.0)
    # background mask uses the pre-normalized magnitude so tiny clusters remain filtered out
    pi_max_raw = pi_clamped.max(dim=-1).values
    threshold = None
    if adaptive_bg is not None:
        # adaptive_bg: [B,1] or [B,K], compute mask per sample
        threshold = adaptive_bg
        if threshold.dim() == 2 and threshold.size(1) == 1:
            threshold = threshold.expand_as(pi_max_raw)
        elif threshold.dim() == 1:
            threshold = threshold.view(-1, 1).expand_as(pi_max_raw)
        threshold = threshold.to(device=pi_max_raw.device, dtype=pi_max_raw.dtype)
    if bg_thresh is not None:
        fixed_threshold = pi_max_raw.new_tensor(float(bg_thresh))
        threshold = (
            fixed_threshold
            if threshold is None
            else torch.maximum(threshold, fixed_threshold)
        )
    if threshold is None:
        # Both bg_thresh and adaptive_bg are None: treat all clusters as valid (no background filtering)
        bg_mask = torch.zeros(B, K, dtype=torch.bool, device=pi_max_raw.device)
    else:
        bg_mask = pi_max_raw < threshold
    pi_tilde = (pi_clamped ** alpha_pi)
    pi_tilde = pi_tilde / (pi_tilde.sum(dim=-1, keepdim=True).clamp_min(1e-6))  # [B,K,C]

    # logits
    pos = torch.logsumexp(S / tau, dim=-1)                          # [B,K,C] over Kp
    all_ = torch.logsumexp(S.view(B, K, -1) / tau, dim=-1, keepdim=True)  # [B,K,1]

    # weighted InfoNCE per cluster
    loss_k = -(pi_tilde * (pos - all_)).sum(dim=-1)                 # [B,K]
    loss_k = loss_k.masked_fill(bg_mask, 0.0)                       # ignore BG

    valid_counts = (~bg_mask).float().sum(dim=1)                    # [B]
    valid_total = valid_counts.sum()
    # Eq. (6) averages over the valid-center set V.  V can legitimately be
    # empty for an early/noisy mini-batch after confidence filtering.  In that
    # case there is no RPSA supervision to apply: return a graph-connected zero
    # instead of dividing by zero, retaining a low-confidence center, or
    # aborting the detector training run.  Shape and non-finite failures remain
    # hard errors in RPSAModule.
    loss = loss_k.sum() / valid_total.clamp_min(1.0)

    stats = {
        "rpsa_pos_mean": pos.mean().detach(),
        "rpsa_bg_ratio": (bg_mask.float().mean().detach()),
        "rpsa_valid_clusters": valid_counts.mean().detach(),
        "rpsa_active": (valid_total > 0).to(dtype=mu.dtype).detach(),
        "rpsa_empty_image_ratio": (valid_counts == 0).float().mean().detach(),
    }
    return loss, stats


def teacher_routed_mode_alignment(
    student_regions: torch.Tensor,
    teacher_regions: torch.Tensor,
    candidate_prototypes: torch.Tensor,
    *,
    candidate_category_probs: Optional[torch.Tensor] = None,
    valid_mask: Optional[torch.Tensor] = None,
    allowed_category_mask: Optional[torch.Tensor] = None,
    proposal_weights: Optional[torch.Tensor] = None,
    group_masks: Optional[torch.Tensor] = None,
    group_weights: Optional[torch.Tensor] = None,
    temperature: float = 0.07,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Distill full-vocabulary CLIP region routing into category prototypes.

    Args:
        student_regions: Detector proposal features ``[B, M, D]``.
        teacher_regions: Frozen CLIP ROI features ``[B, M, D]``.
        candidate_prototypes: Per-proposal category prototype sets
            ``[B, M, L, Kp, D]``. ``L`` is a short full-vocabulary category
            list selected by the teacher; it is independent of FedLoss.
        candidate_category_probs: Frozen CLIP probabilities for those ``L``
            categories, ``[B, M, L]``. The final target factorizes into this
            category distribution and a CLIP-derived distribution over the
            prototypes within each category.
        valid_mask: Proposals trusted for pseudo supervision, ``[B, M]``.
        allowed_category_mask: Optional ``[B, M, L]`` mask. GT-matched
            proposals use it to restrict the teacher target to the known class
            while retaining a soft distribution over that class's modes.
        proposal_weights: Optional per-proposal balancing weights ``[B, M]``.
        group_masks: Optional disjoint supervision groups ``[G, B, M]``.
            When present, the loss is averaged inside each group first, so a
            numerous group cannot drown out a sparse one.
        group_weights: Optional non-negative relative weights ``[G]`` used to
            combine active group means.
        temperature: Category-and-mode distillation temperature.

    The target distribution is detached, while the student features and live
    prototypes receive gradients. Consequently the frozen teacher cannot be
    changed to make its own pseudo labels easier to satisfy.
    """
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if student_regions.ndim != 3 or teacher_regions.ndim != 3:
        raise ValueError("student_regions and teacher_regions must be [B, M, D]")
    if student_regions.shape != teacher_regions.shape:
        raise ValueError(
            "student_regions and teacher_regions must have identical shapes, "
            f"got {student_regions.shape} and {teacher_regions.shape}"
        )
    if candidate_prototypes.ndim != 5:
        raise ValueError("candidate_prototypes must be [B, M, L, Kp, D]")

    batch, proposals, dim = student_regions.shape
    if candidate_prototypes.shape[:2] != (batch, proposals):
        raise ValueError("candidate_prototypes must share [B, M]")
    if candidate_prototypes.shape[-1] != dim:
        raise ValueError("region and prototype dimensions must match")

    device = student_regions.device
    if valid_mask is None:
        valid_mask = torch.ones(batch, proposals, dtype=torch.bool, device=device)
    if valid_mask.shape != (batch, proposals):
        raise ValueError("valid_mask must have shape [B, M]")
    valid_mask = valid_mask.to(device=device, dtype=torch.bool)

    category_count = candidate_prototypes.shape[2]
    if candidate_category_probs is None:
        candidate_category_probs = student_regions.new_ones(
            batch, proposals, category_count
        )
    if candidate_category_probs.shape != (batch, proposals, category_count):
        raise ValueError("candidate_category_probs must have shape [B, M, L]")
    candidate_category_probs = candidate_category_probs.to(
        device=device, dtype=student_regions.dtype
    )
    if (candidate_category_probs < 0).any():
        raise ValueError("candidate_category_probs must be non-negative")

    if allowed_category_mask is None:
        allowed_category_mask = torch.ones(
            batch,
            proposals,
            category_count,
            dtype=torch.bool,
            device=device,
        )
    if allowed_category_mask.shape != (batch, proposals, category_count):
        raise ValueError("allowed_category_mask must have shape [B, M, L]")
    allowed_category_mask = allowed_category_mask.to(device=device, dtype=torch.bool)
    if (valid_mask & ~allowed_category_mask.any(dim=-1)).any():
        raise ValueError("every valid proposal must allow at least one category")

    if proposal_weights is None:
        proposal_weights = student_regions.new_ones((batch, proposals))
    if proposal_weights.shape != (batch, proposals):
        raise ValueError("proposal_weights must have shape [B, M]")
    proposal_weights = proposal_weights.to(
        device=device, dtype=student_regions.dtype
    ).clamp_min(0.0)

    student = l2_normalize(student_regions, dim=-1)
    teacher = l2_normalize(teacher_regions.detach(), dim=-1)
    prototypes = l2_normalize(candidate_prototypes, dim=-1)

    student_logits = torch.einsum("bmd,bmlkd->bmlk", student, prototypes)
    with torch.no_grad():
        teacher_mode_logits = torch.einsum(
            "bmd,bmlkd->bmlk", teacher, prototypes.detach()
        )
        teacher_mode_probs = torch.softmax(
            teacher_mode_logits / temperature,
            dim=-1,
        )
        teacher_category_probs = candidate_category_probs.detach().masked_fill(
            ~allowed_category_mask, 0.0
        )
        teacher_category_probs = teacher_category_probs / (
            teacher_category_probs.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        )
        teacher_probs = (
            teacher_category_probs.unsqueeze(-1) * teacher_mode_probs
        ).flatten(start_dim=2)

    student_log_probs = torch.log_softmax(
        (student_logits / temperature).flatten(start_dim=2), dim=-1
    )
    loss_per_proposal = F.kl_div(
        student_log_probs,
        teacher_probs,
        reduction="none",
    ).sum(dim=-1)

    effective_weights = proposal_weights * valid_mask.to(proposal_weights.dtype)
    weight_sum = effective_weights.sum()
    group_losses = None
    group_active = None
    if group_masks is None:
        loss = (
            (loss_per_proposal * effective_weights).sum()
            / weight_sum.clamp_min(1.0)
        )
    else:
        if group_masks.ndim != 3 or group_masks.shape[1:] != (batch, proposals):
            raise ValueError("group_masks must have shape [G, B, M]")
        group_masks = group_masks.to(device=device, dtype=torch.bool)
        if (group_masks & ~valid_mask.unsqueeze(0)).any():
            raise ValueError("group_masks must be subsets of valid_mask")
        if (group_masks.sum(dim=0) > 1).any():
            raise ValueError("group_masks must be disjoint")

        group_count = group_masks.shape[0]
        if group_weights is None:
            group_weights = student_regions.new_ones(group_count)
        if group_weights.shape != (group_count,):
            raise ValueError("group_weights must have shape [G]")
        group_weights = group_weights.to(
            device=device,
            dtype=student_regions.dtype,
        )
        if (group_weights < 0).any():
            raise ValueError("group_weights must be non-negative")

        grouped_weights = (
            group_masks.to(effective_weights.dtype)
            * effective_weights.unsqueeze(0)
        )
        group_denominators = grouped_weights.sum(dim=(1, 2))
        group_losses = (
            (grouped_weights * loss_per_proposal.unsqueeze(0)).sum(dim=(1, 2))
            / group_denominators.clamp_min(1.0)
        )
        group_active = group_denominators > 0
        active_group_weights = (
            group_weights * group_active.to(group_weights.dtype)
        )
        loss = (
            (group_losses * active_group_weights).sum()
            / active_group_weights.sum().clamp_min(1.0)
        )
    if not torch.isfinite(loss):
        raise FloatingPointError("teacher-routed mode alignment produced non-finite loss")

    with torch.no_grad():
        entropy_denominator = math.log(max(teacher_probs.shape[-1], 2))
        teacher_entropy = -(
            teacher_probs.clamp_min(1e-8).log() * teacher_probs
        ).sum(dim=-1) / entropy_denominator
        marginal_mode_probs = teacher_probs.view(
            batch,
            proposals,
            category_count,
            candidate_prototypes.shape[3],
        ).sum(dim=2)
        stats = {
            "teacher_rpsa_active": (weight_sum > 0).to(student_regions.dtype),
            "teacher_rpsa_valid_proposals": valid_mask.float().sum(dim=1).mean(),
            "teacher_rpsa_valid_ratio": valid_mask.float().mean(),
            "teacher_rpsa_target_entropy": (
                teacher_entropy * valid_mask.to(teacher_entropy.dtype)
            ).sum() / valid_mask.sum().clamp_min(1),
            "teacher_rpsa_mode_max_weight": (
                marginal_mode_probs.max(dim=-1).values
                * valid_mask.to(marginal_mode_probs.dtype)
            ).sum() / valid_mask.sum().clamp_min(1),
            "teacher_rpsa_category_max_weight": (
                teacher_category_probs.max(dim=-1).values
                * valid_mask.to(teacher_category_probs.dtype)
            ).sum() / valid_mask.sum().clamp_min(1),
        }
        if group_losses is not None:
            stats["teacher_rpsa_group_losses"] = group_losses.detach()
            stats["teacher_rpsa_group_active"] = group_active.detach()
    return loss, stats


class RPSAModule(nn.Module):
    def __init__(self,
                 K: int = 8, # number of visual clusters per image
                 sigma: float = 1.0,
                 em_iters: int = 1,
                 tau_align: float = 0.07,
                 alpha_pi: float = 1.0,
                 bg_thresh: Optional[float] = 0.1,
                 subsample_tokens: int = 0,
                 subsample_method: str = "random",
                 subsample_fg_ratio: float = 0.5,
                 bg_percentile: float = 0.0,
                 detach_pi: bool = True,
                 stop_grad_text: bool = False,
                 stop_grad_vision: bool = False):
        """
        Args:
            K: number of visual clusters per image
            sigma: soft k-means temperature on squared distance
            em_iters: soft k-means EM iterations
            tau_align: InfoNCE temperature
            alpha_pi: exponent on pi weights (sharpen/soften)
            bg_thresh: background threshold on max pi per cluster. If None, rely solely on bg_percentile.
            subsample_tokens: if >0, subsample tokens per image for RPSA (speed)
            subsample_method: strategy for subsampling ('random', 'confidence', 'hybrid')
            subsample_fg_ratio: foreground quota when using 'hybrid' strategy
            bg_percentile: optional percentile (0-1) for adaptive bg threshold; applied before bg_thresh
            detach_pi: if True, stop gradients through pi weights (routing)
            stop_grad_text: if True, stop gradient to text prototypes branch
            stop_grad_vision: if True, stop gradient to region features branch
        """
        super().__init__()
        self.K = K
        self.sigma = float(sigma)
        self.em_iters = int(em_iters)
        self.tau_align = float(tau_align)
        self.alpha_pi = float(alpha_pi)
        self.bg_thresh = float(bg_thresh) if bg_thresh is not None else None
        self.subsample_tokens = int(subsample_tokens)
        self.subsample_method = str(subsample_method)
        self.subsample_fg_ratio = float(subsample_fg_ratio)
        self.bg_percentile = float(bg_percentile)
        if not 0.0 <= self.bg_percentile <= 1.0:
            raise ValueError("bg_percentile must be within [0,1]")
        self.detach_pi = bool(detach_pi)
        self.stop_grad_text = bool(stop_grad_text)
        self.stop_grad_vision = bool(stop_grad_vision)
        if self.subsample_tokens < 0:
            raise ValueError("subsample_tokens must be >=0")
        if not (0.0 <= self.subsample_fg_ratio <= 1.0):
            raise ValueError("subsample_fg_ratio must be within [0,1]")

    def _sample_token_indices(self, token_cls_mask: torch.Tensor, target: int) -> torch.Tensor:
        """
        Vectorized subsampling of token indices. Returns LongTensor [B,target].
        Designed to keep the heavy ops on GPU (no per-sample Python loops).
        """
        B, N, _ = token_cls_mask.shape
        device = token_cls_mask.device
        method = self.subsample_method.lower()

        if target <= 0 or target >= N:
            return torch.arange(N, device=device)[None, :target].expand(B, -1)

        if method == "random":
            scores = torch.rand(B, N, device=device)
            idx = scores.topk(target, dim=1).indices
            return idx

        scores = token_cls_mask.max(dim=-1).values  # [B,N]
        if method == "confidence":
            idx = scores.topk(target, dim=1).indices
            return idx

        if method == "hybrid":
            fg_target = max(1, int(round(target * self.subsample_fg_ratio)))
            fg_target = min(fg_target, target)
            bg_target = target - fg_target
            fg_idx = scores.topk(fg_target, dim=1).indices  # [B,fg_target]

            if bg_target > 0:
                rand = torch.rand(B, N, device=device)
                rand.scatter_(1, fg_idx, -1.0)  # mask already selected foreground
                bg_idx = rand.topk(bg_target, dim=1).indices
                idx = torch.cat([fg_idx, bg_idx], dim=1)
            else:
                idx = fg_idx

            if idx.size(1) < target:  # pad (rare: not enough distinct bg tokens)
                pad = fg_idx[:, : target - idx.size(1)]
                idx = torch.cat([idx, pad], dim=1)
            return idx[:, :target]

        raise ValueError(f"[RPSA] Unknown subsample_method: {self.subsample_method}")

    def forward(self,
                region_feats: torch.Tensor,   # [B,N,D]
                text_protos: torch.Tensor,    # [C,Kp,D]
                token_cls_mask: torch.Tensor  # [B,N,C]  (GT or pseudo)
                ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:

        B, N, D = region_feats.shape

        # Optional subsampling for speed
        if (self.subsample_tokens > 0) and (N > self.subsample_tokens):
            target = min(self.subsample_tokens, N)
            idx = self._sample_token_indices(token_cls_mask, target)
            idx = idx.to(dtype=torch.long)
            gather_idx_feat = idx.unsqueeze(-1).expand(-1, -1, D)
            gather_idx_mask = idx.unsqueeze(-1).expand(-1, -1, token_cls_mask.shape[-1])
            region_feats_s = torch.gather(region_feats, 1, gather_idx_feat)
            token_cls_mask_s = torch.gather(token_cls_mask, 1, gather_idx_mask)
        else:
            region_feats_s = region_feats
            token_cls_mask_s = token_cls_mask

        # Optional stop-grad on branches
        v = region_feats_s.detach() if self.stop_grad_vision else region_feats_s
        t = text_protos.detach()   if self.stop_grad_text   else text_protos

        # 1) soft clustering
        # assign_r (软分配矩阵): [B,N,K], 表示每个token属于每个cluster的概率; 
        #          行归一化后为1, 表示软分配; 用途: 计算pi权重，建立聚类与类别的关联
        # centers_mu (聚类中心): [B,K,D], 每个聚类的代表性特征向量, 表示每个cluster的中心; 用途: 作为视觉中心参与InfoNCE对比学
        try:
            assign_r, centers_mu = soft_kmeans_assign(v, K=self.K, sigma=self.sigma, iters=self.em_iters)
        except RuntimeError as e:
            logger.error(f"[RPSA] Error in soft_kmeans_assign: {e}, K={self.K}, v.shape={v.shape}")
            raise

        # 2) pi weights (cluster->class)
        # pi (cluster->class 权重): [B,K,C], 表示每个cluster属于每个类别的概率; 
        #          行归一化后不一定为1, 表示软权重; 用途: 加权InfoNCE损失, 强调重要类别的贡献
        try:
            pi = compute_pi_weights(assign_r, token_cls_mask_s)  # [B,K,C]
        except RuntimeError as e:
            logger.error(f"[RPSA] Error in compute_pi_weights: {e}, assign_r.shape={assign_r.shape}, token_cls_mask_s.shape={token_cls_mask_s.shape}")
            raise

        # 3) weighted InfoNCE alignment
        try:
            pi_used = pi.detach() if self.detach_pi else pi
            adaptive_thresh = None
            if self.bg_percentile > 0.0:
                pi_max = pi.max(dim=-1).values  # [B,K]
                adaptive_thresh = torch.quantile(
                    pi_max,
                    q=min(max(self.bg_percentile, 1e-6), 1.0),
                    dim=1,
                    keepdim=True,
                )
            loss, stats = weighted_infoNCE(
                centers_mu,
                t,
                pi_used,
                tau=self.tau_align,
                alpha_pi=self.alpha_pi,
                bg_thresh=self.bg_thresh,
                adaptive_bg=adaptive_thresh,
            )
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite RPSA loss: {loss.detach().item()}")
        except RuntimeError as e:
            logger.error(f"[RPSA] Error in weighted_infoNCE: {e}, centers_mu.shape={centers_mu.shape}, t.shape={t.shape}, pi.shape={pi.shape}")
            raise

        # Additional diagnostics
        with torch.no_grad():
            # orthogonality of centers (normalized)
            mu_n = l2_normalize(centers_mu, dim=-1)
            gram = torch.einsum('bkd,bmd->bkm', mu_n, mu_n)   # [B,K,K]
            I = torch.eye(self.K, device=gram.device, dtype=gram.dtype)[None]
            orth_mse = ((gram - I)**2).mean()
            # pi entropy
            pi_row = pi / (pi.sum(dim=-1, keepdim=True).clamp_min(1e-6))
            pi_entropy = (-(pi_row.clamp_min(1e-8).log() * pi_row).sum(dim=-1) / math.log(pi_row.size(-1))).mean()

        stats.update({
            "loss_rpsa": loss.detach(),
            "rpsa_center_orth_mse": orth_mse.detach(),
            "rpsa_pi_entropy": pi_entropy.detach(),
        })

        extras = {
            "assign_r": assign_r.detach(),
            "centers_mu": centers_mu.detach(),
            "pi": pi.detach(),
        }

        return loss, stats, extras


# -------------------------------
# Pseudo-label helper (optional)
# -------------------------------


def build_token_class_mask_from_logits(enc_outputs_class: torch.Tensor,
                                       topL: int = 5,
                                       prob_thresh: float = 0.0) -> torch.Tensor:
    """
    Build soft token->class mask from encoder classification logits.
    enc_outputs_class: [B, N, C] (logits or probs). If logits, we apply softmax.
    Return:
        token_cls_mask: [B, N, C] with zeros except top-L (or >thresh) classes per token.
    """
    if enc_outputs_class.dim() != 3:
        raise ValueError("enc_outputs_class must be [B,N,C]")

    # Convert to probabilities if logits
    probs = torch.softmax(enc_outputs_class, dim=-1)

    if topL > 0:
        topk = torch.topk(probs, k=min(topL, probs.size(-1)), dim=-1)
        mask = torch.zeros_like(probs)
        mask.scatter_(-1, topk.indices, topk.values)  # place probs at top-L positions
        token_cls_mask = mask
    else:
        token_cls_mask = probs * (probs >= prob_thresh).float()

    return token_cls_mask
