
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
import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from detectron2.utils.logger import setup_logger

logger = setup_logger()


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
        mass = r.sum(dim=1, keepdim=True).clamp_min(eps)        # [B,1,K]
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


# -------------------------------
# Weighted InfoNCE
# -------------------------------

def weighted_infoNCE(mu: torch.Tensor,
                     text_protos: torch.Tensor,
                     pi: torch.Tensor,
                     tau: float = 0.07,
                     alpha_pi: float = 1.0,
                     bg_thresh: float = 0.1) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
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
    bg_mask = (pi_max_raw < bg_thresh)                              # [B,K]
    pi_tilde = (pi_clamped ** alpha_pi)
    pi_tilde = pi_tilde / (pi_tilde.sum(dim=-1, keepdim=True).clamp_min(1e-6))  # [B,K,C]

    # logits
    pos = torch.logsumexp(S / tau, dim=-1)                          # [B,K,C] over Kp
    all_ = torch.logsumexp(S.view(B, K, -1) / tau, dim=-1, keepdim=True)  # [B,K,1]

    # weighted InfoNCE per cluster
    loss_k = -(pi_tilde * (pos - all_)).sum(dim=-1)                 # [B,K]
    loss_k = loss_k.masked_fill(bg_mask, 0.0)                       # ignore BG

    valid = (~bg_mask).float().sum().clamp_min(1.0)
    loss = loss_k.sum() / valid

    stats = {
        "rpsa_pos_mean": pos.mean().detach(),
        "rpsa_bg_ratio": (bg_mask.float().mean().detach()),
        "rpsa_valid_clusters": valid.detach(),
    }
    return loss, stats


class RPSAModule(nn.Module):
    def __init__(self,
                 K: int = 8, # number of visual clusters per image
                 sigma: float = 1.0,
                 em_iters: int = 1,
                 tau_align: float = 0.07,
                 alpha_pi: float = 1.0,
                 bg_thresh: float = 0.1,
                 subsample_tokens: int = 0,
                 stop_grad_text: bool = False,
                 stop_grad_vision: bool = False):
        """
        Args:
            K: number of visual clusters per image
            sigma: soft k-means temperature on squared distance
            em_iters: soft k-means EM iterations
            tau_align: InfoNCE temperature
            alpha_pi: exponent on pi weights (sharpen/soften)
            bg_thresh: background threshold on max pi per cluster
            subsample_tokens: if >0, randomly subsample N' tokens per image for RPSA (speed)
            stop_grad_text: if True, stop gradient to text prototypes branch
            stop_grad_vision: if True, stop gradient to region features branch
        """
        super().__init__()
        self.K = K
        self.sigma = float(sigma)
        self.em_iters = int(em_iters)
        self.tau_align = float(tau_align)
        self.alpha_pi = float(alpha_pi)
        self.bg_thresh = float(bg_thresh)
        self.subsample_tokens = int(subsample_tokens)
        self.stop_grad_text = bool(stop_grad_text)
        self.stop_grad_vision = bool(stop_grad_vision)

    def forward(self,
                region_feats: torch.Tensor,   # [B,N,D]
                text_protos: torch.Tensor,    # [C,Kp,D]
                token_cls_mask: torch.Tensor  # [B,N,C]  (GT or pseudo)
                ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:

        B, N, D = region_feats.shape

        # Optional subsampling for speed
        if (self.subsample_tokens > 0) and (N > self.subsample_tokens):
            idx = torch.randperm(N, device=region_feats.device)[:self.subsample_tokens]
            region_feats_s = region_feats[:, idx, :]
            token_cls_mask_s = token_cls_mask[:, idx, :]
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
            loss, stats = weighted_infoNCE(centers_mu, t, pi,
                                           tau=self.tau_align,
                                           alpha_pi=self.alpha_pi,
                                           bg_thresh=self.bg_thresh)
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
