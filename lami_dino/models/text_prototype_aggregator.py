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
    - APR: orthogonality + diversity
    - 健康的监控指标: orth(off-diag MSE), usage entropy(soft/hard投票)
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

        # 正则权重（基础值）
        self.lambda_orth_base = float(lambda_orth)
        self.lambda_div_base  = float(lambda_div)
        self.warmup_steps = int(warmup_steps)

        # 投影与查询
        self.key_proj = nn.Linear(dim, hidden_dim)
        self.value_proj = nn.Linear(dim, dim)
        self.prototype_queries = nn.Parameter(torch.empty(num_prototypes, hidden_dim))
        self.dropout = nn.Dropout(dropout)

        # 缓存 & 日志
        self.register_buffer("_eye_buffer", torch.eye(num_prototypes, dtype=torch.float32), persistent=False)
        self._logger = logging.getLogger("lami_dino.tpa")
        self.log_interval = int(log_interval)
        self.entropy_mode = entropy_mode
        self._step = 0

        # 监控缓存（用于外部记录）
        self.last_loss_terms: Dict[str, float] = {}
        self._last_attention: Optional[torch.Tensor] = None    # [C,K,N]
        self._last_prototypes: Optional[torch.Tensor] = None   # [C,K,D]（缓存于dropout之前）
        self.last_monitor_terms: Dict[str, float] = {}

        self._reset_parameters()

    # ------------------------ 初始化 ------------------------
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

        # 投影
        keys = self.key_proj(text_feats)      # [C, N, H]
        values = self.value_proj(text_feats)  # [C, N, D]

        # 注意力: 每个prototype在短语集合(N)上的分布
        logits = torch.einsum("kh,cnh->ckn", self.prototype_queries, keys)  # [C,K,N]
        logits = logits / math.sqrt(self.key_proj.out_features)
        attn = F.softmax(logits / self.tau, dim=-1)  # [C,K,N]，dim=-1是短语维

        # 原型聚合（用于训练的输出在dropout之后；监控用dropout之前）
        prototypes_clean = torch.einsum("ckn,cnd->ckd", attn, values)  # [C,K,D]
        prototypes = self.dropout(prototypes_clean)

        # 缓存监控用张量（避免被dropout噪声影响）
        self._last_attention = attn.detach()
        self._last_prototypes = prototypes_clean.detach()

        # 正则loss
        apr_loss = None
        if with_loss:
            apr_loss = self.compute_apr_loss(prototypes_clean, attn)  # 用clean版本做几何正则更稳
            apr_value = apr_loss.detach()
        else:
            apr_value = self._update_metrics_no_grad(prototypes_clean.detach(), attn.detach())

        self._maybe_log(apr_value)
        return prototypes, apr_loss

    # ------------------------ APR ------------------------
    def _effective_lambdas(self) -> Tuple[float, float]:
        """线性warmup（若启用）"""
        if self.warmup_steps <= 0:
            return self.lambda_orth_base, self.lambda_div_base
        factor = min(1.0, float(self._step + 1) / float(self.warmup_steps))
        return self.lambda_orth_base * factor, self.lambda_div_base * factor

    def compute_apr_loss(self, prototypes: torch.Tensor, attn: torch.Tensor) -> torch.Tensor:
        """
        Adaptive Prototype Regularization:
        - Orthogonality（原型互相正交）
        - Diversity（原型使用多样性）
        """
        loss_orth = self._orthogonality_term(prototypes)
        loss_div  = self._diversity_term(attn, mode=self.entropy_mode)

        lam_orth, lam_div = self._effective_lambdas()
        apr_loss = lam_orth * loss_orth + lam_div * loss_div

        self._store_loss_terms(loss_orth.detach(), loss_div.detach(), apr_loss.detach(), lam_orth, lam_div)
        return apr_loss

    # ------------------------ 正则细项 ------------------------
    def _orthogonality_term(self, prototypes: torch.Tensor) -> torch.Tensor:
        """
        计算 off-diagonal MSE（越小越正交）
        """
        P = F.normalize(prototypes, dim=-1)
        G = torch.einsum("ckd,cmd->ckm", P, P)   # [C,K,K]
        K = G.size(-1)

        # I可能需要搬设备
        I = self._eye_buffer[:K, :K]
        if I.device != G.device:
            I = I.to(G.device)

        # 只取非对角项
        off_mask = (1.0 - torch.eye(K, device=G.device))
        off = ((G - I) ** 2 * off_mask).sum(dim=(-2, -1)) / (K * K - K)
        loss_off = off.mean()
        return loss_off

    def _diversity_term(self, attn: torch.Tensor, mode: str = "soft") -> torch.Tensor:
        """
        使用熵作为多样性：衡量K个原型的“使用均衡度”
        正确做法：以“每个短语在K上的分配”为基本票，再跨N聚合
        """
        C, K, N = attn.shape
        if mode == "hard":
            # 每个短语把票投给概率最大的原型（每类独立）
            winners = attn.argmax(dim=1)                 # [C, N]
            votes = torch.zeros(C, K, device=attn.device, dtype=attn.dtype)
            votes.scatter_add_(1, winners, torch.ones_like(winners, dtype=attn.dtype))
        else:
            # soft票数：先在K上归一，再跨N累加
            w = attn.clamp_min(1e-8)
            w = w / w.sum(dim=1, keepdim=True)          # [C,K,N], sum_K=1
            votes = w.sum(dim=-1)                        # [C,K], sum_K=N

        p = votes / (votes.sum(dim=1, keepdim=True) + 1e-8)   # [C,K]
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

    def _update_metrics_no_grad(self, prototypes: torch.Tensor, attn: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            loss_orth = self._orthogonality_term(prototypes)
            loss_div  = self._diversity_term(attn, mode=self.entropy_mode)
            lam_orth, lam_div = self._effective_lambdas()
            apr_loss = lam_orth * loss_orth + lam_div * loss_div
        self._store_loss_terms(loss_orth, loss_div, apr_loss, lam_orth, lam_div)
        return apr_loss.detach()

    def _maybe_log(self, apr_value: torch.Tensor) -> None:
        # 仅主进程打印；按间隔
        self._step += 1
        if self._logger is None or (self.training and self._step % self.log_interval != 0):
            return
        if not _is_main_process():
            return

        # 计算更稳健的监控指标（对角与非对角分离）
        monitor = monitor_prototype_metrics(
            self._last_prototypes, self._last_attention,
            step=self._step, log_interval=self.log_interval, prefix="[TPA]", entropy_mode=self.entropy_mode
        )
        if monitor:
            self.last_monitor_terms.update(monitor)

        # 组装消息
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

    # 供Trainer/Logger直接读取
    def get_monitor_dict(self) -> Dict[str, float]:
        return {
            **self.last_loss_terms,
            **self.last_monitor_terms,
        }


# ======================= 度量函数（修正版） =======================
@torch.no_grad()
def compute_prototype_orthogonality(prototypes: Optional[torch.Tensor]) -> Tuple[float, float]:
    """
    返回: (off-diagonal MSE, diag MSE)
    - off-diagonal 越小越正交
    - diag MSE 反映对角是否接近1（应接近0）
    """
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
def compute_usage_entropy(attn: Optional[torch.Tensor], mode: str = "soft") -> float:
    """
    正确的“原型使用度”熵（见类内实现说明）
    """
    if attn is None:
        return float("nan")
    C, K, N = attn.shape
    if mode == "hard":
        winners = attn.argmax(dim=1)                 # [C, N]
        votes = torch.zeros(C, K, device=attn.device, dtype=attn.dtype)
        votes.scatter_add_(1, winners, torch.ones_like(winners, dtype=attn.dtype))
    else:
        w = attn.clamp_min(1e-8)
        w = w / w.sum(dim=1, keepdim=True)          # [C,K,N]
        votes = w.sum(dim=-1)                        # [C,K]

    p = votes / (votes.sum(dim=1, keepdim=True) + 1e-8)
    entropy = -(p * (p.clamp_min(1e-8)).log()).sum(dim=1) / math.log(K)
    return float(entropy.mean().item())


def monitor_prototype_metrics(
    prototypes: Optional[torch.Tensor],
    attn: Optional[torch.Tensor],
    step: int = 0,
    log_interval: int = 200,
    prefix: str = "[Monitor]",
    *,
    entropy_mode: str = "soft",
) -> Dict[str, float]:
    off_mse, diag_mse = compute_prototype_orthogonality(prototypes)
    usage_entropy = compute_usage_entropy(attn, mode=entropy_mode)

    if step % log_interval == 0 and _is_main_process():
        print(f"{prefix} step={step:06d} | orth_off={off_mse:.4f} | diag_mse={diag_mse:.4f} | usage_entropy={usage_entropy:.4f}")

    return {
        "orthogonality": off_mse,     # 为了兼容旧字段名，仍沿用 'orthogonality'
        "orth_off_mse": off_mse,
        "diag_mse": diag_mse,
        "usage_entropy": usage_entropy,
    }


# ======================= PrototypeBank (原样 + 小修) =======================
class TextPrototypeBank(nn.Module):
    """
    管理/缓存文本短语嵌入并产出语义原型
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
            or self._cached_prototypes.device != text_feats.device
        ):
            with torch.no_grad():
                prototypes, apr_loss = self.aggregator(text_feats, with_loss=False)
            self._cached_prototypes = prototypes
            self._cached_apr_loss = apr_loss
            self._cached_step = step

        return self._cached_prototypes, self._cached_apr_loss
