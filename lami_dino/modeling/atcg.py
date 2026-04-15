"""
Analogical Textual Concept Generator (ATCG) for Open-Vocabulary Detection.

Inspired by AL-GCD (arXiv:2603.19918), this module generates pseudo text
embeddings for novel-class proposals via cross-attention over base-class
prototypes in CLIP feature space.

For each region proposal, ATCG:
1. Computes visual similarity to base-class visual prototypes (Keys)
2. Uses that similarity to retrieve a weighted combination of base-class
   text embeddings (Values)
3. Scores the resulting pseudo-text against all-class text embeddings
   to produce soft classification targets
4. Applies KL divergence loss between detector logits and soft targets
   on novel-class dimensions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AnalogicalPrototypeBank(nn.Module):
    """Maintains EMA visual prototypes per base class and generates
    analogical text embeddings via cross-attention.

    Args:
        num_base_classes: Number of base (frequent+common) classes.
        num_all_classes: Total number of classes (base + novel).
        feat_dim: CLIP feature dimension (768 for ConvNeXt-L).
        base_class_ids: List/tensor of base class indices in [0, num_all_classes).
        text_embeddings_path: Path to .npy of all-class CLIP text embeddings [C, D].
        temperature: Temperature for cross-attention softmax.
        momentum: EMA momentum for prototype updates (1.0 = no momentum, instant).
        loss_temperature: Temperature for KL divergence target softmax.
        novel_only: If True, only compute loss on novel-class dimensions.
        num_stacked_layers: Number of stacked refinement layers (0 = initial layer only).
    """

    def __init__(
        self,
        num_base_classes: int,
        num_all_classes: int,
        feat_dim: int = 768,
        base_class_ids=None,
        text_embeddings_path: str = None,
        temperature: float = 0.05,
        momentum: float = 0.999,
        loss_temperature: float = 0.1,
        novel_only: bool = True,
        num_stacked_layers: int = 0,
    ):
        super().__init__()
        self.num_base_classes = num_base_classes
        self.num_all_classes = num_all_classes
        self.feat_dim = feat_dim
        self.temperature = temperature
        self.momentum = momentum
        self.loss_temperature = loss_temperature
        self.novel_only = novel_only
        self.num_stacked_layers = num_stacked_layers

        # Base/novel class masks
        base_mask = torch.zeros(num_all_classes, dtype=torch.bool)
        if base_class_ids is not None:
            base_mask[base_class_ids] = True
        self.register_buffer("base_mask", base_mask)
        self.register_buffer("novel_mask", ~base_mask)

        # All-class text embeddings [C, D] — frozen reference
        if text_embeddings_path is not None:
            import numpy as np
            text_emb = np.load(text_embeddings_path)
            if text_emb.ndim == 3:
                # Multi-prototype: [C, K, D] → average to [C, D]
                text_emb = text_emb.mean(axis=1)
            text_emb = torch.tensor(text_emb, dtype=torch.float32)
            text_emb = F.normalize(text_emb, p=2, dim=-1)
        else:
            text_emb = torch.zeros(num_all_classes, feat_dim)
        self.register_buffer("all_text_embeddings", text_emb)  # [C, D]

        # Base-class text embeddings [num_base, D] — extracted view
        base_text = text_emb[base_mask]
        self.register_buffer("base_text_embeddings", base_text)  # [B, D]

        # EMA visual prototypes for base classes [num_base, D]
        # Initialize from text embeddings (reasonable in CLIP's aligned space)
        self.register_buffer("visual_prototypes", base_text.clone())
        self.register_buffer("prototype_counts", torch.zeros(num_base_classes))
        self.register_buffer("initialized", torch.zeros(num_base_classes, dtype=torch.bool))

        # Optional stacked refinement layers (TSA + TIAA from the paper)
        if num_stacked_layers > 0:
            self.stacked_layers = nn.ModuleList([
                StackedRefinementLayer(feat_dim) for _ in range(num_stacked_layers)
            ])
        else:
            self.stacked_layers = None

    @torch.no_grad()
    def update_prototypes(self, roi_features, class_labels):
        """Update EMA visual prototypes with new ROI features from GT boxes.

        Args:
            roi_features: [N, D] normalized CLIP ROI features for GT boxes.
            class_labels: [N] class indices (in all-class space).
        """
        if roi_features.numel() == 0:
            return

        # Build reverse mapping: all-class id → base-class index
        base_ids = self.base_mask.nonzero(as_tuple=True)[0]  # [num_base]
        all_to_base = torch.full((self.num_all_classes,), -1, device=roi_features.device)
        all_to_base[base_ids] = torch.arange(len(base_ids), device=roi_features.device)

        # Filter to base-class GT only
        base_indices = all_to_base[class_labels]  # [N], -1 for novel
        valid = base_indices >= 0
        if not valid.any():
            return

        feats = roi_features[valid]  # [M, D]
        bidx = base_indices[valid]   # [M]

        # EMA update per class
        for cls_idx in bidx.unique():
            mask = bidx == cls_idx
            avg_feat = F.normalize(feats[mask].mean(dim=0), dim=-1)
            if not self.initialized[cls_idx]:
                self.visual_prototypes[cls_idx] = avg_feat
                self.initialized[cls_idx] = True
            else:
                self.visual_prototypes[cls_idx] = (
                    self.momentum * self.visual_prototypes[cls_idx]
                    + (1 - self.momentum) * avg_feat
                )
                self.visual_prototypes[cls_idx] = F.normalize(
                    self.visual_prototypes[cls_idx], dim=-1
                )
            self.prototype_counts[cls_idx] += mask.sum()

    def generate_pseudo_text(self, roi_features):
        """Generate analogical text embeddings via cross-attention.

        Initial layer (Eq. 12 from AL-GCD):
            Q = roi_features (novel proposals)
            K = base visual prototypes
            V = base text embeddings
            pseudo_text = softmax(Q @ K^T / sqrt(d)) @ V

        Args:
            roi_features: [N, D] normalized CLIP ROI features.

        Returns:
            pseudo_text: [N, D] analogical text embeddings.
        """
        # Use initialized prototypes where available, text embeddings as fallback
        keys = self.visual_prototypes.clone()
        not_init = ~self.initialized
        if not_init.any():
            keys[not_init] = self.base_text_embeddings[not_init]

        # Initial layer: cross-attention (Eq. 12)
        # [N, D] @ [B, D]^T → [N, B]
        # temperature controls sharpness (lower = sharper attention)
        attn_logits = roi_features @ keys.t() / self.temperature
        attn_weights = F.softmax(attn_logits, dim=-1)  # [N, B]
        pseudo_text = attn_weights @ self.base_text_embeddings  # [N, D]
        pseudo_text = F.normalize(pseudo_text, dim=-1)

        # Optional stacked refinement layers
        if self.stacked_layers is not None:
            for layer in self.stacked_layers:
                pseudo_text = layer(pseudo_text, roi_features, keys, self.base_text_embeddings)
                pseudo_text = F.normalize(pseudo_text, dim=-1)

        return pseudo_text

    def compute_loss(self, detector_logits, roi_features):
        """Compute ATCG distillation loss.

        Args:
            detector_logits: [N, C] raw logits from the detector (before sigmoid).
            roi_features: [N, D] normalized CLIP ROI features.

        Returns:
            loss: Scalar KL divergence loss.
        """
        if roi_features.shape[0] == 0:
            return roi_features.new_tensor(0.0)

        # Generate pseudo text embeddings
        pseudo_text = self.generate_pseudo_text(roi_features)  # [N, D]

        # Score pseudo text against all-class text embeddings → soft targets
        # [N, D] @ [C, D]^T → [N, C]
        pseudo_logits = pseudo_text @ self.all_text_embeddings.t() / self.loss_temperature

        if self.novel_only:
            # Only supervise novel-class dimensions
            pseudo_logits = pseudo_logits[:, self.novel_mask]
            detector_logits = detector_logits[:, self.novel_mask]

        # Soft targets from pseudo logits
        pseudo_targets = F.softmax(pseudo_logits.detach(), dim=-1)  # [N, C'] stop gradient

        # Detector log-probabilities
        detector_log_probs = F.log_softmax(detector_logits, dim=-1)

        # KL divergence: KL(pseudo || detector)
        loss = F.kl_div(detector_log_probs, pseudo_targets, reduction="batchmean")

        return loss


class StackedRefinementLayer(nn.Module):
    """One stacked refinement layer: TSA (text self-attention) + TIAA
    (text-image analogical attention), following Section 3.5 of AL-GCD.
    """

    def __init__(self, feat_dim):
        super().__init__()
        self.norm1 = nn.LayerNorm(feat_dim)
        self.norm2 = nn.LayerNorm(feat_dim)
        self.ffn = nn.Sequential(
            nn.Linear(feat_dim, feat_dim * 2),
            nn.GELU(),
            nn.Linear(feat_dim * 2, feat_dim),
        )

    def forward(self, pseudo_text, roi_features, vis_protos, text_protos):
        """
        Args:
            pseudo_text: [N, D] current pseudo text embeddings.
            roi_features: [N, D] original ROI features.
            vis_protos: [B, D] base visual prototypes (keys).
            text_protos: [B, D] base text embeddings (values).

        Returns:
            refined_text: [N, D] refined pseudo text embeddings.
        """
        d = pseudo_text.shape[-1]

        # TSA: text self-attention (Eq. 13-14)
        # For single-vector pseudo_text, this is just identity + norm
        tsa_out = self.norm1(pseudo_text)

        # TIAA: text-image analogical attention (Eq. 15-18)
        # Q = concat(tsa_out, roi_features) → [N, 2D]
        query = torch.cat([tsa_out, roi_features], dim=-1)  # [N, 2D]
        # K = concat(text_protos, vis_protos) → [B, 2D]
        key = torch.cat([text_protos, vis_protos], dim=-1)  # [B, 2D]
        # V = text_protos → [B, D]

        attn_logits = query @ key.t() / (2 * d) ** 0.5  # [N, B]
        attn_weights = F.softmax(attn_logits, dim=-1)
        tiaa_out = attn_weights @ text_protos  # [N, D]

        # Residual + FFN
        refined = pseudo_text + tiaa_out
        refined = refined + self.ffn(self.norm2(refined))

        return refined
