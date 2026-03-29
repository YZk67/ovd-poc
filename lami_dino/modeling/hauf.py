"""HAUF: Hybrid Attribute-Uncertainty Fusion for unknown object detection.

Combines attribute similarity scores with known-class uncertainty to
identify unknown objects at inference time.

Reference: OW-OVD (CVPR 2025), adapted for DETR architecture.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class HAUF(nn.Module):
    """Hybrid Attribute-Uncertainty Fusion module.

    At inference, computes an unknown score for each detection proposal:
        unknown_score = (top_k_att_score + uncertainty) / 2 * (1 - max_known_score)

    Args:
        top_k: Number of top attribute scores to aggregate.
        unknown_threshold: Score threshold above which a proposal is classified as unknown.
    """

    def __init__(self, top_k=10, unknown_threshold=0.5, att_weight=0.5, objectness_threshold=0.05):
        super().__init__()
        self.top_k = top_k
        self.unknown_threshold = unknown_threshold
        self.att_weight = att_weight  # weight for attribute score vs uncertainty
        self.objectness_threshold = objectness_threshold  # min known score to be considered an object

    def compute_attribute_scores(self, roi_features, att_embeddings, temperature=100.0):
        """Compute attribute similarity scores for each proposal.

        Args:
            roi_features: [bs, num_proposals, feat_dim] - region features from backbone.
            att_embeddings: [num_attrs, feat_dim] - VSAS-selected attribute embeddings.
            temperature: Scaling factor for cosine similarity.

        Returns:
            att_scores: [bs, num_proposals, num_attrs] - sigmoid of scaled cosine similarity.
        """
        roi_norm = F.normalize(roi_features, p=2, dim=-1)
        att_norm = F.normalize(att_embeddings, p=2, dim=-1)
        # [bs, num_proposals, num_attrs]
        sim = torch.matmul(roi_norm, att_norm.t()) * temperature
        return sim.sigmoid()

    def compute_weighted_top_k(self, att_scores, k=None):
        """Aggregate top-k attribute scores with softmax weighting.

        Args:
            att_scores: [bs, num_proposals, num_attrs]

        Returns:
            weighted_avg: [bs, num_proposals, 1]
        """
        k = k or self.top_k
        k = min(k, att_scores.shape[-1])
        top_k_scores, _ = att_scores.topk(k, dim=-1)
        top_k_weights = F.softmax(top_k_scores, dim=-1)
        weighted_avg = (top_k_scores * top_k_weights).sum(dim=-1, keepdim=True)
        return weighted_avg

    def compute_uncertainty(self, known_logits):
        """Compute binary entropy uncertainty from sigmoid logits.

        Args:
            known_logits: [bs, num_proposals, num_known_classes] - sigmoid activated.

        Returns:
            uncertainty: [bs, num_proposals, 1] - mean binary entropy across classes.
        """
        p = known_logits.clamp(1e-6, 1 - 1e-6)
        entropy = -p * p.log() - (1 - p) * (1 - p).log()
        return entropy.mean(dim=-1, keepdim=True)

    def forward(self, known_cls_scores, roi_features, att_embeddings, temperature=100.0):
        """Compute unknown scores for each proposal.

        Args:
            known_cls_scores: [bs, num_proposals, num_classes] - sigmoid class scores.
            roi_features: [bs, num_proposals, feat_dim] - CLIP region features.
            att_embeddings: [num_attrs, feat_dim] - VSAS attribute embeddings.
            temperature: Cosine similarity temperature.

        Returns:
            unknown_scores: [bs, num_proposals, 1] - unknown confidence.
            combined_scores: [bs, num_proposals, num_classes + 1] - known + unknown logits.
        """
        # Attribute similarity
        att_scores = self.compute_attribute_scores(roi_features, att_embeddings, temperature)
        top_k_att = self.compute_weighted_top_k(att_scores)

        # Known-class uncertainty
        uncertainty = self.compute_uncertainty(known_cls_scores)

        # Max known confidence (inverse = unknown prior)
        max_known, _ = known_cls_scores.max(dim=-1, keepdim=True)

        # HAUF fusion (att_weight controls attribute vs uncertainty balance)
        w = self.att_weight
        unknown_scores = (w * top_k_att + (1 - w) * uncertainty) * (1.0 - max_known)

        # Gate: proposals with very low max_known are likely background, not unknown objects.
        # An unknown object should still trigger *some* known-class activation.
        objectness_gate = (max_known >= self.objectness_threshold).float()
        unknown_scores = unknown_scores * objectness_gate

        # Concatenate known + unknown
        combined_scores = torch.cat([known_cls_scores, unknown_scores], dim=-1)

        return unknown_scores, combined_scores


class VSASDistributionLogger:
    """Logs attribute score distributions during training for VSAS selection.

    Tracks histograms of attribute scores for positive (matched to GT) and
    negative (unmatched) proposals. Used offline by select_attributes_vsas.py.

    Following OW-OVD's log_distribution() pattern.
    """

    def __init__(self, num_attrs, thresholds=(0.3, 0.5, 0.7), num_bins=10000):
        self.num_attrs = num_attrs
        self.thresholds = thresholds
        self.num_bins = num_bins
        self.reset()

    def reset(self):
        """Reset distribution accumulators."""
        self.positive_distributions = [
            [torch.zeros(self.num_bins) for _ in range(self.num_attrs)]
            for _ in self.thresholds
        ]
        self.negative_distributions = [
            [torch.zeros(self.num_bins) for _ in range(self.num_attrs)]
            for _ in self.thresholds
        ]

    @torch.no_grad()
    def log(self, att_scores, assigned_scores, prev_num_classes=0):
        """Log attribute score distributions.

        Args:
            att_scores: [N, num_attrs] - sigmoid attribute scores for all proposals.
            assigned_scores: [N, num_classes] - assignment scores from matcher
                (>0 for matched proposals, 0 for unmatched).
            prev_num_classes: Number of classes from previous tasks (for incremental).
        """
        att_scores = att_scores.float().detach()
        assigned_scores = assigned_scores.float().detach()

        # Zero out previous task classes
        if prev_num_classes > 0:
            assigned_scores[:, :prev_num_classes] = 0

        # Max assignment score per proposal
        max_assigned = assigned_scores.max(dim=-1)[0]

        for thr_idx, thr in enumerate(self.thresholds):
            positive_mask = max_assigned >= thr
            pos_scores = att_scores[positive_mask]
            neg_scores = att_scores[~positive_mask]

            for att_i in range(self.num_attrs):
                if pos_scores.numel() > 0:
                    self.positive_distributions[thr_idx][att_i] += torch.histc(
                        pos_scores[:, att_i].cpu(), bins=self.num_bins, min=0, max=1
                    )
                if neg_scores.numel() > 0:
                    self.negative_distributions[thr_idx][att_i] += torch.histc(
                        neg_scores[:, att_i].cpu(), bins=self.num_bins, min=0, max=1
                    )

    def save(self, path):
        """Save distributions for offline VSAS selection."""
        torch.save(
            {
                "positive_distributions": self.positive_distributions,
                "negative_distributions": self.negative_distributions,
                "thresholds": self.thresholds,
                "num_attrs": self.num_attrs,
            },
            path,
        )
        print(f"Saved distributions to {path}")
