import torch
import torch.nn as nn
from torch import Tensor


class PlainMarginLoss(nn.Module):
    """Pairwise margin loss for confusable class discrimination.

    For each matched query r with GT class c+, penalizes when the score for a
    confusable negative class c- is within `margin` of the positive score:
        L = (1/K) * sum_k max(0, margin + S(r, c_k-) - S(r, c+))

    Args:
        margin (float): Margin value. Defaults to 0.2.
    """

    def __init__(self, margin: float = 0.2):
        super().__init__()
        self.margin = margin

    def forward(self, vis_scores_pos: Tensor,
                vis_scores_neg: Tensor) -> Tensor:
        """Compute margin loss.

        Args:
            vis_scores_pos (Tensor): Shape [N] — score of each matched query
                for its GT class.
            vis_scores_neg (Tensor): Shape [N, K] — scores of each matched
                query for its K confusable negative classes.

        Returns:
            Tensor: Scalar loss value.
        """
        if vis_scores_pos.numel() == 0:
            return vis_scores_pos.new_zeros(1).squeeze()

        # [N, 1] - [N, K] → [N, K]
        margins = self.margin + vis_scores_neg - vis_scores_pos.unsqueeze(1)
        loss = torch.clamp(margins, min=0).mean()
        return loss
