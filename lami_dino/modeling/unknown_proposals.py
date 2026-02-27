import torch

from detrex.layers import box_cxcywh_to_xyxy
from detrex.layers.box_ops import box_iou


def _get_pseudo_mask(target, pseudo_score_thr=1.0):
    if "pseudo_mask" in target:
        return target["pseudo_mask"].to(torch.bool)
    if "scores" in target:
        thr = float(target.get("pseudo_score_thr", pseudo_score_thr))
        # Use a configurable score threshold for pseudo labels.
        return target["scores"] < thr
    return None


@torch.no_grad()
def get_unknown_proposal_match_info(region_proposals, targets, thr=0.5, pseudo_score_thr=1.0):
    """Return unknown mask and per-proposal max IoU against pseudo/unknown GT.

    Args:
        region_proposals (Tensor): [bs, nq, 4] normalized cxcywh in [0, 1].
        targets (list[dict]): target dicts with at least ``boxes`` and either
            ``pseudo_mask`` or ``scores`` (pseudo inferred as ``scores < pseudo_score_thr``).
        thr (float): IoU threshold.
    """
    bs, nq = region_proposals.shape[:2]
    device = region_proposals.device
    proposal_xyxy = box_cxcywh_to_xyxy(region_proposals.flatten(0, 1))

    pseudo_gt_boxes = []
    sizes = []
    for target in targets:
        pseudo_mask = _get_pseudo_mask(target, pseudo_score_thr=pseudo_score_thr)
        if pseudo_mask is None:
            sizes.append(0)
            continue
        boxes = target["boxes"][pseudo_mask]
        sizes.append(int(boxes.shape[0]))
        if boxes.numel() > 0:
            pseudo_gt_boxes.append(box_cxcywh_to_xyxy(boxes))

    if sum(sizes) == 0:
        empty_mask = torch.zeros(bs, nq, dtype=torch.bool, device=device)
        empty_iou = torch.zeros(bs, nq, dtype=region_proposals.dtype, device=device)
        return empty_mask, empty_iou

    pseudo_gt_boxes = torch.cat(pseudo_gt_boxes, dim=0)
    ious = box_iou(proposal_xyxy, pseudo_gt_boxes)[0].view(bs, nq, -1)

    masks = []
    max_ious = []
    for img_idx, iou_chunk in enumerate(ious.split(sizes, dim=-1)):
        if iou_chunk.numel() == 0:
            masks.append(torch.zeros(nq, dtype=torch.bool, device=device))
            max_ious.append(torch.zeros(nq, dtype=region_proposals.dtype, device=device))
            continue
        img_iou = iou_chunk[img_idx]
        img_max_iou = img_iou.max(dim=-1).values
        masks.append(img_max_iou > thr)
        max_ious.append(img_max_iou)
    return torch.stack(masks, dim=0), torch.stack(max_ious, dim=0)


@torch.no_grad()
def get_unknown_proposal_mask(region_proposals, targets, thr=0.5, pseudo_score_thr=1.0):
    mask, _ = get_unknown_proposal_match_info(
        region_proposals, targets, thr=thr, pseudo_score_thr=pseudo_score_thr
    )
    return mask


@torch.no_grad()
def select_unknown_proposals(region_proposals, targets, thr=0.5, pseudo_score_thr=1.0):
    """Return (mask, proposals_per_image) for unknown/pseudo-matched proposals."""
    mask = get_unknown_proposal_mask(
        region_proposals, targets, thr=thr, pseudo_score_thr=pseudo_score_thr
    )
    selected = [region_proposals[i][mask[i]] for i in range(region_proposals.shape[0])]
    return mask, selected
