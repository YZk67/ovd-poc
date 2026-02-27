# coding=utf-8
# Copyright 2022 The IDEA Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn.functional as F

from detrex.modeling.criterion import SetCriterion
from detrex.layers import box_cxcywh_to_xyxy, generalized_box_iou
from detrex.utils import get_world_size, is_dist_avail_and_initialized


class TwoStageCriterion(SetCriterion):
    def __init__(
        self,
        num_classes,
        matcher,
        weight_dict,
        losses=["class", "boxes"],
        eos_coef=None,
        loss_class_type="focal_loss",
        alpha: float = 0.25,
        gamma: float = 2,
        two_stage_binary_cls=False,
        use_pseudo_weight: bool = True,
        min_pseudo_weight: float = 0.0,
        pseudo_weight_power: float = 1.0,
    ):
        super().__init__(
            num_classes, matcher, weight_dict, losses, eos_coef, loss_class_type, alpha, gamma,
        )
        self.two_stage_binary_cls = two_stage_binary_cls
        self.use_pseudo_weight = use_pseudo_weight
        self.min_pseudo_weight = min_pseudo_weight
        self.pseudo_weight_power = pseudo_weight_power

    def _get_matched_weights(self, targets, indices, device):
        matched_weights = []
        for target, (_, tgt_idx) in zip(targets, indices):
            if tgt_idx.numel() == 0:
                continue
            weights = torch.ones_like(tgt_idx, dtype=torch.float32, device=device)
            if not self.use_pseudo_weight:
                matched_weights.append(weights)
                continue

            if "pseudo_mask" in target:
                pseudo_mask = target["pseudo_mask"].to(torch.bool).to(device)
            elif "scores" in target:
                pseudo_thr = float(target.get("pseudo_score_thr", 1.0))
                pseudo_mask = target["scores"].to(device=device, dtype=torch.float32) < pseudo_thr
            else:
                pseudo_mask = torch.zeros_like(target["labels"], dtype=torch.bool, device=device)

            if "pseudo_weight" in target:
                pseudo_weight = target["pseudo_weight"].to(device=device, dtype=torch.float32)
            elif "scores" in target:
                pseudo_weight = target["scores"].to(device=device, dtype=torch.float32)
            else:
                pseudo_weight = torch.ones_like(target["labels"], dtype=torch.float32, device=device)

            pseudo_weight = pseudo_weight.clamp(min=self.min_pseudo_weight, max=1.0)
            if self.pseudo_weight_power != 1.0:
                pseudo_weight = pseudo_weight.pow(self.pseudo_weight_power)

            pseudo_on_matched = pseudo_mask[tgt_idx]
            weights[pseudo_on_matched] = pseudo_weight[tgt_idx][pseudo_on_matched]
            matched_weights.append(weights)

        if len(matched_weights) == 0:
            return torch.zeros(0, device=device, dtype=torch.float32)
        return torch.cat(matched_weights, dim=0)

    def loss_labels(self, outputs, targets, indices, num_boxes):
        assert "pred_logits" in outputs
        src_logits = outputs["pred_logits"]
        num_classes = src_logits.shape[2]
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(
            src_logits.shape[:2],
            num_classes,
            dtype=torch.int64,
            device=src_logits.device,
        )
        target_classes[idx] = target_classes_o

        if self.loss_class_type == "ce_loss":
            loss_class = F.cross_entropy(
                src_logits.transpose(1, 2), target_classes, self.empty_weight
            )
            return {"loss_class": loss_class}
        elif self.loss_class_type == "contrastive_loss":
            raise NotImplementedError("contrastive_loss is not implemented in TwoStageCriterion")

        target_classes_onehot = torch.zeros(
            [src_logits.shape[0], src_logits.shape[1], src_logits.shape[2] + 1],
            dtype=src_logits.dtype,
            layout=src_logits.layout,
            device=src_logits.device,
        )
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)
        target_classes_onehot = target_classes_onehot[:, :, :-1]

        prob = src_logits.sigmoid()
        ce_loss = F.binary_cross_entropy_with_logits(src_logits, target_classes_onehot, reduction="none")
        p_t = prob * target_classes_onehot + (1 - prob) * (1 - target_classes_onehot)
        loss = ce_loss * ((1 - p_t) ** self.gamma)
        if self.alpha >= 0:
            alpha_t = self.alpha * target_classes_onehot + (1 - self.alpha) * (1 - target_classes_onehot)
            loss = alpha_t * loss

        query_weights = torch.ones(
            src_logits.shape[:2], dtype=src_logits.dtype, device=src_logits.device
        )
        matched_weights = self._get_matched_weights(targets, indices, src_logits.device).to(src_logits.dtype)
        if matched_weights.numel() > 0:
            query_weights[idx] = matched_weights

        loss_class = (loss * query_weights.unsqueeze(-1)).sum() / num_boxes
        return {"loss_class": loss_class}

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        assert "pred_boxes" in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs["pred_boxes"][idx]
        target_boxes = torch.cat([t["boxes"][i] for t, (_, i) in zip(targets, indices)], dim=0)
        matched_weights = self._get_matched_weights(targets, indices, src_boxes.device).to(src_boxes.dtype)
        if matched_weights.numel() == 0:
            matched_weights = torch.ones((0,), dtype=src_boxes.dtype, device=src_boxes.device)

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction="none")
        loss_bbox = (loss_bbox.sum(dim=-1) * matched_weights).sum() / num_boxes

        loss_giou = 1 - torch.diag(
            generalized_box_iou(
                box_cxcywh_to_xyxy(src_boxes),
                box_cxcywh_to_xyxy(target_boxes),
            )
        )
        loss_giou = (loss_giou * matched_weights).sum() / num_boxes
        return {"loss_bbox": loss_bbox, "loss_giou": loss_giou}

    def forward(self, outputs, targets):
        """This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc

             return_indices: used for vis. if True, the layer0-5 indices will be returned as well.

        """

        outputs_without_aux = {k: v for k, v in outputs.items() if k != "aux_outputs"}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor(
            [num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device
        )
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if "aux_outputs" in outputs:
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes)
                    l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        # for two stage
        if "enc_outputs" in outputs:
            enc_outputs = outputs["enc_outputs"]
            if self.two_stage_binary_cls:
                for bt in targets:
                    bt["labels"] = torch.zeros_like(bt["labels"])
            indices = self.matcher(enc_outputs, targets)
            for loss in self.losses:
                l_dict = self.get_loss(loss, enc_outputs, targets, indices, num_boxes)
                l_dict = {k + "_enc": v for k, v in l_dict.items()}
                losses.update(l_dict)

        return losses
