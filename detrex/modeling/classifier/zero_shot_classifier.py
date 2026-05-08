# Copyright (c) Facebook, Inc. and its affiliates.
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from detectron2.config import configurable
from detectron2.layers import Linear, ShapeSpec

class ZeroShotClassifier(nn.Module):
    @configurable
    def __init__(
        self,
        input_shape: ShapeSpec,
        *,
        num_classes: int,
        zs_weight_path: str,
        eval_zs_weight_path: str,
        zs_weight_dim: int = 512,
        use_bias: float = 0.0, 
        norm_weight: bool = True,
        norm_temperature: float = 50.0,
        multi_prototype_score_agg: str = "logsumexp",
    ):
        super().__init__()
        if isinstance(input_shape, int):  # some backward compatibility
            input_shape = ShapeSpec(channels=input_shape)
        input_size = input_shape.channels * (input_shape.width or 1) * (input_shape.height or 1)
        self.norm_weight = norm_weight
        self.norm_temperature = norm_temperature
        self.static_multi_prototype = False
        self.multi_prototype_score_agg = multi_prototype_score_agg
        valid_aggs = {"logsumexp", "mean", "max"}
        if self.multi_prototype_score_agg not in valid_aggs:
            raise ValueError(
                "multi_prototype_score_agg must be one of "
                f"{sorted(valid_aggs)}, got {self.multi_prototype_score_agg!r}."
            )

        self.use_bias = use_bias < 0
        if self.use_bias:
            self.cls_bias = nn.Parameter(torch.ones(1) * use_bias)

        self.linear = nn.Linear(input_size, zs_weight_dim)
        
        if zs_weight_path == 'rand':
            zs_weight = torch.randn((zs_weight_dim, num_classes))
            nn.init.normal_(zs_weight, std=0.01)
        else:
            zs_weight, static_multi = self._load_static_embeddings(zs_weight_path)
            eval_zs_weight, eval_static_multi = self._load_static_embeddings(eval_zs_weight_path)
            if static_multi != eval_static_multi:
                raise ValueError(
                    "Train and eval zero-shot embeddings must both be single- or multi-prototype, "
                    f"got {zs_weight.shape} and {eval_zs_weight.shape}."
                )
            self.static_multi_prototype = static_multi
            feat_dim = zs_weight.shape[-1] if self.static_multi_prototype else zs_weight.shape[0]
            if feat_dim != zs_weight_dim:
                self.linear = nn.Linear(input_size, feat_dim)
                zs_weight_dim = feat_dim
 
        """
        zs_weight = torch.cat(
            [zs_weight, zs_weight.new_zeros((zs_weight_dim, 1))], 
            dim=1) # D x (C + 1)
        """
        if self.norm_weight:
            if self.static_multi_prototype:
                zs_weight = F.normalize(zs_weight, p=2, dim=-1)
                eval_zs_weight = F.normalize(eval_zs_weight, p=2, dim=-1)
            else:
                zs_weight = F.normalize(zs_weight, p=2, dim=0)
                eval_zs_weight = F.normalize(eval_zs_weight, p=2, dim=0)
        if zs_weight_path == 'rand':
            self.zs_weight = nn.Parameter(zs_weight)
        else:
            self.register_buffer('zs_weight', zs_weight)
            self.register_buffer('eval_zs_weight', eval_zs_weight)

        # assert self.zs_weight.shape[1] == num_classes + 1, self.zs_weight.shape


    @classmethod
    def from_config(cls, cfg, input_shape):
        return {
            'input_shape': input_shape,
            'num_classes': cfg.MODEL.ROI_HEADS.NUM_CLASSES,
            'zs_weight_path': cfg.MODEL.ROI_BOX_HEAD.ZEROSHOT_WEIGHT_PATH,
            'zs_weight_dim': cfg.MODEL.ROI_BOX_HEAD.ZEROSHOT_WEIGHT_DIM,
            'use_bias': cfg.MODEL.ROI_BOX_HEAD.USE_BIAS,
            'norm_weight': cfg.MODEL.ROI_BOX_HEAD.NORM_WEIGHT,
            'norm_temperature': cfg.MODEL.ROI_BOX_HEAD.NORM_TEMP,
        }

    @staticmethod
    def _load_static_embeddings(path: str):
        arr = np.load(path)
        if arr.ndim == 2:
            return torch.tensor(arr, dtype=torch.float32).permute(1, 0).contiguous(), False
        if arr.ndim == 3:
            return torch.tensor(arr, dtype=torch.float32).contiguous(), True
        raise ValueError(f"Expected zero-shot embeddings with 2 or 3 dims, got shape {arr.shape}.")

    def _normalize_features(self, x):
        if self.norm_weight:
            return self.norm_temperature * F.normalize(x, p=2, dim=-1)
        return x

    def _aggregate_multi_prototype_logits(self, logits):
        if self.multi_prototype_score_agg == "mean":
            return logits.mean(dim=-1)
        if self.multi_prototype_score_agg == "max":
            return logits.max(dim=-1).values
        return torch.logsumexp(logits, dim=-1)

    def forward(self, x, classifier=None, content_inds=None, additional_class=None):
        '''
        Inputs:
            x: B x D'
            classifier_info: (C', C' x D)
        '''
        features = self._normalize_features(self.linear(x))
        if classifier is not None:
            zs_weight = classifier.permute(1, 0).contiguous() # D x C'
            zs_weight = F.normalize(zs_weight, p=2, dim=0) \
                if self.norm_weight else zs_weight
        else:
            if self.training:
                if self.static_multi_prototype:
                    zs_weight = self.zs_weight[content_inds] if content_inds is not None else self.zs_weight
                else:
                    zs_weight = self.zs_weight[:, content_inds] if content_inds is not None else self.zs_weight
            else:
                zs_weight = self.eval_zs_weight
                if self.static_multi_prototype and content_inds is not None:
                    zs_weight = zs_weight[content_inds]

        if self.static_multi_prototype and classifier is None:
            logits = torch.einsum("bqd,ckd->bqck", features, zs_weight)
            logits = self._aggregate_multi_prototype_logits(logits)
            if additional_class is not None:
                additional = additional_class.to(device=features.device, dtype=features.dtype)
                if self.norm_weight:
                    additional = F.normalize(additional, p=2, dim=-1)
                additional_logits = torch.einsum("bqd,nd->bqn", features, additional)
                logits = torch.cat([logits, additional_logits], dim=-1)
            x = logits
        else:
            if additional_class is not None:
                additional_zs_weight = additional_class.t()
                if self.norm_weight:
                    additional_zs_weight = F.normalize(additional_zs_weight, p=2, dim=0)
                zs_weight = torch.cat([zs_weight, additional_zs_weight], dim=1)
            x = torch.matmul(features, zs_weight)
        if self.use_bias:
            x = x + self.cls_bias
        return x
