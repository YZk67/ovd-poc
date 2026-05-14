import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from detectron2.layers import ShapeSpec
from typing import Optional, Tuple

from lami_dino.models import TextPrototypeAggregator


class TextClassifier(nn.Module):
    """
    Classification head that supports both static zero-shot weights and the
    learnable Text Prototype Aggregator (TPA).
    """

    def __init__(
        self,
        input_shape,
        *,
        num_classes: int,
        zs_weight_path: str,
        eval_zs_weight_path: str,
        zs_weight_dim: int = 512,
        use_bias: float = 0.0,
        norm_weight: bool = True,
        norm_temperature: float = 50.0,
        use_tpa: bool = False,
        text_embed_path: Optional[str] = None,
        eval_text_embed_path: Optional[str] = None,
        tpa_num_prototypes: int = 4,
        tpa_hidden_dim: int = 256,
        tpa_dropout: float = 0.1,
        tpa_tau: float = 0.1,
        tpa_log_interval: int = 200,
        static_multi_prototype_agg: str = "logsumexp",
    ) -> None:
        super().__init__()

        if isinstance(input_shape, int):
            input_shape = ShapeSpec(channels=input_shape)

        input_dim = input_shape.channels * (input_shape.width or 1) * (input_shape.height or 1)
        self.norm_weight = norm_weight
        self.norm_temperature = norm_temperature

        self.use_bias = use_bias < 0
        if self.use_bias:
            self.cls_bias = nn.Parameter(torch.ones(1) * use_bias)

        self.use_tpa = use_tpa
        self.num_classes = num_classes
        if static_multi_prototype_agg not in {"logsumexp", "max", "mean"}:
            raise ValueError(
                "static_multi_prototype_agg must be one of "
                f"'logsumexp', 'max', or 'mean', got {static_multi_prototype_agg!r}."
            )
        self.static_multi_prototype_agg = static_multi_prototype_agg

        if self.use_tpa:
            train_feats = self._load_text_embeddings(text_embed_path or zs_weight_path)
            eval_feats = self._load_text_embeddings(
                eval_text_embed_path or eval_zs_weight_path or text_embed_path or zs_weight_path
            )

            if train_feats.shape[-1] != eval_feats.shape[-1]:
                raise ValueError(
                    "Train and eval text embeddings must share the same feature dimension, "
                    f"got {train_feats.shape[-1]} vs {eval_feats.shape[-1]}."
                )

            if zs_weight_dim != train_feats.shape[-1]:
                # When using TPA we expect zs_weight_dim to match the text feature dimension.
                # If they differ we silently update so the linear layer projects to the correct dim.
                zs_weight_dim = train_feats.shape[-1]

            self.register_buffer("train_text_feats", train_feats, persistent=False)
            self.register_buffer("eval_text_feats", eval_feats, persistent=False)

            self.tpa = TextPrototypeAggregator(
                dim=train_feats.shape[-1],
                num_prototypes=tpa_num_prototypes,
                hidden_dim=tpa_hidden_dim,
                dropout=tpa_dropout,
                tau=tpa_tau,
                log_interval=tpa_log_interval,
            )
            self.num_prototypes = tpa_num_prototypes
            self._cached_eval = None
            self.zs_weight = None
            self.eval_zs_weight = None
            self.tpa.log_owner = True
            self._external_prototypes = None
            self._external_apr_loss = None
        else:
            self.static_multi_prototype = False
            if zs_weight_path == "rand":
                zs_weight = torch.randn((zs_weight_dim, num_classes))
                nn.init.normal_(zs_weight, std=0.01)
                self.zs_weight = nn.Parameter(zs_weight)
                self.eval_zs_weight = None
            else:
                zs_weight, static_multi = self._load_static_embeddings(zs_weight_path)
                eval_zs_weight, eval_static_multi = self._load_static_embeddings(eval_zs_weight_path)
                if static_multi != eval_static_multi:
                    raise ValueError(
                        "Train and eval static embeddings must both be single- or multi-prototype, "
                        f"got {zs_weight.shape} and {eval_zs_weight.shape}."
                    )
                train_dim = zs_weight.shape[-1] if static_multi else zs_weight.shape[0]
                eval_dim = eval_zs_weight.shape[-1] if eval_static_multi else eval_zs_weight.shape[0]
                if train_dim != eval_dim:
                    raise ValueError(
                        "Train and eval static embeddings must share the same feature dimension, "
                        f"got {train_dim} vs {eval_dim}."
                    )
                zs_weight_dim = train_dim
                self.static_multi_prototype = static_multi
                self.register_buffer("zs_weight", zs_weight)
                self.register_buffer("eval_zs_weight", eval_zs_weight)

        self.linear = nn.Linear(input_dim, zs_weight_dim)

    @staticmethod
    def _load_text_embeddings(path: str) -> torch.Tensor:
        if not path:
            raise ValueError("text embedding path must be provided when using TPA.")
        arr = np.load(path)
        if arr.ndim == 2:
            arr = arr[:, None, :]
        if arr.ndim != 3:
            raise ValueError(f"Expected text embedding array with 2 or 3 dimensions, got shape {arr.shape}.")
        return torch.from_numpy(arr).to(dtype=torch.float32)

    @staticmethod
    def _load_static_embeddings(path: str) -> Tuple[torch.Tensor, bool]:
        arr = np.load(path)
        if arr.ndim == 2:
            weight = torch.from_numpy(arr).to(dtype=torch.float32).permute(1, 0).contiguous()
            return weight, False
        if arr.ndim == 3:
            weight = torch.from_numpy(arr).to(dtype=torch.float32).contiguous()
            return weight, True
        raise ValueError(f"Expected static embedding array with 2 or 3 dimensions, got shape {arr.shape}.")

    def train(self, mode: bool = True):
        super().train(mode)
        if mode:
            self._cached_eval = None
        self._external_prototypes = None
        self._external_apr_loss = None
        return self

    def _maybe_move_text_feats(self, training: bool) -> torch.Tensor:
        feats = self.train_text_feats if training else self.eval_text_feats
        device = self.linear.weight.device
        if feats.device != device:
            feats = feats.to(device)
            if training:
                self.train_text_feats = feats
            else:
                self.eval_text_feats = feats
        return feats

    def _compute_static_logits(
        self,
        x: torch.Tensor,
        *,
        classifier: Optional[torch.Tensor],
        content_inds: Optional[torch.Tensor],
        additional_class: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if classifier is not None:
            zs_weight = classifier.permute(1, 0).contiguous()
        else:
            if self.training:
                if self.static_multi_prototype:
                    zs_weight = self.zs_weight[content_inds] if content_inds is not None else self.zs_weight
                else:
                    zs_weight = self.zs_weight[:, content_inds] if content_inds is not None else self.zs_weight
            else:
                zs_weight = self.eval_zs_weight if self.eval_zs_weight is not None else self.zs_weight
                if self.static_multi_prototype and content_inds is not None:
                    zs_weight = zs_weight[content_inds]
        if self.static_multi_prototype and classifier is None:
            if self.norm_weight:
                zs_weight = F.normalize(zs_weight, p=2, dim=-1)
            features = self._normalize_features(x)
            logits = torch.einsum("bqd,ckd->bqck", features, zs_weight)
            if self.static_multi_prototype_agg == "logsumexp":
                logits = torch.logsumexp(logits, dim=-1)
            elif self.static_multi_prototype_agg == "max":
                logits = logits.max(dim=-1).values
            elif self.static_multi_prototype_agg == "mean":
                logits = logits.mean(dim=-1)
            if additional_class is not None:
                additional = additional_class.to(device=features.device, dtype=features.dtype)
                if self.norm_weight:
                    additional = F.normalize(additional, p=2, dim=-1)
                additional_logits = torch.einsum("bqd,nd->bqn", features, additional)
                logits = torch.cat([logits, additional_logits], dim=-1)
            if self.use_bias:
                logits = logits + self.cls_bias
            return logits
        if self.norm_weight:
            zs_weight = F.normalize(zs_weight, p=2, dim=0)
        if additional_class is not None:
            additional_weight = additional_class.to(
                device=zs_weight.device, dtype=zs_weight.dtype
            ).t()
            if self.norm_weight:
                additional_weight = F.normalize(additional_weight, p=2, dim=0)
            zs_weight = torch.cat([zs_weight, additional_weight], dim=1)
        logits = torch.matmul(self._normalize_features(x), zs_weight)
        if self.use_bias:
            logits = logits + self.cls_bias
        return logits

    def _normalize_features(self, x: torch.Tensor) -> torch.Tensor:
        if self.norm_weight:
            x = self.norm_temperature * F.normalize(x, p=2, dim=-1)
        return x

    def set_external_prototypes(
        self,
        prototypes: Optional[torch.Tensor],
        apr_loss: Optional[torch.Tensor] = None,
    ) -> None:
        """Inject prototypes computed once outside (e.g. by DINO.forward) so that
        every class_embed layer reuses the same TPA output instead of running TPA
        with independent dropout draws."""
        self._external_prototypes = prototypes
        self._external_apr_loss = apr_loss
        self.apr_loss = apr_loss

    def _compute_tpa_logits(
        self,
        x: torch.Tensor,
        *,
        content_inds: Optional[torch.Tensor],
        additional_class: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if self._external_prototypes is not None:
            prototypes = self._external_prototypes
            self.apr_loss = self._external_apr_loss
        elif self.training:
            text_feats = self._maybe_move_text_feats(training=True)
            if content_inds is not None:
                text_feats = text_feats[content_inds]
            prototypes, apr_loss = self.tpa(text_feats, with_loss=True)
            self.apr_loss = apr_loss
        else:
            text_feats = self._maybe_move_text_feats(training=False)
            cache_valid = self._cached_eval is not None and self._cached_eval.device == x.device
            if not cache_valid:
                full_prototypes, apr_loss = self.tpa(text_feats, with_loss=False)
                self._cached_eval = full_prototypes.detach()
                self.apr_loss = apr_loss
            else:
                full_prototypes = self._cached_eval
                self.apr_loss = getattr(self, "apr_loss", None)
            prototypes = full_prototypes[content_inds] if content_inds is not None else full_prototypes

        prototypes = prototypes.to(dtype=x.dtype)

        if self.norm_weight:
            prototypes = F.normalize(prototypes, p=2, dim=-1)

        features = self._normalize_features(x)
        logits = torch.einsum("bqd,ckd->bqck", features, prototypes)
        logits = torch.logsumexp(logits, dim=-1)

        if additional_class is not None:
            additional = additional_class.to(device=features.device, dtype=features.dtype)
            if self.norm_weight:
                additional = F.normalize(additional, p=2, dim=-1)
            additional_logits = torch.einsum("bqd,nd->bqn", features, additional)
            logits = torch.cat([logits, additional_logits], dim=-1)

        if self.use_bias:
            logits = logits + self.cls_bias
        return logits

    def forward(
        self,
        x: torch.Tensor,
        classifier: Optional[torch.Tensor] = None,
        content_inds: Optional[torch.Tensor] = None,
        additional_class: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = self.linear(x)
        if self.use_tpa:
            return self._compute_tpa_logits(
                x,
                content_inds=content_inds,
                additional_class=additional_class,
            )
        return self._compute_static_logits(
            x,
            classifier=classifier,
            content_inds=content_inds,
            additional_class=additional_class,
        )
