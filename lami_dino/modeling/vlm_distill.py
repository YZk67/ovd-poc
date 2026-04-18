"""VLM distillation: inject pre-computed soft targets into training batches.

Loaded targets are a dict keyed by COCO image_id mapping to a list of per-object
dicts with keys: ann_id, bbox, similarity_distribution (65-dim), uncertainty,
attributes, description.
"""

import logging

import torch

logger = logging.getLogger(__name__)


def _torch_load_compat(path, **kwargs):
    try:
        return torch.load(path, weights_only=False, **kwargs)
    except TypeError:
        return torch.load(path, **kwargs)


class VLMSoftTargetInjector:
    def __init__(self, targets_path):
        logger.info(f"[VLM distill] loading soft targets from {targets_path}")
        self.targets = _torch_load_compat(targets_path, map_location="cpu")
        logger.info(f"[VLM distill] loaded {len(self.targets)} images")
        sample_keys = list(self.targets.keys())[:3]
        logger.info(
            f"[VLM distill] sample target keys: {sample_keys} "
            f"(type={type(sample_keys[0]).__name__ if sample_keys else 'n/a'})"
        )
        self._diag_batches_remaining = 5
        self._diag_hit = 0
        self._diag_miss = 0

    def inject(self, batched_inputs):
        for inp in batched_inputs:
            image_id = inp.get("image_id", None)
            tgt = self.targets.get(image_id) if image_id is not None else None
            inp["vlm_soft_targets"] = tgt
            if self._diag_batches_remaining > 0:
                if tgt is not None:
                    self._diag_hit += 1
                else:
                    self._diag_miss += 1
        if self._diag_batches_remaining > 0:
            self._diag_batches_remaining -= 1
            if self._diag_batches_remaining == 0:
                sample_ids = [inp.get("image_id") for inp in batched_inputs]
                logger.info(
                    f"[VLM distill] first-5-batch diag: "
                    f"hits={self._diag_hit}, misses={self._diag_miss}, "
                    f"last-batch image_ids={sample_ids} "
                    f"(type={type(sample_ids[0]).__name__ if sample_ids else 'n/a'})"
                )
        return batched_inputs
