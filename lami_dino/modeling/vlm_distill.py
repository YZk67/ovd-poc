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

    def inject(self, batched_inputs):
        for inp in batched_inputs:
            image_id = inp.get("image_id", None)
            inp["vlm_soft_targets"] = (
                self.targets.get(image_id) if image_id is not None else None
            )
        return batched_inputs
