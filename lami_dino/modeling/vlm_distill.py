"""VLM Distillation: inject soft targets into training batches.

Loads pre-computed VLM soft targets and injects them into batched_inputs
during training. Works as a wrapper around the existing data pipeline.
"""

import torch
import logging

logger = logging.getLogger(__name__)


def _torch_load_compat(path, **kwargs):
    try:
        return torch.load(path, weights_only=False, **kwargs)
    except TypeError:
        return torch.load(path, **kwargs)


class VLMSoftTargetInjector:
    """Loads VLM soft targets and injects them into training batches.

    Usage:
        injector = VLMSoftTargetInjector("dataset/vlm_targets_crop/vlm_soft_targets.pth")

        # In training loop, after getting batched_inputs:
        batched_inputs = injector.inject(batched_inputs)
    """

    def __init__(self, targets_path):
        logger.info(f"Loading VLM soft targets from {targets_path}")
        self.targets = _torch_load_compat(targets_path, map_location="cpu")
        logger.info(f"Loaded VLM targets for {len(self.targets)} images")

    def inject(self, batched_inputs):
        """Inject VLM soft targets into batched_inputs.

        For each image in the batch, looks up its VLM targets by image_id
        and adds them to the input dict.
        """
        for inp in batched_inputs:
            image_id = inp.get("image_id", None)
            if image_id is not None and image_id in self.targets:
                inp["vlm_soft_targets"] = self.targets[image_id]
            else:
                inp["vlm_soft_targets"] = None

        return batched_inputs
