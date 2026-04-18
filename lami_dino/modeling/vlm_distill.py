"""VLM distillation: inject pre-computed soft targets into training batches.

Loaded targets are a dict keyed by COCO image_id mapping to a list of per-object
dicts with keys: ann_id, bbox, similarity_distribution (65-dim), uncertainty,
attributes, description.
"""

import logging

import torch

# Use a name under "detectron2" so detectron2.setup_logger attaches the INFO handler.
logger = logging.getLogger("detectron2.vlm_distill")


def _torch_load_compat(path, **kwargs):
    try:
        return torch.load(path, weights_only=False, **kwargs)
    except TypeError:
        return torch.load(path, **kwargs)


class VLMSoftTargetInjector:
    def __init__(self, targets_path):
        logger.warning(f"[VLM distill] loading soft targets from {targets_path}")
        self.targets = _torch_load_compat(targets_path, map_location="cpu")
        sample_keys = list(self.targets.keys())[:3]
        logger.warning(
            f"[VLM distill] loaded {len(self.targets)} images; "
            f"sample keys={sample_keys} "
            f"(type={type(sample_keys[0]).__name__ if sample_keys else 'n/a'})"
        )
        self._diag_batches_remaining = 20
        self._diag_hit = 0
        self._diag_miss = 0
        self._ann_id_hit = 0
        self._ann_id_miss = 0

    def inject(self, batched_inputs):
        for inp in batched_inputs:
            image_id = inp.get("image_id", None)
            tgt = self.targets.get(image_id) if image_id is not None else None
            inp["vlm_soft_targets"] = tgt
            if self._diag_batches_remaining > 0:
                if tgt is not None:
                    self._diag_hit += 1
                    # Also peek whether GT ann_ids will line up with target ann_ids.
                    gt_inst = inp.get("instances", None)
                    if gt_inst is not None and gt_inst.has("gt_ann_ids"):
                        gt_ids = set(int(x) for x in gt_inst.gt_ann_ids.tolist())
                        tgt_ids = {int(t["ann_id"]) for t in tgt if t and "ann_id" in t}
                        self._ann_id_hit += len(gt_ids & tgt_ids)
                        self._ann_id_miss += len(gt_ids - tgt_ids)
                else:
                    self._diag_miss += 1
        if self._diag_batches_remaining > 0:
            self._diag_batches_remaining -= 1
            if self._diag_batches_remaining == 0:
                sample_ids = [inp.get("image_id") for inp in batched_inputs]
                logger.warning(
                    f"[VLM distill] first-20-batch diag: "
                    f"image-hits={self._diag_hit}, image-misses={self._diag_miss}; "
                    f"ann_id-hits={self._ann_id_hit}, ann_id-misses={self._ann_id_miss}; "
                    f"last-batch image_ids={sample_ids} "
                    f"(type={type(sample_ids[0]).__name__ if sample_ids else 'n/a'})"
                )
        return batched_inputs
