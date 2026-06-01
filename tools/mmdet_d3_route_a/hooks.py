"""Training hooks for MMDetection Route-A experiments."""

from __future__ import annotations

from typing import Iterable, Tuple

from mmengine.hooks import Hook
from mmengine.logging import MMLogger

from mmdet.registry import HOOKS


def _unwrap_model(model):
    return model.module if hasattr(model, "module") else model


@HOOKS.register_module()
class TrainableParamFreezeHook(Hook):
    """Freeze all parameters except a small allowlist of module prefixes."""

    priority = "VERY_HIGH"

    def __init__(self, trainable_prefixes: Iterable[str]) -> None:
        self.trainable_prefixes: Tuple[str, ...] = tuple(str(prefix) for prefix in trainable_prefixes)
        if not self.trainable_prefixes:
            raise ValueError("TrainableParamFreezeHook requires at least one trainable prefix.")

    def before_train(self, runner) -> None:
        model = _unwrap_model(runner.model)
        trainable_params = 0
        frozen_params = 0
        trainable_tensors = 0

        for name, param in model.named_parameters():
            trainable = name.startswith(self.trainable_prefixes)
            param.requires_grad_(trainable)
            if trainable:
                trainable_params += param.numel()
                trainable_tensors += 1
            else:
                frozen_params += param.numel()

        logger = MMLogger.get_current_instance()
        logger.info(
            "TrainableParamFreezeHook prefixes=%s trainable_tensors=%d "
            "trainable_params=%.3fM frozen_params=%.3fM",
            self.trainable_prefixes,
            trainable_tensors,
            trainable_params / 1_000_000,
            frozen_params / 1_000_000,
        )

    def before_train_iter(self, runner, batch_idx: int, data_batch=None) -> None:
        model = _unwrap_model(runner.model)
        for module_name, module in model.named_modules():
            if not module_name:
                continue
            is_trainable_or_parent = module_name.startswith(self.trainable_prefixes) or any(
                prefix.startswith(f"{module_name}.") for prefix in self.trainable_prefixes
            )
            if not is_trainable_or_parent:
                module.eval()

