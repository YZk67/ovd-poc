import copy

import torch

from detectron2.engine import HookBase


def _unwrap(model):
    return model.module if hasattr(model, "module") else model


class ModelEma:
    """Exponential moving average of model parameters/buffers."""

    def __init__(self, model, decay=0.9999, device=None):
        self.decay = decay
        self.module = copy.deepcopy(_unwrap(model))
        self.module.eval()
        for p in self.module.parameters():
            p.requires_grad_(False)
        if device is not None:
            self.module.to(device)

    @torch.no_grad()
    def update(self, model):
        msd = _unwrap(model).state_dict()
        for k, v in self.module.state_dict().items():
            src = msd[k].detach()
            if v.dtype.is_floating_point and src.dtype.is_floating_point:
                v.mul_(self.decay).add_(src.to(v.dtype), alpha=1.0 - self.decay)
            else:
                v.copy_(src)


class EMAHook(HookBase):
    def __init__(self, ema, update_period=1):
        self.ema = ema
        self.update_period = max(1, int(update_period))

    def after_step(self):
        if (self.trainer.iter + 1) % self.update_period == 0:
            self.ema.update(self.trainer.model)
