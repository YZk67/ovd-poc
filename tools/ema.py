import copy

import torch

from detectron2.engine import HookBase


def _unwrap(model):
    return model.module if hasattr(model, "module") else model


class ModelEma:
    """Exponential moving average of model parameters/buffers.

    Uses timm-style warmup so the EMA tracks the current model closely in early
    iters (when decay^step is still far from 1) and gradually converges to the
    target decay:  d_effective = min(decay, (1 + step) / (warmup_tau + step))
    """

    def __init__(self, model, decay=0.9999, warmup_tau=10.0, device=None):
        self.decay = decay
        self.warmup_tau = float(warmup_tau)
        self.num_updates = 0
        self.module = copy.deepcopy(_unwrap(model))
        self.module.eval()
        for p in self.module.parameters():
            p.requires_grad_(False)
        if device is not None:
            self.module.to(device)

    def _effective_decay(self):
        # classic timm ModelEmaV2 warmup
        return min(self.decay, (1.0 + self.num_updates) / (self.warmup_tau + self.num_updates))

    @torch.no_grad()
    def update(self, model):
        self.num_updates += 1
        d = self._effective_decay()
        msd = _unwrap(model).state_dict()
        for k, v in self.module.state_dict().items():
            src = msd[k].detach()
            if v.dtype.is_floating_point and src.dtype.is_floating_point:
                v.mul_(d).add_(src.to(v.dtype), alpha=1.0 - d)
            else:
                v.copy_(src)


class EMAHook(HookBase):
    def __init__(self, ema, update_period=1):
        self.ema = ema
        self.update_period = max(1, int(update_period))

    def after_step(self):
        if (self.trainer.iter + 1) % self.update_period == 0:
            self.ema.update(self.trainer.model)
