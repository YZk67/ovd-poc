"""Training hook for VSAS distribution logging.

Saves attribute score distributions at the end of each epoch.
Used offline by select_attributes_vsas.py.
"""

import logging

from detectron2.engine import HookBase
from detectron2.utils import comm

logger = logging.getLogger(__name__)


class VSASDistributionHook(HookBase):
    """Hook that saves VSAS distributions at configured iteration intervals.

    Args:
        save_period: Save distributions every N iterations.
            Typically set to 1 epoch worth of iterations.
    """

    def __init__(self, save_period=None):
        self.save_period = save_period

    def after_step(self):
        if self.save_period is None:
            return

        next_iter = self.trainer.iter + 1
        if next_iter % self.save_period == 0 and comm.is_main_process():
            model = self.trainer.model
            if hasattr(model, "module"):
                model = model.module

            if hasattr(model, "save_vsas_distributions"):
                logger.info(f"[VSAS] Saving distributions at iter {next_iter}")
                model.save_vsas_distributions()

    def after_train(self):
        """Save distributions at end of training."""
        if comm.is_main_process():
            model = self.trainer.model
            if hasattr(model, "module"):
                model = model.module

            if hasattr(model, "save_vsas_distributions"):
                logger.info("[VSAS] Saving final distributions")
                model.save_vsas_distributions()
