"""Resume the A/P endpoints, preserving the original paired data timeline."""
from __future__ import annotations

import json
from pathlib import Path

from tools.apr_projection_trial_ops import ARMS as PARENT_ARMS, make_trainer_class as parent_trainer
from tools.decoder_aux_ablation_ops import START, state_digest
from tools.decoder_loss_audit_ops import isolated_rng
from tools.gt_normalization_extension_ops import arm_view, resume_identity, verify_replayed_input

ARMS = {arm: dict(PARENT_ARMS[arm]) for arm in ("A", "P")}


def trial_tag(m, arm, *, parent=False):
    return {"arm": arm, "policy": ARMS[arm], "start": START,
            "updates": m["completed_updates"] if parent else m["total_updates"],
            "manifest_fingerprint": m["parent_fingerprint"] if parent else m["fingerprint"]}


def cursor_receipt(m, arm, rank):
    return {"verified": True, "windows": m["completed_updates"],
            "microbatches": m["completed_updates"]*2, "model_forwards": 0,
            "optimizer_updates": 0, "state_unchanged": True,
            "transcript": m["sources"][arm]["transcripts"][rank]}


def extension_tag(m, total, replay):
    return {"parent_fingerprint": m["parent_fingerprint"],
            "completed_updates": m["completed_updates"],
            "additional_updates": total-m["completed_updates"], "total_updates": total,
            "manifest_fingerprint": m["fingerprint"], "cursor_replay": replay}


def make_trainer_class(native, manifest, arm, rank, prompts, *, rng_context=isolated_rng):
    if arm not in ARMS:
        raise ValueError("Only A/P may be extended; R is excluded")
    parent = parent_trainer(native, arm_view(manifest, arm), arm, rank, prompts, rng_context=rng_context)
    completed, total = manifest["completed_updates"], manifest["total_updates"]
    start = START+completed
    source = manifest["sources"][arm]

    class ExtensionTrainer(parent):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.actual_updates = completed
            self.cursor_replay = None

        def load_state_dict(self, state):
            if state.get("apr_projection_trial") != trial_tag(manifest, arm, parent=True):
                raise ValueError("Endpoint is from the wrong arm/trial/update count")
            if resume_identity(state, start-1, source["resume"]["lrs"]) != source["resume"]:
                raise ValueError("Saved endpoint optimizer/scheduler/scaler differs")
            # Skip only the parent's 8ep-specific validation, not native loading.
            native.load_state_dict(self, state)
            restored = resume_identity(self.state_dict(), start-1, source["resume"]["lrs"])
            digest = state_digest({k.removeprefix("module."): v for k, v in self.model.state_dict().items()})
            if restored != source["resume"] or digest != source["model_digest"]:
                raise ValueError("Own endpoint state not restored exactly")
            self.initial_state = {k: restored[k] for k in ("optimizer", "scheduler", "scaler")}
            self.initial_state["model"] = digest

        def replay_cursor(self):
            if (self.cursor_replay is not None or self.initial_state is None or self.iter != start
                    or self.actual_updates != completed
                    or getattr(self, "_data_loader_iter_obj", None) is not None):
                raise ValueError("Cursor replay requires a fresh loader and fully restored endpoint")
            if rank == 0:
                print(f"[cursor {arm}] verifying {completed} DATA windows; no model forward or update", flush=True)
            before = self.actual_updates
            try:
                with Path(source["transcripts"][rank]["path"]).open() as handle:
                    for update in range(completed):
                        self.iter, self._micro = START+update, 0
                        self._prefetch_window()  # Data/augmentation only; advances sampler and grouping.
                        for record in self._records:
                            line = handle.readline()
                            if not line:
                                raise ValueError("Parent transcript ended before cursor replay")
                            verify_replayed_input(record, self.plan, json.loads(line))
                        self._window = None
                        if rank == 0 and (update+1) % 50 == 0:
                            print(f"[cursor {arm}] {update+1}/{completed} windows verified; zero model updates", flush=True)
                    if handle.readline():
                        raise ValueError("Parent transcript has extra records")
            finally:
                self.iter, self._micro, self._window = start, 0, None
            after = {"optimizer": state_digest(self.optimizer.state_dict()),
                     "scheduler": state_digest(self.state_dict()["hooks"]["LRScheduler"]),
                     "scaler": state_digest(self.grad_scaler.state_dict()),
                     "model": state_digest({k.removeprefix("module."): v for k, v in self.model.state_dict().items()})}
            if after != self.initial_state or self.actual_updates != before:
                raise ValueError("Data-only replay changed model/optimizer/scheduler/scaler")
            self.cursor_replay = cursor_receipt(manifest, arm, rank)
            self.check_health()
            if self.health_records[-1] != source["final_rank"]:
                raise ValueError("Resumed full-bank geometry differs from the parent endpoint")

        def run_step(self):
            if not start <= self.iter < START+total:
                raise ValueError("Extension iteration outside its locked budget")
            if self.cursor_replay is None:
                self.replay_cursor()
            super().run_step()  # Unchanged native losses/projection/AMP/clipping/AdamW.

        def state_dict(self):
            result = super().state_dict()
            result["apr_projection_extension"] = extension_tag(manifest, self.actual_updates, self.cursor_replay)
            return result

    return ExtensionTrainer
