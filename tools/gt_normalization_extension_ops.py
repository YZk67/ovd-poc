"""Continue verified A/B endpoints; replay only the data cursor, never training."""
from __future__ import annotations

import json
import math
from pathlib import Path

from tools.decoder_aux_ablation_ops import START, HORIZON, state_digest
from tools.decoder_loss_audit_ops import isolated_rng
from tools.gt_normalization_trial_ops import ARMS, make_trainer_class as make_parent_class


def resume_identity(state, iteration, lrs):
    """Full native continuation at an arbitrary certified trial endpoint."""
    optimizer = state.get("optimizer", {})
    scheduler = state.get("hooks", {}).get("LRScheduler", {})
    scaler = state.get("grad_scaler", {})
    groups = optimizer.get("param_groups", [])
    if (state.get("iteration") != iteration or state.get("lr_scheduler_max_iter") != HORIZON
            or state.get("gradient_accumulation_steps") != 2
            or scheduler.get("last_epoch") != iteration+1
            or scheduler.get("base_lrs") != lrs
            or [g.get("lr") for g in groups] != lrs
            or not optimizer.get("state") or not groups
            or not math.isfinite(float(scaler.get("scale", float("nan")))) or scaler["scale"] <= 0):
        raise ValueError("Incomplete endpoint or changed LR/scheduler/scaler/accumulation")
    for group in groups:
        if (tuple(group.get("betas", ())) != (.9,.999)
                or not any(math.isclose(group.get("weight_decay", -1), wd) for wd in (0.,1e-4))):
            raise ValueError("AdamW policy changed")
    return {"optimizer":state_digest(optimizer), "scheduler":state_digest(scheduler),
            "scaler":state_digest(scaler), "lrs":list(lrs)}


def parent_trial_tag(manifest, arm):
    return {"arm":arm, "normalization":ARMS[arm], "start":START,
            "updates":manifest["completed_updates"],
            "manifest_fingerprint":manifest["parent_fingerprint"]}


def arm_view(manifest, arm):
    source = manifest["sources"][arm]
    return {**manifest, "updates":manifest["total_updates"], "checkpoint":source["checkpoint"],
            "resume":source["resume"], "model_digest":source["model_digest"]}


def verify_replayed_input(current, plan, reference):
    for key in ("iteration", "micro", "data_seed", "forward_seed", "mapped"):
        if current[key] != reference[key]:
            raise ValueError(f"Data cursor replay mismatch: {key} at {current['iteration']}/{current['micro']}")
    if plan != reference["normalization"]:
        raise ValueError("Global GT counts/normalization differ during data cursor replay")


def make_trainer_class(native, manifest, arm, rank, prompts, *, rng_context=isolated_rng):
    view = arm_view(manifest, arm)
    parent = make_parent_class(native, view, arm, rank, prompts, rng_context=rng_context)
    completed, total = manifest["completed_updates"], manifest["total_updates"]
    start = START+completed

    class ExtensionTrainer(parent):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            self.actual_updates = completed  # The existing native counter increments from here.
            self.cursor_replay = None

        def load_state_dict(self, state):
            if state.get("gt_normalization_trial") != parent_trial_tag(manifest, arm):
                raise ValueError("Endpoint is from the wrong arm/trial/update count")
            expected = manifest["sources"][arm]
            if resume_identity(state, start-1, expected["resume"]["lrs"]) != expected["resume"]:
                raise ValueError("Saved endpoint state differs from the verified parent")
            # Bypass only the parent's *8ep-specific* check, not native restoration.
            native.load_state_dict(self, state)
            restored = resume_identity(self.state_dict(), start-1, expected["resume"]["lrs"])
            digest = state_digest({k.removeprefix("module."):v for k,v in self.model.state_dict().items()})
            if restored != expected["resume"] or digest != expected["model_digest"]:
                raise ValueError("Endpoint weights/optimizer/scheduler/scaler not restored exactly")
            self.initial_state = {k:restored[k] for k in ("optimizer","scheduler","scaler")}
            self.initial_state["model"] = digest

        def replay_cursor(self):
            if (self.cursor_replay is not None or self.initial_state is None or self.iter != start
                    or self.actual_updates != completed
                    or getattr(self, "_data_loader_iter_obj", None) is not None):
                raise ValueError("Cursor replay requires a fresh loader and fully restored endpoint")
            path = manifest["sources"][arm]["transcripts"][rank]["path"]
            before = self.actual_updates
            # The original isolated mapping contexts restore global RNG. Replay
            # also advances private sampler and aspect-ratio grouping state.
            try:
                with Path(path).open() as handle:
                    for update in range(completed):
                        self.iter = START+update
                        self._micro = 0
                        self._prefetch_window()  # DATA ONLY. No model forward/backward/step.
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
                self.iter = start
                self._window = None
                self._micro = 0
            expected = manifest["sources"][arm]
            # The train loop has set trainer.iter to the NEXT iteration already;
            # compare optimizer/scheduler/scaler directly (no synthetic resume).
            after = {"optimizer":state_digest(self.optimizer.state_dict()),
                     "scheduler":state_digest(self.state_dict()["hooks"]["LRScheduler"]),
                     "scaler":state_digest(self.grad_scaler.state_dict()),
                     "model":state_digest({k.removeprefix("module."):v for k,v in self.model.state_dict().items()})}
            if after != self.initial_state or self.actual_updates != before:
                raise ValueError("Data replay changed weights/optimizer/scheduler/scaler")
            self.cursor_replay = {"verified":True, "windows":completed, "microbatches":completed*2,
                                  "model_forwards":0, "optimizer_updates":0,
                                  "transcript":expected["transcripts"][rank], "state_unchanged":True}
            self.check_health()  # Same full-bank guard as before, at cumulative update500.

        def run_step(self):
            if not start <= self.iter < START+total:
                raise ValueError("Extension iteration outside its locked budget")
            if self.cursor_replay is None:
                self.replay_cursor()
            super().run_step()  # Exactly the existing paired loss/routing/clipping/AdamW path.

        def state_dict(self):
            result = super().state_dict()
            result["gt_normalization_extension"] = {
                "parent_fingerprint":manifest["parent_fingerprint"], "completed_updates":completed,
                "additional_updates":self.actual_updates-completed, "total_updates":self.actual_updates,
                "manifest_fingerprint":manifest["fingerprint"], "cursor_replay":self.cursor_replay}
            return result

    return ExtensionTrainer
