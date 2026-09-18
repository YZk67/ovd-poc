"""One-variable paired continuation; production Trainer is not modified."""
from __future__ import annotations

from contextlib import contextmanager
import json
import math
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F

from tools.accumulation_objective_ops import normalization_plan, split_objective
from tools.decoder_aux_ablation_ops import START, HORIZON, state_digest
from tools.decoder_loss_audit_ops import isolated_rng
from tools.tpa_formula_screen_ops import live_tpa_geometry, verify_training_formula
from tools.tpa_geometry_audit_ops import reconstruct_tpa

ARMS = {"A": "native_micro", "B": "pooled_gt"}
RANK_PERIOD = 50
# Predeclared engineering stop rules, NOT AP-tuned hyperparameters or proof
# that every individual category remains noncollapsed.
RANK_GUARD = {"mean_rank_min": 4.0, "p10_rank_min": 3.0, "mean_cos_max": 0.8}


def reweight_losses(losses, multiplier):
    detection, _ = split_objective(losses)
    if not math.isfinite(multiplier) or multiplier <= 0:
        raise ValueError("Invalid GT normalization multiplier")
    result = dict(losses)
    if multiplier != 1.0:
        for key in detection:
            result[key] = result[key] * multiplier
    return result


def verify_pair(current, reference):
    # Actual losses/matching may diverge AFTER an optimizer update. All draws,
    # mapped inputs, native category lists, LR and AMP scales must stay paired.
    ignored = {"losses", "multiplier"}
    if ({k: v for k, v in current.items() if k not in ignored}
            != {k: v for k, v in reference.items() if k not in ignored}):
        raise ValueError("Unpaired data/RNG/FedLoss/LR/AMP/normalization plan")
    if current["iteration"] == START and any(
            not math.isclose(v, reference["losses"][k], rel_tol=2e-4, abs_tol=2e-5)
            for k, v in current["losses"].items()):
        raise ValueError("First-window raw forward differs before any update")


@torch.no_grad()
def bank_health(state, prompts, rare_mask):
    if (state["prototype_queries"].shape[0] != 5
            or float(state["prototype_mode_strength"]) != 0
            or abs(float(state["slot_prior_strength"]) - .2) > 1e-6):
        raise ValueError("Expected no-radius K5/slot-prior=.2")
    bank = reconstruct_tpa(prompts, state, .004375)["after"]
    if rare_mask.shape != (len(bank),) or not rare_mask.any():
        raise ValueError("Invalid rare mask for full prompt bank")
    if not torch.isfinite(bank).all() or (bank.norm(dim=-1) <= 1e-12).any():
        raise ValueError("Nonfinite/zero prototype")
    unit = F.normalize(bank, dim=-1)
    gram = unit @ unit.transpose(-1, -2)
    singular = torch.linalg.eigvalsh(gram).clamp_min(0).sqrt()
    fractions = singular / singular.sum(-1, keepdim=True).clamp_min(1e-12)
    ranks = (-(fractions * fractions.clamp_min(1e-12).log()).sum(-1)).exp()
    cosines = (gram.sum((-2, -1)) - gram.diagonal(dim1=-2, dim2=-1).sum(-1)) / 20
    result = {}
    for name, mask in (("all", torch.ones(len(bank), dtype=torch.bool)), ("rare", rare_mask.cpu().bool())):
        r, c = ranks[mask], cosines[mask]
        result[name] = {"classes": len(r), "mean_rank": float(r.mean()),
                        "p10_rank": float(torch.quantile(r, .1)), "min_rank": float(r.min()),
                        "mean_cos": float(c.mean()), "rank_below_2_count": int((r < 2).sum())}
    result["guard_pass"] = all(
        row["mean_rank"] >= RANK_GUARD["mean_rank_min"]
        and row["p10_rank"] >= RANK_GUARD["p10_rank_min"]
        and row["mean_cos"] <= RANK_GUARD["mean_cos_max"]
        for row in (result["all"], result["rare"]))
    return result


def make_trainer_class(native, manifest, arm, rank, prompts, *, rng_context=isolated_rng):
    if arm not in ARMS:
        raise ValueError("Unknown normalization arm")
    output = Path(manifest["output_dir"]) / arm

    class NormalizationTrainer(native):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            if (not self.amp or not self.separate_tpa_grad_clip or not self.tpa_conflict_projection
                    or self.gradient_accumulation_steps != 2
                    or dict(self.clip_grad_params or {}) != {"max_norm": .5, "norm_type": 2}):
                raise ValueError("Native AMP/routing/clipping/accumulation policy changed")
            self.raw_model = self.model.module if hasattr(self.model, "module") else self.model
            verify_training_formula(self.raw_model, "calibrated")
            self.actual_updates = 0
            self.initial_state = None
            self._micro = 0
            self._window = None
            self._device_index = torch.cuda.current_device() if next(self.raw_model.parameters()).is_cuda else None
            self.pair_stream = (output / f"pairing_rank{rank}.jsonl").open("x")
            self.update_stream = (output / f"updates_rank{rank}.jsonl").open("x")
            self.health_stream = (output / "rank_health.jsonl").open("x") if rank == 0 else None
            self.reference_stream = ((output.parent / "A" / f"pairing_rank{rank}.jsonl").open()
                                     if arm == "B" else None)
            self.health_records = []
            original = self.optimizer.step

            def counted_step(*a, **kw):
                result = original(*a, **kw)  # TPA remains trainable, including moments/decay.
                self.actual_updates += 1
                return result

            self.optimizer.step = counted_step
            self.optimizer.step._with_counter = True
            self.optimizer.step._wrapped_by_lr_sched = True

        def load_state_dict(self, state):
            super().load_state_dict(state)
            loaded = {"optimizer": state_digest(self.optimizer.state_dict()),
                      "scheduler": state_digest(self.state_dict()["hooks"]["LRScheduler"]),
                      "scaler": state_digest(self.grad_scaler.state_dict())}
            if self.iter != START-1 or loaded != {k: manifest["resume"][k] for k in loaded}:
                raise ValueError("Full 8ep optimizer/scheduler/scaler not restored exactly")
            digest = state_digest({k.removeprefix("module."): v for k, v in self.model.state_dict().items()})
            if digest != manifest["model_digest"]:
                raise ValueError("Full 8ep weights not restored exactly")
            self.initial_state = {**loaded, "model": digest}

        def _prefetch_window(self):
            batches, records, counts = [], [], []
            for micro in range(2):
                seed = manifest["seed"] + (self.iter-START)*128 + rank*8 + micro*2
                with rng_context(seed, self._device_index):
                    data = next(super(NormalizationTrainer, self)._data_loader_iter)
                if len(data) != 4:
                    raise ValueError("Expected four images/rank/microbatch")
                mapped, count = [], 0
                for item in data:
                    classes = item["instances"].gt_classes
                    if any(c < 0 or c >= len(self.raw_model.novel_idx) or bool(self.raw_model.novel_idx[c])
                           for c in classes.tolist()):
                        raise ValueError("Rare/invalid GT entered train_norare")
                    count += len(classes)
                    mapped.append({"image_id": int(item["image_id"]), "image": state_digest(item["image"]),
                                   "boxes": state_digest(item["instances"].gt_boxes.tensor),
                                   "classes": state_digest(classes)})
                records.append({"iteration": self.iter, "micro": micro, "data_seed": seed,
                                "forward_seed": seed+1, "mapped": mapped})
                batches.append(data)
                counts.append(count)
            totals = torch.tensor(counts, dtype=torch.long, device=next(self.raw_model.parameters()).device)
            if dist.is_available() and dist.is_initialized():
                if dist.get_world_size() != 4:
                    raise ValueError("Require four ranks")
                dist.all_reduce(totals)
            self.plan = normalization_plan(totals.cpu().tolist(), world_size=4, reference_world_size=8)
            self._window, self._records = batches, records

        @property
        def _data_loader_iter(self):
            trainer = self
            class Iterator:
                def __next__(self):
                    if trainer._window is None or trainer._micro >= 2:
                        raise ValueError("Native step requested data outside its prefetched window")
                    return trainer._window[trainer._micro]
            return Iterator()

        @contextmanager
        def paired_forward(self):
            old_forward, old_filter = self.model.forward, self.raw_model.filter_content_info
            sampled = {}

            def filtered(data):
                indices, mapped = old_filter(data)  # No shared-window category sampling.
                sampled["fedloss"] = indices.detach().cpu().tolist()
                return indices, mapped

            def forward(data):
                sampled.clear()
                record = self._records[self._micro]
                with rng_context(record["forward_seed"], self._device_index):
                    raw = old_forward(data)
                    rng = {"cpu": state_digest(torch.random.get_rng_state())}
                    if self._device_index is not None:
                        rng["cuda"] = state_digest(torch.cuda.get_rng_state(self._device_index))
                if "fedloss" not in sampled:
                    raise ValueError("Missing native FedLoss record")
                multiplier = self.plan["detection_loss_multipliers"][self._micro] if arm == "B" else 1.
                result = reweight_losses(raw, multiplier)
                losses = {k: float(v.detach()) for k, v in raw.items() if k.startswith("loss")}
                record = {**record, **sampled, "rng_after": rng, "normalization": self.plan,
                          "multiplier": multiplier, "losses": losses, "loss_keys": sorted(losses),
                          "lrs": [g["lr"] for g in self.optimizer.param_groups],
                          "amp_scale": self.grad_scaler.get_scale()}
                if record["lrs"] != manifest["resume"]["lrs"]:
                    raise ValueError("LR changed during the locked high-LR short trial")
                if arm == "B":
                    line = self.reference_stream.readline()
                    if not line:
                        raise ValueError("A pairing transcript ended early")
                    verify_pair(record, json.loads(line))
                self.pair_stream.write(json.dumps(record) + "\n")
                self.pair_stream.flush()
                self._micro += 1
                return result

            self.model.forward, self.raw_model.filter_content_info = forward, filtered
            try:
                yield
            finally:
                self.model.forward, self.raw_model.filter_content_info = old_forward, old_filter

        def clip_model_grads(self):
            norms = super().clip_model_grads()  # Each arm's OWN native coefficients; no manual equalization.
            self.last_norms = {"detector": float(norms[0]), "tpa": float(norms[1])}
            if any(not math.isfinite(x) for x in self.last_norms.values()):
                raise FloatingPointError("Nonfinite gradients; no optimizer step allowed")
            return norms

        def check_health(self):
            # CPU reconstruction avoids changing model mode, dropout, cached
            # banks, TPA schedule or any RNG. All ranks agree before continuing.
            packet = [None]
            if rank == 0:
                try:
                    health = bank_health(live_tpa_geometry(self.raw_model), prompts,
                                         self.raw_model.novel_idx.cpu().bool())
                    packet[0] = {"update": self.actual_updates, **health}
                except Exception as exc:
                    packet[0] = {"update": self.actual_updates, "guard_pass": False, "error": str(exc)}
                self.health_stream.write(json.dumps(packet[0]) + "\n")
                self.health_stream.flush()
            if dist.is_available() and dist.is_initialized():
                dist.broadcast_object_list(packet, src=0)
            self.health_records.append(packet[0])
            if not packet[0]["guard_pass"]:
                raise ValueError(f"Prototype health guard failed: {packet[0]}; no success receipt")
            if rank == 0:
                print(f"[rank {arm}] update={self.actual_updates} {packet[0]}", flush=True)

        def run_step(self):
            if not START <= self.iter < START+manifest["updates"] or self.initial_state is None:
                raise ValueError("Not fully restored or update budget exceeded")
            if self.actual_updates == 0:
                self.check_health()
            self._micro = 0
            self._prefetch_window()
            before = self.actual_updates
            try:
                with self.paired_forward():
                    super().run_step()
            finally:
                self._window = None
            if self.actual_updates != before+1 or self._micro != 2:
                raise ValueError("Skipped AdamW step/incorrect accumulation; paired trial aborted")
            row = {"iteration": self.iter, "update": self.actual_updates, "arm": arm,
                   "normalization": self.plan, "preclip_norms": self.last_norms,
                   "routing": self._last_tpa_projection_metrics,
                   "amp_scale_after": self.grad_scaler.get_scale()}
            self.update_stream.write(json.dumps(row) + "\n")
            self.update_stream.flush()
            if self.actual_updates % RANK_PERIOD == 0 or self.actual_updates == manifest["updates"]:
                self.check_health()
            if rank == 0 and (self.actual_updates % 25 == 0 or self.actual_updates == manifest["updates"]):
                print(f"[train {arm}] updates={self.actual_updates}/{manifest['updates']} "
                      f"GT={self.plan['global_gt_counts']} norms={self.last_norms}", flush=True)

        def state_dict(self):
            result = super().state_dict()
            result["gt_normalization_trial"] = {
                "arm": arm, "normalization": ARMS[arm], "start": START,
                "updates": self.actual_updates, "manifest_fingerprint": manifest["fingerprint"]}
            return result

        def close_streams(self):
            for stream in (self.pair_stream, self.update_stream, self.reference_stream, self.health_stream):
                if stream is not None:
                    stream.close()

    return NormalizationTrainer
