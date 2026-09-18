"""Fresh-start paired GT-normalization training without altering native Trainer."""
from __future__ import annotations

from contextlib import contextmanager
import json
import math
from pathlib import Path

import torch
import torch.distributed as dist

from tools.compare_rare_pr_reports import load_json, save_json
from tools.decoder_aux_ablation_ops import HORIZON, state_digest
from tools.decoder_loss_audit_ops import isolated_rng
from tools.gt_normalization_trial_ops import (
    ARMS, RANK_GUARD, RANK_PERIOD, bank_health, reweight_losses,
)
from tools.accumulation_objective_ops import normalization_plan
from tools.tpa_formula_screen_ops import live_tpa_geometry, verify_training_formula

FRESH_RANK_GUARD = {
    "formation_updates":500,
    "initial":{"mean_rank_min":2.5,"p10_rank_min":2.25,"min_rank_min":2.0,
               "mean_cos_max":.95,"rank_below_2_max":0},
    "max_initial_regression":{"mean_rank":.2,"p10_rank":.2,"mean_cos":.03},
    "final":RANK_GUARD,
}


def apply_fresh_rank_guard(health, update, initial_health=None):
    """Allow APR formation from fresh slots, while rejecting actual collapse."""
    strict = bool(health["guard_pass"])
    progress = min(max(update/FRESH_RANK_GUARD["formation_updates"],0.),1.)
    first, final = FRESH_RANK_GUARD["initial"],FRESH_RANK_GUARD["final"]
    thresholds = {
        "mean_rank_min":first["mean_rank_min"]+(final["mean_rank_min"]-first["mean_rank_min"])*progress,
        "p10_rank_min":first["p10_rank_min"]+(final["p10_rank_min"]-first["p10_rank_min"])*progress,
        "min_rank_min":first["min_rank_min"],
        "mean_cos_max":first["mean_cos_max"]+(final["mean_cos_max"]-first["mean_cos_max"])*progress,
        "rank_below_2_max":first["rank_below_2_max"],
    }
    passed = True
    for split in ("all","rare"):
        row = health[split]
        passed &= (row["mean_rank"] >= thresholds["mean_rank_min"]
                   and row["p10_rank"] >= thresholds["p10_rank_min"]
                   and row["min_rank"] >= thresholds["min_rank_min"]
                   and row["mean_cos"] <= thresholds["mean_cos_max"]
                   and row["rank_below_2_count"] <= thresholds["rank_below_2_max"])
        if initial_health is not None:
            baseline = initial_health[split]
            regression = FRESH_RANK_GUARD["max_initial_regression"]
            passed &= (row["mean_rank"] >= baseline["mean_rank"]-regression["mean_rank"]
                       and row["p10_rank"] >= baseline["p10_rank"]-regression["p10_rank"]
                       and row["mean_cos"] <= baseline["mean_cos"]+regression["mean_cos"])
    result = {k:v for k,v in health.items() if k != "guard_pass"}
    result.update(strict_guard_pass=strict,guard_thresholds=thresholds,
                  formation_progress=progress,guard_pass=bool(passed and (strict if progress == 1. else True)))
    return result


def verify_fresh_pair(current, reference):
    """Require the same inputs/draws; first forward must also have same losses."""
    ignored = {"losses", "multiplier"}
    if ({k:v for k,v in current.items() if k not in ignored}
            != {k:v for k,v in reference.items() if k not in ignored}):
        raise ValueError("Unpaired fresh-start data/RNG/FedLoss/LR/AMP/normalization")
    if current["iteration"] == 0:
        if current["losses"].keys() != reference["losses"].keys() or any(
                not math.isclose(value, reference["losses"][key], rel_tol=2e-4, abs_tol=2e-5)
                for key,value in current["losses"].items()):
            raise ValueError("Fresh A/B first-window raw forward differs")


def make_trainer_class(native, manifest, arm, rank, prompts, inventory, *, rng_context=isolated_rng):
    if arm not in ARMS:
        raise ValueError("Unknown normalization arm")
    output = Path(manifest["output_dir"])/arm

    class FreshNormalizationTrainer(native):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            if (not self.amp or not self.separate_tpa_grad_clip or not self.tpa_conflict_projection
                    or self.gradient_accumulation_steps != 2
                    or dict(self.clip_grad_params or {}) != {"max_norm":.5,"norm_type":2}):
                raise ValueError("Native AMP/routing/clipping/accumulation policy changed")
            self.raw_model = self.model.module if hasattr(self.model,"module") else self.model
            verify_training_formula(self.raw_model,"calibrated")
            self.actual_updates = 0
            self.initial_state = None
            self._micro = 0
            self._window = None
            self._device_index = torch.cuda.current_device() if next(self.raw_model.parameters()).is_cuda else None
            self.pair_stream = (output/f"pairing_rank{rank}.jsonl").open("x")
            self.update_stream = (output/f"updates_rank{rank}.jsonl").open("x")
            self.health_stream = (output/"rank_health.jsonl").open("x") if rank == 0 else None
            self.reference_stream = ((output.parent/"A"/f"pairing_rank{rank}.jsonl").open()
                                     if arm == "B" else None)
            self.health_records = []
            self.initial_health = None
            self.inventory = inventory
            original = self.optimizer.step

            def counted_step(*a, **kw):
                result = original(*a, **kw)
                self.actual_updates += 1
                return result

            self.optimizer.step = counted_step
            self.optimizer.step._with_counter = True
            self.optimizer.step._wrapped_by_lr_sched = True

        def initialize_fresh(self):
            if self.iter != 0 or self.actual_updates != 0 or self.initial_state is not None:
                raise ValueError("Fresh initialization must be captured before iteration zero")
            optimizer = self.optimizer.state_dict()
            if optimizer.get("state"):
                raise ValueError("Fresh optimizer unexpectedly has moments")
            native_state = native.state_dict(self)
            scheduler = native_state.get("hooks",{}).get("LRScheduler",{})
            scaler = self.grad_scaler.state_dict()
            lrs = [group["lr"] for group in optimizer.get("param_groups",[])]
            if (self.lr_scheduler_max_iter != HORIZON or not lrs
                    or scheduler.get("last_epoch") != 0
                    or scheduler.get("base_lrs") != [group.get("initial_lr") for group in optimizer["param_groups"]]
                    or not scaler or not math.isfinite(float(scaler.get("scale",float("nan"))))):
                raise ValueError("Fresh scheduler/AMP/LR state is incomplete")
            model = {k.removeprefix("module."):v for k,v in self.model.state_dict().items()}
            self.initial_state = {"model":state_digest(model),"optimizer":state_digest(optimizer),
                                  "scheduler":state_digest(scheduler),"scaler":state_digest(scaler),
                                  "lrs":lrs,"lr_scheduler_max_iter":self.lr_scheduler_max_iter,
                                  "gradient_accumulation_steps":self.gradient_accumulation_steps,
                                  "inventory":self.inventory}
            path = output/f"initial_rank{rank}.json"
            if path.exists():
                raise ValueError("Fresh initialization receipt already exists")
            if arm == "B":
                reference = load_json(output.parent/"A"/f"initial_rank{rank}.json")
                if self.initial_state != reference:
                    raise ValueError("Fresh A/B model/optimizer/scheduler/scaler initialization differs")
            save_json(path,self.initial_state)

        def _prefetch_window(self):
            batches,records,counts = [],[],[]
            for micro in range(2):
                seed = manifest["seed"] + self.iter*128 + rank*8 + micro*2
                with rng_context(seed,self._device_index):
                    data = next(super(FreshNormalizationTrainer,self)._data_loader_iter)
                if len(data) != 4:
                    raise ValueError("Expected four images/rank/microbatch")
                mapped,count = [],0
                for item in data:
                    classes = item["instances"].gt_classes
                    if any(c < 0 or c >= len(self.raw_model.novel_idx) or bool(self.raw_model.novel_idx[c])
                           for c in classes.tolist()):
                        raise ValueError("Rare/invalid GT entered train_norare")
                    count += len(classes)
                    mapped.append({"image_id":int(item["image_id"]),"image":state_digest(item["image"]),
                                   "boxes":state_digest(item["instances"].gt_boxes.tensor),
                                   "classes":state_digest(classes)})
                records.append({"iteration":self.iter,"micro":micro,"data_seed":seed,
                                "forward_seed":seed+1,"mapped":mapped})
                batches.append(data)
                counts.append(count)
            totals = torch.tensor(counts,dtype=torch.long,device=next(self.raw_model.parameters()).device)
            if dist.is_available() and dist.is_initialized():
                if dist.get_world_size() != 4:
                    raise ValueError("Require four ranks")
                dist.all_reduce(totals)
            self.plan = normalization_plan(totals.cpu().tolist(),world_size=4,reference_world_size=8)
            self._window,self._records = batches,records

        @property
        def _data_loader_iter(self):
            trainer = self
            class Iterator:
                def __next__(self):
                    if trainer._window is None or trainer._micro >= 2:
                        raise ValueError("Native step requested data outside prefetched window")
                    return trainer._window[trainer._micro]
            return Iterator()

        @contextmanager
        def paired_forward(self):
            old_forward,old_filter = self.model.forward,self.raw_model.filter_content_info
            sampled = {}

            def filtered(data):
                indices,mapped = old_filter(data)
                sampled["fedloss"] = indices.detach().cpu().tolist()
                return indices,mapped

            def forward(data):
                sampled.clear()
                current = self._records[self._micro]
                with rng_context(current["forward_seed"],self._device_index):
                    raw = old_forward(data)
                    rng = {"cpu":state_digest(torch.random.get_rng_state())}
                    if self._device_index is not None:
                        rng["cuda"] = state_digest(torch.cuda.get_rng_state(self._device_index))
                if "fedloss" not in sampled:
                    raise ValueError("Missing native FedLoss record")
                multiplier = self.plan["detection_loss_multipliers"][self._micro] if arm == "B" else 1.
                losses = {k:float(v.detach()) for k,v in raw.items() if k.startswith("loss")}
                record = {**current,**sampled,"rng_after":rng,"normalization":self.plan,
                          "multiplier":multiplier,"losses":losses,"loss_keys":sorted(losses),
                          "lrs":[g["lr"] for g in self.optimizer.param_groups],
                          "amp_scale":self.grad_scaler.get_scale()}
                if arm == "B":
                    line = self.reference_stream.readline()
                    if not line:
                        raise ValueError("Fresh A pairing transcript ended early")
                    verify_fresh_pair(record,json.loads(line))
                self.pair_stream.write(json.dumps(record)+"\n")
                self.pair_stream.flush()
                self._micro += 1
                return reweight_losses(raw,multiplier)

            self.model.forward,self.raw_model.filter_content_info = forward,filtered
            try:
                yield
            finally:
                self.model.forward,self.raw_model.filter_content_info = old_forward,old_filter

        def clip_model_grads(self):
            norms = super().clip_model_grads()
            self.last_norms = {"detector":float(norms[0]),"tpa":float(norms[1])}
            if any(not math.isfinite(value) for value in self.last_norms.values()):
                raise FloatingPointError("Nonfinite gradients; optimizer step forbidden")
            return norms

        def check_health(self):
            packet = [None]
            if rank == 0:
                try:
                    raw = bank_health(live_tpa_geometry(self.raw_model),prompts,
                                      self.raw_model.novel_idx.cpu().bool())
                    health = apply_fresh_rank_guard(raw,self.actual_updates,self.initial_health)
                    if self.actual_updates == 0:
                        self.initial_health = health
                    packet[0] = {"update":self.actual_updates,**health}
                except Exception as exc:
                    packet[0] = {"update":self.actual_updates,"guard_pass":False,"error":str(exc)}
                self.health_stream.write(json.dumps(packet[0])+"\n")
                self.health_stream.flush()
            if dist.is_available() and dist.is_initialized():
                dist.broadcast_object_list(packet,src=0)
            self.health_records.append(packet[0])
            if not packet[0]["guard_pass"]:
                raise ValueError(f"Prototype health guard failed: {packet[0]}")
            if rank == 0:
                print(f"[rank {arm}] update={self.actual_updates} {packet[0]}",flush=True)

        def run_step(self):
            if not 0 <= self.iter < manifest["updates"]:
                raise ValueError("Fresh trial iteration outside locked budget")
            if self.initial_state is None:
                self.initialize_fresh()
                self.check_health()
            self._micro = 0
            self._prefetch_window()
            before = self.actual_updates
            try:
                with self.paired_forward():
                    native.run_step(self)
            finally:
                self._window = None
            if self.actual_updates != before+1 or self._micro != 2:
                raise ValueError("Skipped AdamW step/incorrect accumulation")
            row = {"iteration":self.iter,"update":self.actual_updates,"arm":arm,
                   "normalization":self.plan,"preclip_norms":self.last_norms,
                   "routing":self._last_tpa_projection_metrics,
                   "amp_scale_after":self.grad_scaler.get_scale()}
            self.update_stream.write(json.dumps(row)+"\n")
            self.update_stream.flush()
            if self.actual_updates % RANK_PERIOD == 0 or self.actual_updates == manifest["updates"]:
                self.check_health()
            if rank == 0 and (self.actual_updates%25 == 0 or self.actual_updates == manifest["updates"]):
                print(f"[train {arm}] updates={self.actual_updates}/{manifest['updates']} "
                      f"GT={self.plan['global_gt_counts']} norms={self.last_norms}",flush=True)

        def state_dict(self):
            result = native.state_dict(self)
            result["fresh_gt_normalization_trial"] = {
                "arm":arm,"normalization":ARMS[arm],"start":0,"updates":self.actual_updates,
                "manifest_fingerprint":manifest["fingerprint"],"initial_state":self.initial_state}
            return result

        def close_streams(self):
            for stream in (self.pair_stream,self.update_stream,self.reference_stream,self.health_stream):
                if stream is not None:
                    stream.close()

    return FreshNormalizationTrainer
