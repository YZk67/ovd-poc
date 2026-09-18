"""Three-arm APR/projection trial; production forward and Trainer stay unchanged."""
from __future__ import annotations

from contextlib import contextmanager
import json
import math
from pathlib import Path

import torch
import torch.distributed as dist

from tools.decoder_aux_ablation_ops import START, state_digest
from tools.decoder_loss_audit_ops import isolated_rng
from tools.gt_normalization_trial_ops import (
    RANK_PERIOD, bank_health, make_trainer_class as normalization_trainer,
)
from tools.tpa_formula_screen_ops import live_tpa_geometry, verify_training_formula

ARMS = {
    "A": {"projection": True, "barrier_weight": .1, "balance_weight": .03},
    "P": {"projection": False, "barrier_weight": .1, "balance_weight": .03},
    "R": {"projection": False, "barrier_weight": 0., "balance_weight": .03},
}


def configure_policy(trainer, arm):
    policy = ARMS[arm]
    heads = list(trainer.raw_model.transformer.decoder.class_embed)
    tpa = trainer._get_tpa()
    if (not heads or any(head.tpa is not tpa for head in heads)
            or not math.isclose(tpa.lambda_orth_base, .1)
            or not math.isclose(tpa.lambda_div_base, .03) or tpa.warmup_steps != 0
            or not math.isclose(tpa.diversity_barrier_eps, 1e-4)
            or not math.isclose(trainer.raw_model.criterion.weight_dict["loss_apr"], 1.)):
        raise ValueError("Expected shared TPA, native .1 barrier/.03 balance, no APR warmup, APR weight1")
    # Nonpersistent loss coefficient, not a model weight/buffer. Full model and
    # optimizer state can therefore be restored identically in all three arms.
    # R still computes the native loss; only its barrier coefficient is zero.
    tpa.lambda_orth_base = policy["barrier_weight"]
    if trainer.tpa_conflict_projection != policy["projection"]:
        raise ValueError("Projection flag does not match the declared arm")
    return dict(policy)


def check_apr_record(losses, terms, policy):
    required = ("loss_prototype_diversity", "loss_balance", "lambda_orth", "lambda_balance")
    if any(k not in terms or not math.isfinite(terms[k]) for k in required):
        raise ValueError("Missing/nonfinite APR component diagnostics")
    if (not math.isclose(terms["lambda_orth"], policy["barrier_weight"])
            or not math.isclose(terms["lambda_balance"], policy["balance_weight"])):
        raise ValueError("Barrier/balance coefficient changed")
    expected = terms["lambda_orth"]*terms["loss_prototype_diversity"] + terms["lambda_balance"]*terms["loss_balance"]
    if not math.isclose(losses["loss_apr"], expected, rel_tol=2e-3, abs_tol=2e-5):
        raise ValueError("Native APR loss differs from barrier + balance")


def verify_pair(current, reference):
    ignored = {"losses", "apr_components", "policy"}
    if ({k:v for k,v in current.items() if k not in ignored}
            != {k:v for k,v in reference.items() if k not in ignored}):
        raise ValueError("Unpaired data/RNG/FedLoss/LR/AMP/native GT normalization")
    if current["iteration"] == START:
        # R's APR value changes by design, but all other first-window forward
        # losses and BOTH unweighted regularizer terms must remain identical.
        pairs = [(v,reference["losses"][k]) for k,v in current["losses"].items() if k != "loss_apr"]
        pairs += [(current["apr_components"][k],reference["apr_components"][k])
                  for k in ("loss_prototype_diversity", "loss_balance")]
        if any(not math.isclose(a,b,rel_tol=2e-4,abs_tol=2e-5) for a,b in pairs):
            raise ValueError("First-window model forward differs before any update")


def make_trainer_class(native, manifest, arm, rank, prompts, *, rng_context=isolated_rng):
    if arm not in ARMS:
        raise ValueError("Unknown APR/projection arm")
    # Reuse the unchanged full-resume checks, input prefetch/iterator, native
    # clipping wrapper and stream cleanup, NOT its loss-reweighting run_step.
    parent = normalization_trainer(native, manifest, "A", rank, prompts, rng_context=rng_context)
    output = Path(manifest["output_dir"])/arm

    class TrialTrainer(parent):
        def __init__(self, *args, **kwargs):
            native.__init__(self, *args, **kwargs)
            if (not self.amp or not self.separate_tpa_grad_clip or self.gradient_accumulation_steps != 2
                    or dict(self.clip_grad_params or {}) != {"max_norm":.5,"norm_type":2}):
                raise ValueError("Native AMP/clipping/accumulation policy changed")
            self.raw_model = self.model.module if hasattr(self.model,"module") else self.model
            verify_training_formula(self.raw_model,"calibrated")
            self.policy = configure_policy(self,arm)
            self.actual_updates, self.initial_state, self._micro, self._window = 0, None, 0, None
            self._device_index = torch.cuda.current_device() if next(self.raw_model.parameters()).is_cuda else None
            self.pair_stream = (output/f"pairing_rank{rank}.jsonl").open("x")
            self.update_stream = (output/f"updates_rank{rank}.jsonl").open("x")
            self.health_stream = (output/"rank_health.jsonl").open("x") if rank == 0 else None
            self.reference_stream = ((output.parent/"A"/f"pairing_rank{rank}.jsonl").open()
                                     if arm != "A" else None)
            self.health_records = []
            original = self.optimizer.step
            def counted(*a,**kw):
                value = original(*a,**kw)
                self.actual_updates += 1
                return value
            self.optimizer.step = counted
            self.optimizer.step._with_counter = True
            self.optimizer.step._wrapped_by_lr_sched = True

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
                row = self._records[self._micro]
                with rng_context(row["forward_seed"],self._device_index):
                    raw = old_forward(data)
                    rng = {"cpu":state_digest(torch.random.get_rng_state())}
                    if self._device_index is not None:
                        rng["cuda"] = state_digest(torch.cuda.get_rng_state(self._device_index))
                if "fedloss" not in sampled:
                    raise ValueError("Missing native FedLoss record")
                losses = {k:float(v.detach()) for k,v in raw.items() if k.startswith("loss")}
                if any(not math.isfinite(v) for v in losses.values()):
                    raise FloatingPointError("Nonfinite loss before optimizer update")
                terms = self._get_tpa().last_loss_terms
                check_apr_record(losses,terms,self.policy)
                components = {k:terms[k] for k in ("loss_prototype_diversity","loss_balance","lambda_orth","lambda_balance")}
                record = {**row,**sampled,"rng_after":rng,"normalization":self.plan,"multiplier":1.,
                          "losses":losses,"loss_keys":sorted(losses),"apr_components":components,"policy":self.policy,
                          "lrs":[g["lr"] for g in self.optimizer.param_groups],"amp_scale":self.grad_scaler.get_scale()}
                if record["lrs"] != manifest["resume"]["lrs"]:
                    raise ValueError("LR changed during locked high-LR trial")
                if self.reference_stream:
                    line = self.reference_stream.readline()
                    if not line:
                        raise ValueError("A pairing transcript ended early")
                    verify_pair(record,json.loads(line))
                self.pair_stream.write(json.dumps(record)+"\n")
                self.pair_stream.flush()
                self._micro += 1
                return raw  # NO GT reweighting, APR loss replacement or other gradient hook.
            self.model.forward,self.raw_model.filter_content_info = forward,filtered
            try:
                yield
            finally:
                self.model.forward,self.raw_model.filter_content_info = old_forward,old_filter

        def check_health(self):
            packet = [None]
            if rank == 0:
                try:
                    health = bank_health(live_tpa_geometry(self.raw_model),prompts,self.raw_model.novel_idx.cpu().bool())
                    packet[0] = {"update":self.actual_updates,**health}
                except Exception as exc:
                    packet[0] = {"update":self.actual_updates,"guard_pass":False,"error":str(exc)}
                self.health_stream.write(json.dumps(packet[0])+"\n")
                self.health_stream.flush()
            if dist.is_available() and dist.is_initialized():
                dist.broadcast_object_list(packet,src=0)
            self.health_records.append(packet[0])
            if not packet[0]["guard_pass"]:
                raise ValueError(f"Prototype health guard failed: {packet[0]}; trial stopped")
            if rank == 0:
                print(f"[rank {arm}] update={self.actual_updates} {packet[0]}",flush=True)

        def run_step(self):
            if not START <= self.iter < START+manifest["updates"] or self.initial_state is None:
                raise ValueError("Not fully restored or trial budget exceeded")
            if self.actual_updates == 0:
                self.check_health()
            self._micro = 0
            self._prefetch_window()
            before = self.actual_updates
            try:
                with self.paired_forward():
                    native.run_step(self)  # Actual production routing, unscale, clipping and AdamW.
            finally:
                self._window = None
            if self.actual_updates != before+1 or self._micro != 2:
                raise ValueError("Skipped optimizer step or incorrect microbatch count")
            routing = self._last_tpa_projection_metrics if self.tpa_conflict_projection else {"enabled":False}
            row = {"iteration":self.iter,"update":self.actual_updates,"arm":arm,"policy":self.policy,
                   "normalization":self.plan,"preclip_norms":self.last_norms,"routing":routing,
                   "amp_scale_after":self.grad_scaler.get_scale()}
            self.update_stream.write(json.dumps(row,allow_nan=False)+"\n")
            self.update_stream.flush()
            if self.actual_updates % RANK_PERIOD == 0 or self.actual_updates == manifest["updates"]:
                self.check_health()
            if rank == 0 and (self.actual_updates % 25 == 0 or self.actual_updates == manifest["updates"]):
                print(f"[train {arm}] {self.actual_updates}/{manifest['updates']} policy={self.policy}",flush=True)

        def state_dict(self):
            state = native.state_dict(self)
            state["apr_projection_trial"] = {"arm":arm,"policy":self.policy,"start":START,
                "updates":self.actual_updates,"manifest_fingerprint":manifest["fingerprint"]}
            return state

    return TrialTrainer
