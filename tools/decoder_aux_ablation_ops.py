"""Controlled auxiliary-classification intervention; no Detectron2 imports.

Only the direct auxiliary-classification gradient to decoder_core is removed.
Native TPA routing and the clipping coefficient computed from FULL gradients
are retained. AdamW momentum/weight decay are not erased by this intervention.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import math
from pathlib import Path
import re

import torch
import torch.distributed as dist

from tools.decoder_loss_audit_ops import isolated_rng
from tools.query_path_update_ops import canonical_name, key_group

START = 56800
HORIZON = 85200
ARMS = ("A", "B")


def state_digest(value):
    """Device-independent, type/shape-sensitive digest of trusted state dicts."""
    h = hashlib.sha256()
    def visit(v):
        if torch.is_tensor(v):
            t = v.detach().cpu().contiguous()
            h.update(str((str(t.dtype), tuple(t.shape))).encode())
            h.update(t.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(v, dict):
            h.update(b"dict")
            for k in sorted(v, key=lambda x: (type(x).__name__, str(x))):
                visit(k)
                visit(v[k])
        elif isinstance(v, (tuple, list)):
            h.update(type(v).__name__.encode())
            for item in v:
                visit(item)
        else:
            h.update(repr((type(v).__name__, v)).encode())
        h.update(b"\0")
    visit(value)
    return h.hexdigest()


def validate_resume(checkpoint):
    trainer = checkpoint.get("trainer", {})
    if checkpoint.get("iteration") != START - 1:
        raise ValueError("Requires the completed 8ep model_0056799.pth")
    for key, expected in (("iteration", START-1), ("lr_scheduler_max_iter", HORIZON),
                          ("gradient_accumulation_steps", 2)):
        if trainer.get(key) != expected:
            raise ValueError(f"Full resume requires trainer.{key}={expected}")
    optimizer = trainer.get("optimizer", {})
    scheduler = trainer.get("hooks", {}).get("LRScheduler", {})
    scaler = trainer.get("grad_scaler", {})
    groups = optimizer.get("param_groups", [])
    if not optimizer.get("state") or not groups or scheduler.get("last_epoch") != START:
        raise ValueError("Missing optimizer moments or original scheduler at 56800")
    if (not scaler or not math.isfinite(float(scaler.get("scale", float("nan"))))
            or scaler["scale"] <= 0):
        raise ValueError("Missing valid AMP GradScaler state; weights-only initialization is forbidden")
    if len(scheduler.get("base_lrs", [])) != len(groups):
        raise ValueError("Scheduler and optimizer group layout differ")
    for g, lr in zip(groups, scheduler["base_lrs"]):
        if not math.isfinite(lr) or lr <= 0 or not math.isclose(g["lr"], lr, rel_tol=1e-7):
            raise ValueError("8ep must remain on the original 12ep high-LR plateau")
        # Native get_default_optimizer_params gives normalization groups zero decay.
        if (tuple(g.get("betas", ())) != (0.9, 0.999)
                or not any(math.isclose(g.get("weight_decay", -1), wd) for wd in (0., 1e-4))):
            raise ValueError("Unexpected AdamW hyperparameters")
    return {"optimizer": state_digest(optimizer), "scheduler": state_digest(scheduler),
            "scaler": state_digest(scaler), "lrs": [g["lr"] for g in groups]}


def auxiliary_gradient(losses, parameters, accumulation):
    keys = sorted(k for k in losses if re.fullmatch(r"loss_class_\d+", k))
    if keys != [f"loss_class_{i}" for i in range(5)]:
        raise ValueError("Expected exactly five decoder auxiliary classification losses (not DN/encoder)")
    loss = sum(losses[k] for k in keys) / accumulation
    values = torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)
    return torch.cat([(v.detach() if v is not None else torch.zeros_like(p)).reshape(-1)
                      for p, v in zip(parameters, values)])


def synchronize_auxiliary(gradient):
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(gradient)
        gradient.div_(dist.get_world_size())
    if not torch.isfinite(gradient).all():
        raise FloatingPointError("Nonfinite synchronized auxiliary gradient")
    return gradient


def subtract_clipped_auxiliary(parameters, gradient, full_norm, max_norm, *, remove):
    if gradient.numel() != sum(p.numel() for p in parameters):
        raise ValueError("Auxiliary vector layout differs from decoder parameters")
    if not torch.isfinite(gradient).all():
        raise FloatingPointError("Nonfinite auxiliary gradient")
    if not torch.isfinite(full_norm).all():
        raise FloatingPointError("Nonfinite full-gradient clipping norm")
    coefficient = (max_norm / (full_norm + 1e-6)).clamp(max=1.)
    offset = 0
    with torch.no_grad():
        for p in parameters:
            n = p.numel()
            part = gradient[offset:offset+n].view_as(p)
            if remove:
                if p.grad is None:
                    if torch.count_nonzero(part):
                        raise ValueError("Auxiliary gradient exists without a full parameter gradient")
                else:
                    p.grad.sub_(part * coefficient.to(p.grad.device))
            offset += n
    if offset != gradient.numel():
        raise ValueError("Auxiliary vector layout differs from decoder parameters")
    return float(coefficient)


def pairing_fields(record):
    return {k: v for k, v in record.items() if k != "losses"}


def verify_pair(current, reference):
    if pairing_fields(current) != pairing_fields(reference):
        raise ValueError("A/B data, augmentation, FedLoss, RNG, LR, loss keys or AMP scale differ")
    if current["iteration"] == START:
        if current["losses"].keys() != reference["losses"].keys() or any(
                not math.isclose(v, reference["losses"][k], rel_tol=2e-4, abs_tol=2e-5)
                for k, v in current["losses"].items()):
            raise ValueError("First-update A/B forward differs BEFORE any intervention")


def make_trainer_class(native, manifest, arm, rank, *, rng_context=isolated_rng):
    """Use the native run_step, optimizer, APR routing and checkpoint machinery."""
    if arm not in ARMS:
        raise ValueError("Unknown arm")
    output = Path(manifest["output_dir"]) / arm
    source = Path(manifest["output_dir"]) / "A" / f"pairing_rank{rank}.jsonl"

    class PairedTrainer(native):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            if not self.separate_tpa_grad_clip or not self.tpa_conflict_projection or self.gradient_accumulation_steps != 2:
                raise ValueError("Paired run must preserve TPA routing, separate clipping and accumulation=2")
            if self.clip_grad_params is None or dict(self.clip_grad_params) != {"max_norm": .5, "norm_type": 2}:
                raise ValueError("Expected original .5 L2 clipping policy")
            raw = self.model.module if hasattr(self.model, "module") else self.model
            named = sorted((canonical_name(k), p) for k, p in raw.named_parameters()
                           if key_group(canonical_name(k)) == "decoder_core")
            if [k for k, _ in named] != manifest["decoder_keys"]:
                raise ValueError("Live decoder parameter layout differs from audited decoder_core")
            self.core_parameters = tuple(p for _, p in named)
            self.raw_model = raw
            self.actual_updates = 0
            self._micro = 0
            self._paired_iterator = None
            self._aux_gradient = None
            self.pair_stream = (output / f"pairing_rank{rank}.jsonl").open("x")
            self.update_stream = (output / f"updates_rank{rank}.jsonl").open("x")
            self.reference_stream = source.open() if arm == "B" else None
            self.initial_state = None
            self._device_index = torch.cuda.current_device() if next(raw.parameters()).is_cuda else None
            original_step = self.optimizer.step
            def counted_step(*a, **kw):
                result = original_step(*a, **kw)
                self.actual_updates += 1
                return result
            self.optimizer.step = counted_step
            self.optimizer.step._with_counter = True  # Preserve torch 1.12 LR scheduler wrapper marker.
            self.optimizer.step._wrapped_by_lr_sched = True

        def load_state_dict(self, state):
            super().load_state_dict(state)
            expected = manifest["resume"]
            loaded = {"optimizer": state_digest(self.optimizer.state_dict()),
                      "scheduler": state_digest(self.state_dict()["hooks"]["LRScheduler"]),
                      "scaler": state_digest(self.grad_scaler.state_dict())}
            if loaded != {k: expected[k] for k in loaded} or self.iter != START-1:
                raise ValueError("Optimizer/scheduler/scaler not exactly restored from 8ep")
            model_digest = state_digest({k.removeprefix("module."): v for k, v in self.model.state_dict().items()})
            if model_digest != manifest["model_digest"]:
                raise ValueError("Model weights not exactly restored from 8ep")
            self.initial_state = {**loaded, "model": model_digest}

        @property
        def _data_loader_iter(self):
            trainer = self
            if self._paired_iterator is None:
                class Iterator:
                    def __next__(self):
                        seed = manifest["seed"] + (trainer.iter-START)*128 + rank*8 + trainer._micro*2
                        trainer._forward_seed = seed + 1
                        with rng_context(seed, trainer._device_index):
                            data = next(super(PairedTrainer, trainer)._data_loader_iter)
                        if len(data) != 4:
                            raise ValueError("Expected four images per rank/microbatch")
                        mapped = []
                        for item in data:
                            classes = item["instances"].gt_classes
                            if any(c < 0 or c >= len(trainer.raw_model.novel_idx) or bool(trainer.raw_model.novel_idx[c])
                                   for c in classes.tolist()):
                                raise ValueError("Rare/invalid GT entered train_norare")
                            mapped.append({"image_id": int(item["image_id"]), "image": state_digest(item["image"]),
                                           "boxes": state_digest(item["instances"].gt_boxes.tensor),
                                           "classes": state_digest(classes)})
                        trainer._pending = {"iteration": trainer.iter, "micro": trainer._micro,
                                            "data_seed": seed, "forward_seed": seed+1, "mapped": mapped}
                        return data
                self._paired_iterator = Iterator()
            return self._paired_iterator

        def _compute_apr_gradients(self, loss_dict, *, gradient_scale=1.):
            g = auxiliary_gradient(loss_dict, self.core_parameters, self.gradient_accumulation_steps)
            self._aux_gradient = g if self._aux_gradient is None else self._aux_gradient + g
            return super()._compute_apr_gradients(loss_dict, gradient_scale=gradient_scale)

        def clip_model_grads(self):
            self._aux_gradient = synchronize_auxiliary(self._aux_gradient)
            norms = super().clip_model_grads()  # FULL gradients: do not recompute after subtraction.
            coefficient = subtract_clipped_auxiliary(self.core_parameters, self._aux_gradient, norms[0],
                                                     .5, remove=arm == "B")
            self.last_intervention = {"full_detector_norm": float(norms[0]), "clip_coefficient": coefficient,
                                      "aux_norm": float(self._aux_gradient.norm()), "remove": arm == "B"}
            self.last_intervention["core_post_norm"] = float(torch.stack([
                p.grad.norm() for p in self.core_parameters if p.grad is not None]).norm())
            if any(not math.isfinite(v) for k, v in self.last_intervention.items() if k != "remove"):
                raise FloatingPointError("Nonfinite intervention; refusing optimizer update")
            return norms

        @contextmanager
        def paired_forward(self):
            old_forward, old_filter = self.model.forward, self.raw_model.filter_content_info
            sampled = {}
            def filter_info(data):
                indices, targets = old_filter(data)
                sampled["fedloss"] = indices.detach().cpu().tolist()
                return indices, targets
            def forward(data):
                sampled.clear()
                with rng_context(self._forward_seed, self._device_index):
                    result = old_forward(data)
                    rng = {"cpu": state_digest(torch.random.get_rng_state())}
                    if self._device_index is not None:
                        rng["cuda"] = state_digest(torch.cuda.get_rng_state(self._device_index))
                if "fedloss" not in sampled:
                    raise ValueError("Missing native FedLoss sampling record")
                losses = {k: float(v.detach()) for k, v in result.items() if k.startswith("loss")}
                if any(not math.isfinite(v) for v in losses.values()):
                    raise FloatingPointError("Nonfinite loss; paired trial aborted")
                record = {**self._pending, **sampled, "rng_after": rng,
                          "lrs": [g["lr"] for g in self.optimizer.param_groups],
                          "amp_scale": self.grad_scaler.get_scale() if self.amp else None,
                          "loss_keys": sorted(losses), "losses": losses}
                if arm == "B":
                    line = self.reference_stream.readline()
                    if not line:
                        raise ValueError("A transcript ended before B")
                    verify_pair(record, json.loads(line))
                self.pair_stream.write(json.dumps(record) + "\n")
                self.pair_stream.flush()
                self._micro += 1
                return result
            self.model.forward, self.raw_model.filter_content_info = forward, filter_info
            try:
                yield
            finally:
                self.model.forward, self.raw_model.filter_content_info = old_forward, old_filter

        def run_step(self):
            if not START <= self.iter < START + manifest["updates"] or self.initial_state is None:
                raise ValueError("Paired run not restored or update budget exceeded")
            self._micro, self._aux_gradient = 0, None
            before = self.actual_updates
            with self.paired_forward():
                super().run_step()
            if self.actual_updates != before + 1 or self._micro != 2:
                raise ValueError("Skipped optimizer step or incorrect accumulation; not a 500-update paired trial")
            self.update_stream.write(json.dumps({"iteration": self.iter, "update": self.actual_updates,
                                                  **self.last_intervention}) + "\n")
            self.update_stream.flush()
            if self.actual_updates % 25 == 0 or self.actual_updates == manifest["updates"]:
                print(f"[paired {arm} rank={rank}] updates={self.actual_updates}/{manifest['updates']} "
                      f"iter={self.iter} clip={self.last_intervention}", flush=True)
            self._aux_gradient = None

        def state_dict(self):
            result = super().state_dict()
            result["decoder_aux_trial"] = {"arm": arm, "start": START, "updates": self.actual_updates,
                                           "manifest_fingerprint": manifest["fingerprint"],
                                           "clipping": "full-gradient-reference"}
            return result

    return PairedTrainer
