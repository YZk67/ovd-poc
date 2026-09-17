"""Paired DN-classification intervention built on a verified native A arm.

Only the direct gradient of ``loss_class_dn`` and ``loss_class_dn_0..4`` to
``decoder_core`` is removed.  The source A trajectory is immutable and reused
from the completed auxiliary-gradient trial to avoid retraining the same
native control.
"""
from __future__ import annotations

from contextlib import contextmanager
import json
import math
from pathlib import Path
import re

import torch

from tools.decoder_aux_ablation_ops import (
    HORIZON, START, isolated_rng, state_digest, subtract_clipped_auxiliary,
    synchronize_auxiliary, verify_pair,
)
from tools.query_path_update_ops import canonical_name, key_group


def dn_classification_gradient(losses, parameters, accumulation):
    """Differentiate all six DN classification terms, and no other loss."""
    keys = sorted(k for k in losses if re.fullmatch(r"loss_class_dn(?:_\d+)?", k))
    expected = ["loss_class_dn", *(f"loss_class_dn_{i}" for i in range(5))]
    if set(keys) != set(expected) or len(keys) != len(expected):
        raise ValueError("Expected final plus five auxiliary DN classification losses")
    loss = sum(losses[key] for key in expected) / accumulation
    values = torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)
    return torch.cat([
        (value.detach() if value is not None else torch.zeros_like(parameter)).reshape(-1)
        for parameter, value in zip(parameters, values)
    ])


def make_trainer_class(native, manifest, rank, *, rng_context=isolated_rng):
    """Return B trainer paired to the immutable reference-A transcript."""
    output = Path(manifest["output_dir"]) / "B"
    source = Path(manifest["reference_A"]["transcripts"][str(rank)]["path"])

    class PairedDNTrainer(native):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            if (not self.separate_tpa_grad_clip or not self.tpa_conflict_projection
                    or self.gradient_accumulation_steps != 2):
                raise ValueError("Paired run must preserve TPA routing, separate clipping and accumulation=2")
            if self.clip_grad_params is None or dict(self.clip_grad_params) != {
                    "max_norm": .5, "norm_type": 2}:
                raise ValueError("Expected original .5 L2 clipping policy")
            raw = self.model.module if hasattr(self.model, "module") else self.model
            named = sorted(
                (canonical_name(name), parameter)
                for name, parameter in raw.named_parameters()
                if key_group(canonical_name(name)) == "decoder_core"
            )
            if [name for name, _ in named] != manifest["decoder_keys"]:
                raise ValueError("Live decoder parameter layout differs from audited decoder_core")
            self.core_parameters = tuple(parameter for _, parameter in named)
            self.raw_model = raw
            self.actual_updates = 0
            self._micro = 0
            self._paired_iterator = None
            self._dn_gradient = None
            self.pair_stream = (output / f"pairing_rank{rank}.jsonl").open("x")
            self.update_stream = (output / f"updates_rank{rank}.jsonl").open("x")
            self.reference_stream = source.open()
            self.initial_state = None
            self._device_index = torch.cuda.current_device() if next(raw.parameters()).is_cuda else None
            original_step = self.optimizer.step

            def counted_step(*args, **kwargs):
                result = original_step(*args, **kwargs)
                self.actual_updates += 1
                return result

            self.optimizer.step = counted_step
            # Preserve the torch 1.12 LR-scheduler wrapper markers.
            self.optimizer.step._with_counter = True
            self.optimizer.step._wrapped_by_lr_sched = True

        def load_state_dict(self, state):
            super().load_state_dict(state)
            expected = manifest["resume"]
            loaded = {
                "optimizer": state_digest(self.optimizer.state_dict()),
                "scheduler": state_digest(self.state_dict()["hooks"]["LRScheduler"]),
                "scaler": state_digest(self.grad_scaler.state_dict()),
            }
            if loaded != {key: expected[key] for key in loaded} or self.iter != START-1:
                raise ValueError("Optimizer/scheduler/scaler not exactly restored from 8ep")
            model_digest = state_digest({
                name.removeprefix("module."): value
                for name, value in self.model.state_dict().items()
            })
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
                            data = next(super(PairedDNTrainer, trainer)._data_loader_iter)
                        if len(data) != 4:
                            raise ValueError("Expected four images per rank/microbatch")
                        mapped = []
                        for item in data:
                            classes = item["instances"].gt_classes
                            if any(
                                category < 0
                                or category >= len(trainer.raw_model.novel_idx)
                                or bool(trainer.raw_model.novel_idx[category])
                                for category in classes.tolist()
                            ):
                                raise ValueError("Rare/invalid GT entered train_norare")
                            mapped.append({
                                "image_id": int(item["image_id"]),
                                "image": state_digest(item["image"]),
                                "boxes": state_digest(item["instances"].gt_boxes.tensor),
                                "classes": state_digest(classes),
                            })
                        trainer._pending = {
                            "iteration": trainer.iter, "micro": trainer._micro,
                            "data_seed": seed, "forward_seed": seed+1, "mapped": mapped,
                        }
                        return data

                self._paired_iterator = Iterator()
            return self._paired_iterator

        def _compute_apr_gradients(self, loss_dict, *, gradient_scale=1.):
            gradient = dn_classification_gradient(
                loss_dict, self.core_parameters, self.gradient_accumulation_steps)
            self._dn_gradient = (
                gradient if self._dn_gradient is None else self._dn_gradient + gradient)
            return super()._compute_apr_gradients(loss_dict, gradient_scale=gradient_scale)

        def clip_model_grads(self):
            self._dn_gradient = synchronize_auxiliary(self._dn_gradient)
            # FULL gradients determine the unchanged native clipping coefficient.
            norms = super().clip_model_grads()
            coefficient = subtract_clipped_auxiliary(
                self.core_parameters, self._dn_gradient, norms[0], .5, remove=True)
            self.last_intervention = {
                "full_detector_norm": float(norms[0]),
                "clip_coefficient": coefficient,
                "dn_norm": float(self._dn_gradient.norm()),
                "remove": True,
            }
            self.last_intervention["core_post_norm"] = float(torch.stack([
                parameter.grad.norm()
                for parameter in self.core_parameters if parameter.grad is not None
            ]).norm())
            if any(
                not math.isfinite(value)
                for key, value in self.last_intervention.items() if key != "remove"
            ):
                raise FloatingPointError("Nonfinite intervention; refusing optimizer update")
            return norms

        @contextmanager
        def paired_forward(self):
            old_forward = self.model.forward
            old_filter = self.raw_model.filter_content_info
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
                losses = {
                    key: float(value.detach())
                    for key, value in result.items() if key.startswith("loss")
                }
                if any(not math.isfinite(value) for value in losses.values()):
                    raise FloatingPointError("Nonfinite loss; paired trial aborted")
                record = {
                    **self._pending, **sampled, "rng_after": rng,
                    "lrs": [group["lr"] for group in self.optimizer.param_groups],
                    "amp_scale": self.grad_scaler.get_scale() if self.amp else None,
                    "loss_keys": sorted(losses), "losses": losses,
                }
                line = self.reference_stream.readline()
                if not line:
                    raise ValueError("Reference A transcript ended before DN B")
                verify_pair(record, json.loads(line))
                self.pair_stream.write(json.dumps(record) + "\n")
                self.pair_stream.flush()
                self._micro += 1
                return result

            self.model.forward = forward
            self.raw_model.filter_content_info = filter_info
            try:
                yield
            finally:
                self.model.forward = old_forward
                self.raw_model.filter_content_info = old_filter

        def run_step(self):
            if not START <= self.iter < START+manifest["updates"] or self.initial_state is None:
                raise ValueError("Paired run not restored or update budget exceeded")
            self._micro, self._dn_gradient = 0, None
            before = self.actual_updates
            with self.paired_forward():
                super().run_step()
            if self.actual_updates != before+1 or self._micro != 2:
                raise ValueError("Skipped optimizer step or incorrect accumulation")
            self.update_stream.write(json.dumps({
                "iteration": self.iter, "update": self.actual_updates,
                **self.last_intervention,
            }) + "\n")
            self.update_stream.flush()
            if self.actual_updates % 25 == 0 or self.actual_updates == manifest["updates"]:
                print(
                    f"[paired DN B rank={rank}] updates={self.actual_updates}/"
                    f"{manifest['updates']} iter={self.iter} clip={self.last_intervention}",
                    flush=True,
                )
            self._dn_gradient = None

        def state_dict(self):
            result = super().state_dict()
            result["decoder_dn_trial"] = {
                "arm": "B", "start": START, "updates": self.actual_updates,
                "manifest_fingerprint": manifest["fingerprint"],
                "source": "loss_class_dn_and_0_through_4",
                "clipping": "full-gradient-reference",
            }
            return result

    return PairedDNTrainer
