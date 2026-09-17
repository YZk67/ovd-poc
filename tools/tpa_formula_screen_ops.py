"""Paired short-run utilities for the three Eq. (2) training formulas.

The TPA geometry is frozen exactly in every arm.  Classification gradients may
still flow through the fixed prototypes into detector/query features, which is
the treatment this screen is intended to compare.
"""
from __future__ import annotations

from contextlib import contextmanager
import json
import math
from pathlib import Path

import torch

from lami_dino.prototype_ops import TPA_TRAIN_AGGREGATIONS
from tools.decoder_aux_ablation_ops import HORIZON, START, state_digest
from tools.decoder_loss_audit_ops import isolated_rng
from tools.tpa_geometry_audit_ops import BUFFERS, WEIGHTS


ARMS = {
    "A": "calibrated",
    "B": "calibrated_plus_logK",
    "C": "legacy",
}


def _raw_model(model):
    return model.module if hasattr(model, "module") else model


def _tpa_heads(model):
    raw = _raw_model(model)
    heads = list(raw.transformer.decoder.class_embed)
    if not heads or any(not hasattr(head, "tpa") for head in heads):
        raise ValueError("Expected TPA on every classification head")
    return heads


def tpa_parameters(model):
    unique = {}
    for head in _tpa_heads(model):
        for parameter in head.tpa.parameters():
            unique[id(parameter)] = parameter
    if not unique:
        raise ValueError("No live TPA parameters found")
    return tuple(unique.values())


def live_tpa_geometry(model):
    """Return one exact geometry state and reject unequal shared-head aliases."""
    selected = None
    for head in _tpa_heads(model):
        block = head.tpa.state_dict()
        current = {}
        for name in WEIGHTS + BUFFERS:
            value = block.get(name)
            if value is None or not value.is_floating_point() or not torch.isfinite(value).all():
                raise ValueError(f"Missing/nonfinite live TPA geometry tensor: {name}")
            current[name] = value.detach().cpu().clone()
        if selected is not None:
            for name in current:
                if (selected[name].shape != current[name].shape
                        or selected[name].dtype != current[name].dtype
                        or not torch.equal(selected[name], current[name])):
                    raise ValueError(f"Unequal live TPA aliases: {name}")
        selected = current
    return selected


def geometry_digest(model):
    return state_digest(live_tpa_geometry(model))


def set_training_formula(model, aggregation):
    if aggregation not in TPA_TRAIN_AGGREGATIONS:
        raise ValueError(f"Unknown TPA training aggregation: {aggregation}")
    for head in _tpa_heads(model):
        head.tpa_train_aggregation = aggregation


def verify_training_formula(model, aggregation):
    values = {head.tpa_train_aggregation for head in _tpa_heads(model)}
    if values != {aggregation}:
        raise ValueError(f"Live TPA training formulas differ: {values}")


def pairing_fields(record):
    # Loss values must differ: that is the intended formula intervention.
    return {key: value for key, value in record.items() if key != "losses"}


def verify_formula_pair(current, reference):
    if pairing_fields(current) != pairing_fields(reference):
        raise ValueError(
            "Formula arms differ in data, augmentation, FedLoss, RNG, LR, "
            "loss keys or AMP scale"
        )


def make_trainer_class(native, manifest, arm, rank, *, rng_context=isolated_rng):
    if arm not in ARMS:
        raise ValueError("Unknown formula arm")
    aggregation = ARMS[arm]
    output = Path(manifest["output_dir"]) / arm
    reference_path = Path(manifest["output_dir"]) / "A" / f"pairing_rank{rank}.jsonl"

    class FormulaTrainer(native):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            if (not self.separate_tpa_grad_clip or not self.tpa_conflict_projection
                    or self.gradient_accumulation_steps != 2):
                raise ValueError("Formula screen must preserve native TPA routing and accumulation=2")
            if self.clip_grad_params is None or dict(self.clip_grad_params) != {"max_norm": .5, "norm_type": 2}:
                raise ValueError("Expected original .5 L2 clipping policy")
            self.raw_model = _raw_model(self.model)
            set_training_formula(self.raw_model, aggregation)
            verify_training_formula(self.raw_model, aggregation)
            self.frozen_tpa_parameters = tpa_parameters(self.raw_model)
            self.actual_updates = 0
            self._micro = 0
            self._paired_iterator = None
            self.pair_stream = (output / f"pairing_rank{rank}.jsonl").open("x")
            self.update_stream = (output / f"updates_rank{rank}.jsonl").open("x")
            self.reference_stream = reference_path.open() if arm != "A" else None
            self.initial_state = None
            self.initial_geometry_digest = None
            self._device_index = (
                torch.cuda.current_device() if next(self.raw_model.parameters()).is_cuda else None
            )
            original_step = self.optimizer.step

            def frozen_tpa_step(*args, **kwargs):
                # Setting grad=None is stronger than LR=0: AdamW neither changes
                # weights nor advances moments/weight decay for these parameters.
                for parameter in self.frozen_tpa_parameters:
                    parameter.grad = None
                result = original_step(*args, **kwargs)
                self.actual_updates += 1
                return result

            self.optimizer.step = frozen_tpa_step
            # Torch 1.12's LR hook expects these markers after wrapping step().
            self.optimizer.step._with_counter = True
            self.optimizer.step._wrapped_by_lr_sched = True

        def _assert_geometry_frozen(self):
            current = geometry_digest(self.raw_model)
            if self.initial_geometry_digest is None or current != self.initial_geometry_digest:
                raise ValueError("TPA geometry changed in a frozen formula arm")
            return current

        def load_state_dict(self, state):
            super().load_state_dict(state)
            expected = manifest["resume"]
            loaded = {
                "optimizer": state_digest(self.optimizer.state_dict()),
                "scheduler": state_digest(self.state_dict()["hooks"]["LRScheduler"]),
                "scaler": state_digest(self.grad_scaler.state_dict()),
            }
            if loaded != {key: expected[key] for key in loaded} or self.iter != START - 1:
                raise ValueError("Optimizer/scheduler/scaler not exactly restored from 8ep")
            model_digest = state_digest({
                key.removeprefix("module."): value
                for key, value in self.model.state_dict().items()
            })
            if model_digest != manifest["model_digest"]:
                raise ValueError("Model weights not exactly restored from 8ep")
            self.initial_geometry_digest = geometry_digest(self.raw_model)
            if self.initial_geometry_digest != manifest["tpa_geometry_digest"]:
                raise ValueError("Live/source TPA geometry differs before training")
            verify_training_formula(self.raw_model, aggregation)
            self.initial_state = {**loaded, "model": model_digest}

        @property
        def _data_loader_iter(self):
            trainer = self
            if self._paired_iterator is None:
                class Iterator:
                    def __next__(self):
                        seed = manifest["seed"] + (trainer.iter - START) * 128 + rank * 8 + trainer._micro * 2
                        trainer._forward_seed = seed + 1
                        with rng_context(seed, trainer._device_index):
                            data = next(super(FormulaTrainer, trainer)._data_loader_iter)
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
                            "iteration": trainer.iter,
                            "micro": trainer._micro,
                            "data_seed": seed,
                            "forward_seed": seed + 1,
                            "mapped": mapped,
                        }
                        return data
                self._paired_iterator = Iterator()
            return self._paired_iterator

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
                verify_training_formula(self.raw_model, aggregation)
                with rng_context(self._forward_seed, self._device_index):
                    result = old_forward(data)
                    rng = {"cpu": state_digest(torch.random.get_rng_state())}
                    if self._device_index is not None:
                        rng["cuda"] = state_digest(torch.cuda.get_rng_state(self._device_index))
                if "fedloss" not in sampled:
                    raise ValueError("Missing native FedLoss sampling record")
                losses = {
                    key: float(value.detach())
                    for key, value in result.items()
                    if key.startswith("loss")
                }
                if any(not math.isfinite(value) for value in losses.values()):
                    raise FloatingPointError("Nonfinite loss; formula screen aborted")
                record = {
                    **self._pending,
                    **sampled,
                    "rng_after": rng,
                    "lrs": [group["lr"] for group in self.optimizer.param_groups],
                    "amp_scale": self.grad_scaler.get_scale() if self.amp else None,
                    "loss_keys": sorted(losses),
                    "losses": losses,
                }
                if self.reference_stream is not None:
                    line = self.reference_stream.readline()
                    if not line:
                        raise ValueError("A transcript ended before comparison arm")
                    verify_formula_pair(record, json.loads(line))
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
            if not START <= self.iter < START + manifest["updates"] or self.initial_state is None:
                raise ValueError("Formula arm not restored or update budget exceeded")
            self._micro = 0
            before = self.actual_updates
            with self.paired_forward():
                super().run_step()
            if self.actual_updates != before + 1 or self._micro != 2:
                raise ValueError("Skipped optimizer step or incorrect accumulation")
            frozen_digest = self._assert_geometry_frozen()
            row = {
                "iteration": self.iter,
                "update": self.actual_updates,
                "aggregation": aggregation,
                "tpa_geometry_digest": frozen_digest,
            }
            self.update_stream.write(json.dumps(row) + "\n")
            self.update_stream.flush()
            if self.actual_updates % 25 == 0 or self.actual_updates == manifest["updates"]:
                print(
                    f"[formula {arm}/{aggregation} rank={rank}] "
                    f"updates={self.actual_updates}/{manifest['updates']} TPA=frozen",
                    flush=True,
                )

        def state_dict(self):
            result = super().state_dict()
            result["tpa_formula_trial"] = {
                "arm": arm,
                "aggregation": aggregation,
                "start": START,
                "updates": self.actual_updates,
                "manifest_fingerprint": manifest["fingerprint"],
                "tpa_geometry_frozen": True,
            }
            return result

    return FormulaTrainer
