"""Checkpoint helpers for reproducible model initialization.

The paper initializes only the ConvNeXt-L visual backbone from CLIP.  Loading a
checkpoint through the detector-level checkpointer cannot enforce that scope:
any detector keys present in the file would be restored as well.  This module
selects keys against the backbone's own state dict before loading, so the
detector, transformer, and prototype modules necessarily keep their fresh
initialization.
"""

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Union

import torch


def load_trusted_torch_file(path: Union[str, Path], *, map_location="cpu") -> Any:
    """Load a trusted local PyTorch artifact across old and new versions.

    PyTorch 2.6 changed ``torch.load`` to default to ``weights_only=True``.
    The repository's legacy CLIP head contains serialized ``nn.Module`` objects,
    so formal runs must opt into the old trusted-file behavior explicitly.
    """

    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:  # PyTorch versions before the weights_only argument.
        return torch.load(path, map_location=map_location)


def validate_backbone_trainable_scope(
    backbone: torch.nn.Module,
    scope: str,
) -> tuple[str, ...]:
    """Validate the paper run's declared ConvNeXt fine-tuning scope."""

    trainable = tuple(name for name, param in backbone.named_parameters() if param.requires_grad)
    if scope == "full":
        return trainable
    if scope == "frozen":
        unexpected = trainable
    elif scope == "output_norm_only":
        # The LaMI ConvNeXt trunk freezes downsample_layers/stages and leaves the
        # detection output norms (norm1/norm2/norm3) trainable at backbone LR.
        unexpected = tuple(name for name in trainable if not name.startswith("norm"))
    else:
        raise ValueError(
            "backbone_trainable_scope must be 'full', 'frozen', or "
            f"'output_norm_only', got {scope!r}"
        )
    if unexpected:
        raise ValueError(
            f"backbone scope {scope!r} violated by trainable parameters: "
            + ", ".join(unexpected[:10])
        )
    return trainable


def _checkpoint_state_dict(checkpoint: Any) -> Mapping[str, Any]:
    if not isinstance(checkpoint, Mapping):
        raise ValueError("checkpoint must be a mapping containing a state dict")

    for key in ("model", "state_dict"):
        value = checkpoint.get(key)
        if isinstance(value, Mapping):
            return value

    # Native PyTorch state-dict checkpoints have tensor names at the top level.
    if any(isinstance(key, str) and torch.is_tensor(value) for key, value in checkpoint.items()):
        return checkpoint

    raise ValueError("checkpoint does not contain a model/state_dict tensor mapping")


def _remove_wrapper_prefixes(key: str) -> str:
    # DDP and some training frameworks add one or both of these wrappers.
    changed = True
    while changed:
        changed = False
        for prefix in ("module.", "model."):
            if key.startswith(prefix):
                key = key[len(prefix) :]
                changed = True
    return key


def select_backbone_state_dict(
    checkpoint: Any,
    target_state_dict: Mapping[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Select compatible ConvNeXt keys without exposing any detector weights.

    Converted detector checkpoints usually store keys as ``backbone.<name>``;
    backbone-only checkpoints may store ``<name>`` directly.  Both forms are
    accepted, but a source tensor is loaded only when its name and shape match
    the target backbone state dict.
    """

    source_state = _checkpoint_state_dict(checkpoint)
    selected: dict[str, torch.Tensor] = {}
    shape_mismatches: list[str] = []
    tensor_count = 0

    for source_key, value in source_state.items():
        if not isinstance(source_key, str) or not torch.is_tensor(value):
            continue
        tensor_count += 1
        key = _remove_wrapper_prefixes(source_key)
        if key.startswith("backbone."):
            key = key[len("backbone.") :]

        target_value = target_state_dict.get(key)
        if target_value is None:
            continue
        if tuple(value.shape) != tuple(target_value.shape):
            shape_mismatches.append(source_key)
            continue
        selected[key] = value

    if not selected:
        raise ValueError(
            "checkpoint contains no tensors compatible with the configured backbone"
        )

    loaded_numel = sum(target_state_dict[key].numel() for key in selected)
    target_numel = sum(value.numel() for value in target_state_dict.values())
    coverage = loaded_numel / max(target_numel, 1)
    report = {
        "source_tensor_count": tensor_count,
        "loaded_tensor_count": len(selected),
        "target_tensor_count": len(target_state_dict),
        "ignored_tensor_count": tensor_count - len(selected),
        "shape_mismatches": tuple(shape_mismatches),
        "parameter_coverage": coverage,
    }
    return selected, report


def load_backbone_only(
    backbone: torch.nn.Module,
    checkpoint_path: str,
    *,
    min_parameter_coverage: float = 0.95,
) -> dict[str, Any]:
    """Load only backbone tensors and return an auditable coverage report."""

    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"backbone initialization checkpoint not found: {path}")

    checkpoint = load_trusted_torch_file(path)

    target_state = backbone.state_dict()
    selected, report = select_backbone_state_dict(checkpoint, target_state)
    if report["parameter_coverage"] < min_parameter_coverage:
        raise ValueError(
            "backbone checkpoint coverage is too low: "
            f"{report['parameter_coverage']:.2%} < {min_parameter_coverage:.2%}; "
            "refusing to start a partially initialized paper run"
        )

    incompatible = backbone.load_state_dict(selected, strict=False)
    report["missing_keys"] = tuple(incompatible.missing_keys)
    report["unexpected_keys"] = tuple(incompatible.unexpected_keys)
    report["checkpoint_path"] = str(path)
    return report
