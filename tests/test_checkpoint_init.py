import pytest
import torch
from torch import nn

from lami_dino.checkpoint_init import (
    load_backbone_only,
    load_trusted_torch_file,
    select_backbone_state_dict,
    validate_backbone_trainable_scope,
)


class TinyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Linear(3, 4)
        self.norm = nn.LayerNorm(4)


def test_select_backbone_keys_ignores_detector_tensors():
    backbone = TinyBackbone()
    source = {
        "model": {
            "backbone.stem.weight": torch.full_like(backbone.stem.weight, 2.0),
            "backbone.stem.bias": torch.full_like(backbone.stem.bias, 3.0),
            "backbone.norm.weight": torch.full_like(backbone.norm.weight, 4.0),
            "backbone.norm.bias": torch.full_like(backbone.norm.bias, 5.0),
            "transformer.encoder.weight": torch.randn(4, 4),
            "classifier.weight": torch.randn(7, 4),
        }
    }

    selected, report = select_backbone_state_dict(source, backbone.state_dict())

    assert set(selected) == set(backbone.state_dict())
    assert report["loaded_tensor_count"] == 4
    assert report["ignored_tensor_count"] == 2
    assert report["parameter_coverage"] == 1.0


def test_load_backbone_only_accepts_wrapped_and_direct_keys(tmp_path):
    backbone = TinyBackbone()
    checkpoint = {
        "state_dict": {
            "module.model.backbone.stem.weight": torch.full_like(
                backbone.stem.weight, 7.0
            ),
            "stem.bias": torch.full_like(backbone.stem.bias, 8.0),
            "norm.weight": torch.full_like(backbone.norm.weight, 9.0),
            "norm.bias": torch.full_like(backbone.norm.bias, 10.0),
            "module.model.decoder.weight": torch.randn(3, 3),
        }
    }
    path = tmp_path / "clip_backbone.pth"
    torch.save(checkpoint, path)

    report = load_backbone_only(backbone, str(path))

    torch.testing.assert_close(backbone.stem.weight, torch.full_like(backbone.stem.weight, 7.0))
    torch.testing.assert_close(backbone.stem.bias, torch.full_like(backbone.stem.bias, 8.0))
    torch.testing.assert_close(backbone.norm.weight, torch.full_like(backbone.norm.weight, 9.0))
    torch.testing.assert_close(backbone.norm.bias, torch.full_like(backbone.norm.bias, 10.0))
    assert report["ignored_tensor_count"] == 1


def test_load_backbone_only_rejects_low_coverage(tmp_path):
    backbone = TinyBackbone()
    path = tmp_path / "partial.pth"
    torch.save(
        {"model": {"backbone.norm.bias": torch.zeros_like(backbone.norm.bias)}},
        path,
    )

    with pytest.raises(ValueError, match="coverage is too low"):
        load_backbone_only(backbone, str(path), min_parameter_coverage=0.95)


def test_trusted_loader_accepts_legacy_module_checkpoint(tmp_path):
    path = tmp_path / "legacy_modules.pth"
    torch.save([nn.Identity(), nn.Linear(2, 3)], path)

    loaded = load_trusted_torch_file(path)

    assert isinstance(loaded[0], nn.Identity)
    assert isinstance(loaded[1], nn.Linear)


def test_backbone_scope_allows_only_detection_output_norms():
    class ScopedBackbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.stage = nn.Linear(2, 2)
            self.norm1 = nn.LayerNorm(2)

    backbone = ScopedBackbone()
    for parameter in backbone.stage.parameters():
        parameter.requires_grad = False

    trainable = validate_backbone_trainable_scope(backbone, "output_norm_only")
    assert set(trainable) == {"norm1.weight", "norm1.bias"}

    backbone.stage.weight.requires_grad = True
    with pytest.raises(ValueError, match="violated"):
        validate_backbone_trainable_scope(backbone, "output_norm_only")
