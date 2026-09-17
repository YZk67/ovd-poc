import math

import pytest
import torch

from tools.audit_pairing_optimizer_sources import _loss_gradients, _reconstruction_error
from tools.pairing_optimizer_source_ops import (
    SOURCES,
    adamw_delta,
    clip_coefficient,
    counterfactual_norm,
    split_loss_keys,
    summarize_screen,
    vector_metrics,
)


def native_losses(x):
    return {
        "loss_class": x.square(),
        **{f"loss_class_{i}": (x * (i + 2)).square() for i in range(5)},
        "loss_class_dn": (x * 7).square(),
        **{f"loss_class_dn_{i}": (x * (i + 8)).square() for i in range(5)},
        "loss_class_enc": (x * 14).square(),
        "loss_bbox": (x * 15).square(),
        "loss_giou": (x * 16).square(),
        **{f"loss_bbox_{i}": (x * (i + 17)).square() for i in range(5)},
        **{f"loss_giou_{i}": (x * (i + 22)).square() for i in range(5)},
        "loss_bbox_enc": (x * 27).square(),
        "loss_giou_enc": (x * 28).square(),
        "loss_bbox_dn": (x * 29).square(),
        "loss_giou_dn": (x * 30).square(),
        **{f"loss_bbox_dn_{i}": (x * (i + 31)).square() for i in range(5)},
        **{f"loss_giou_dn_{i}": (x * (i + 36)).square() for i in range(5)},
        "loss_apr": (x * 41).square(),
        "loss_rpsa": (x * 42).square(),
    }


def test_split_loss_keys_is_exhaustive_and_rejects_unknown_loss():
    x = torch.tensor(1.0)
    groups = split_loss_keys(native_losses(x))
    assert tuple(groups) == SOURCES
    assert groups["class_final"] == ["loss_class"]
    assert len(groups["class_aux"]) == 5
    assert len(groups["class_dn"]) == 6
    assert groups["class_encoder"] == ["loss_class_enc"]
    assert len(groups["box"]) == 26
    assert groups["apr"] == ["loss_apr"]
    assert groups["rpsa"] == ["loss_rpsa"]
    bad = native_losses(x)
    bad["loss_unclassified"] = x
    with pytest.raises(ValueError, match="Unclassified"):
        split_loss_keys(bad)


def test_component_gradients_reconstruct_full_target_gradient():
    query = torch.nn.Parameter(torch.tensor([0.4, -0.3]))
    bank = torch.nn.Parameter(torch.tensor([0.2, 0.7]))
    scalar = query.sum() + 2 * bank.sum()
    losses = native_losses(scalar)
    targets = {"query": [("query", query)], "bank": [("bank", bank)]}
    components, connections, keys = _loss_gradients(losses, targets, accumulation=2)
    total = sum(losses.values()) / 2
    full = torch.autograd.grad(total, (query, bank))
    assert set(keys) == set(SOURCES)
    assert all(connections[source][target] == [True]
               for source in SOURCES for target in ("query", "bank"))
    assert _reconstruction_error(
        [full[0]], {source: components[source]["query"] for source in SOURCES}
    ) < 1e-6
    assert _reconstruction_error(
        [full[1]], {source: components[source]["bank"] for source in SOURCES}
    ) < 1e-6


def test_counterfactual_norm_and_clip_coefficient():
    full_parts = [torch.tensor([3.0, 0.0])]
    after = [torch.tensor([0.0, 5.0])]
    # Full vector also contains an untouched [0, 12] block: sqrt(3^2+12^2) -> 13.
    assert counterfactual_norm(math.sqrt(3**2 + 12**2), full_parts, after) == pytest.approx(13.0)
    assert clip_coefficient(2.0, 0.5) == pytest.approx(0.25, rel=1e-6)
    assert clip_coefficient(0.1, 0.5) == 1.0


def test_adamw_delta_matches_torch_optimizer_next_step():
    parameter = torch.nn.Parameter(torch.tensor([0.7, -1.2], dtype=torch.float64))
    optimizer = torch.optim.AdamW(
        [parameter], lr=3e-3, betas=(0.8, 0.95), eps=1e-7, weight_decay=0.02
    )
    parameter.grad = torch.tensor([0.4, -0.6], dtype=torch.float64)
    optimizer.step()
    optimizer.zero_grad()
    before = parameter.detach().clone()
    gradient = torch.tensor([-0.2, 0.9], dtype=torch.float64)
    predicted = adamw_delta(parameter, gradient, optimizer.state[parameter], optimizer.param_groups[0])
    parameter.grad = gradient.clone()
    optimizer.step()
    assert torch.allclose(predicted, parameter.detach() - before, atol=1e-14, rtol=1e-12)


def test_vector_metrics_and_screening_ranking_do_not_claim_causality():
    normal = [torch.tensor([2.0, 0.0])]
    counterfactual = [torch.tensor([1.0, 0.0])]
    historical = [torch.tensor([3.0, 0.0])]
    metrics = vector_metrics(normal, counterfactual, historical)
    assert metrics["source_effect_historical_cosine"] == pytest.approx(1.0)
    assert metrics["source_effect_historical_projection"] == pytest.approx(1 / 3)
    window = {"sources": {}}
    for index, source in enumerate(SOURCES):
        window["sources"][source] = {}
        for target in ("query", "bank"):
            window["sources"][source][target] = {
                **metrics,
                "source_gradient_l2": 1.0 if index == 0 else 0.0,
            }
            window["sources"][source][target]["source_effect_historical_cosine"] = 1.0 - index
    rows = summarize_screen([window])
    assert rows[0]["source"] == SOURCES[0]
    assert rows[2]["mean_effect_historical_cosine"] == 0.0
    assert "causal" not in " ".join(rows[0]).lower()
