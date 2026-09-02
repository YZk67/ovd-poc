import math

import pytest
import torch
import torch.nn.functional as F

from lami_dino.models.rpsa import (
    RPSAModule,
    select_high_confidence_tokens,
    weighted_infoNCE,
)


def test_token_selection_excludes_padding_and_never_duplicates_indices():
    batch, tokens, classes, dim = 2, 8, 4, 3
    feats = torch.arange(batch * tokens * dim, dtype=torch.float32).view(
        batch, tokens, dim
    )
    probs = torch.zeros(batch, tokens, classes)
    probs[..., 0] = torch.tensor(
        [[0.1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3],
         [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2]]
    )
    valid = torch.tensor(
        [[False, True, True, True, True, False, False, False],
         [True, True, True, True, True, True, False, False]]
    )

    selected, selected_probs, indices = select_high_confidence_tokens(
        feats,
        probs,
        valid_mask=valid,
        topk=3,
        confidence_threshold=0.0,
        min_tokens=2,
    )

    assert selected.shape == (batch, 3, dim)
    assert selected_probs.shape == (batch, 3, classes)
    for row in range(batch):
        assert indices[row].unique().numel() == indices.size(1)
        assert valid[row, indices[row]].all()


def test_token_selection_rejects_too_strict_threshold_without_fallback_duplicates():
    feats = torch.randn(1, 6, 4)
    probs = torch.full((1, 6, 3), 0.1)
    probs[0, 0, 0] = 0.9

    with pytest.raises(ValueError, match="not enough distinct"):
        select_high_confidence_tokens(
            feats,
            probs,
            topk=6,
            confidence_threshold=0.5,
            min_tokens=3,
        )


def test_fixed_and_adaptive_background_thresholds_are_combined():
    mu = F.normalize(torch.randn(1, 2, 4), dim=-1)
    prototypes = F.normalize(torch.randn(2, 1, 4), dim=-1)
    pi = torch.tensor([[[0.04, 0.01], [0.8, 0.2]]])

    _, stats = weighted_infoNCE(
        mu,
        prototypes,
        pi,
        bg_thresh=0.1,
        adaptive_bg=torch.tensor([[0.02]]),
    )

    torch.testing.assert_close(stats["rpsa_bg_ratio"], torch.tensor(0.5))


def test_rpsa_is_invariant_to_uniform_prototype_duplication():
    torch.manual_seed(0)
    mu = F.normalize(torch.randn(2, 3, 8), dim=-1)
    single = F.normalize(torch.randn(5, 1, 8), dim=-1)
    duplicated = single.expand(-1, 5, -1).contiguous()
    pi = torch.softmax(torch.randn(2, 3, 5), dim=-1)

    loss_k1, _ = weighted_infoNCE(mu, single, pi, tau=0.07, bg_thresh=None)
    loss_k5, _ = weighted_infoNCE(mu, duplicated, pi, tau=0.07, bg_thresh=None)

    torch.testing.assert_close(loss_k1, loss_k5, atol=1e-5, rtol=1e-5)
    assert math.isfinite(float(loss_k5))


def test_full_rpsa_module_propagates_finite_gradients():
    torch.manual_seed(1)
    regions = torch.randn(2, 12, 8, requires_grad=True)
    prototypes = torch.randn(5, 3, 8, requires_grad=True)
    token_mask = torch.softmax(torch.randn(2, 12, 5), dim=-1)
    module = RPSAModule(K=3, em_iters=1, bg_thresh=None, bg_percentile=0.0)

    loss, stats, extras = module(regions, prototypes, token_mask)
    loss.backward()

    assert torch.isfinite(loss)
    assert regions.grad is not None and torch.isfinite(regions.grad).all()
    assert prototypes.grad is not None and torch.isfinite(prototypes.grad).all()
    assert float(stats["rpsa_valid_clusters"]) > 0
    assert extras["centers_mu"].shape == (2, 3, 8)
