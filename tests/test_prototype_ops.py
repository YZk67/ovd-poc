import torch
import torch.nn.functional as F

from lami_dino.prototype_ops import (
    calibrated_logmeanexp_similarity,
    prototype_task_view,
    route_conflicting_task_gradient,
    soft_category_prototype_fusion,
)


def test_prototype_stabilization_blocks_only_task_gradient():
    prototypes = torch.randn(3, 5, 8, requires_grad=True)

    forming = prototype_task_view(
        prototypes,
        iteration=9,
        stabilization_steps=10,
        training=True,
    )
    joint = prototype_task_view(
        prototypes,
        iteration=10,
        stabilization_steps=10,
        task_gradient_scale=1.0,
        training=True,
    )
    evaluating = prototype_task_view(
        prototypes,
        iteration=0,
        stabilization_steps=10,
        training=False,
    )

    assert not forming.requires_grad
    assert joint is prototypes
    assert evaluating is prototypes


def test_prototype_task_view_scales_gradient_without_changing_forward():
    prototypes = torch.randn(3, 5, 8, requires_grad=True)
    task_view = prototype_task_view(
        prototypes,
        iteration=10,
        stabilization_steps=10,
        task_gradient_scale=0.1,
        training=True,
    )

    torch.testing.assert_close(task_view, prototypes)
    task_view.sum().backward()
    torch.testing.assert_close(prototypes.grad, torch.full_like(prototypes, 0.1))


def test_prototype_task_view_rejects_invalid_gradient_scale():
    prototypes = torch.randn(3, 5, 8)
    for scale in (-0.1, 1.1):
        try:
            prototype_task_view(
                prototypes,
                iteration=10,
                stabilization_steps=10,
                task_gradient_scale=scale,
                training=True,
            )
        except ValueError as error:
            assert "task_gradient_scale" in str(error)
        else:
            raise AssertionError(f"expected invalid task gradient scale {scale} to fail")


def test_conflict_projection_removes_only_opposing_task_component():
    apr_gradient = torch.tensor([1.0, 0.0])
    task_gradient = torch.tensor([-2.0, 3.0])
    total_gradient = apr_gradient + task_gradient

    routed, stats = route_conflicting_task_gradient(total_gradient, apr_gradient)

    torch.testing.assert_close(routed, torch.tensor([1.0, 3.0]))
    routed_task = routed - apr_gradient
    assert torch.dot(routed_task, apr_gradient).abs() < 1e-6
    assert routed_task[1] == task_gradient[1]
    assert stats["conflict_projected"].item() == 1.0
    assert stats["task_apr_cosine"].item() < 0.0


def test_conflict_projection_preserves_full_aligned_task_gradient():
    apr_gradient = torch.tensor([1.0, 0.0])
    task_gradient = torch.tensor([2.0, 3.0])
    total_gradient = apr_gradient + task_gradient

    routed, stats = route_conflicting_task_gradient(total_gradient, apr_gradient)

    torch.testing.assert_close(routed, total_gradient)
    assert stats["conflict_projected"].item() == 0.0


def test_conflict_projection_is_noop_without_apr_gradient():
    total_gradient = torch.tensor([-2.0, 3.0])
    apr_gradient = torch.zeros_like(total_gradient)

    routed, stats = route_conflicting_task_gradient(total_gradient, apr_gradient)

    torch.testing.assert_close(routed, total_gradient)
    assert stats["conflict_projected"].item() == 0.0


def test_logmeanexp_is_invariant_to_duplicate_prototypes():
    torch.manual_seed(0)
    features = F.normalize(torch.randn(2, 7, 16), dim=-1)
    single = F.normalize(torch.randn(5, 1, 16), dim=-1)
    duplicated = single.expand(-1, 5, -1).contiguous()

    logits_k1 = calibrated_logmeanexp_similarity(
        features, single, temperature=0.07, logit_scale=50.0
    )
    logits_k5 = calibrated_logmeanexp_similarity(
        features, duplicated, temperature=0.07, logit_scale=50.0
    )

    torch.testing.assert_close(logits_k1, logits_k5, atol=1e-5, rtol=1e-5)


def test_logmeanexp_k1_reduces_to_scaled_cosine():
    torch.manual_seed(1)
    features = F.normalize(torch.randn(2, 3, 8), dim=-1)
    prototypes = F.normalize(torch.randn(4, 1, 8), dim=-1)

    actual = calibrated_logmeanexp_similarity(
        features, prototypes, temperature=0.2, logit_scale=17.0
    )
    expected = 17.0 * torch.einsum("bqd,cd->bqc", features, prototypes[:, 0])
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)


def test_soft_fusion_keeps_top_r_categories_and_normalized_weights():
    torch.manual_seed(2)
    b, q, c, k, d = 2, 4, 6, 3, 8
    regions = torch.randn(b, q, d)
    logits = torch.randn(b, q, c)
    prototypes = torch.randn(c, k, d)

    fused, category_ids, gamma, alpha = soft_category_prototype_fusion(
        regions,
        logits,
        prototypes,
        category_topk=3,
        category_temperature=1.0,
        prototype_temperature=0.15,
    )

    assert fused.shape == (b, q, d)
    assert category_ids.shape == (b, q, 3)
    assert gamma.shape == (b, q, 3)
    assert alpha.shape == (b, q, 3, k)
    torch.testing.assert_close(gamma.sum(-1), torch.ones(b, q))
    torch.testing.assert_close(alpha.sum(-1), torch.ones(b, q, 3))
    torch.testing.assert_close(category_ids, logits.topk(3, dim=-1).indices)


def test_soft_fusion_top1_identical_prototypes_reduces_to_that_direction():
    torch.manual_seed(3)
    b, q, c, k, d = 1, 5, 4, 5, 8
    regions = torch.randn(b, q, d)
    logits = torch.randn(b, q, c)
    directions = F.normalize(torch.randn(c, d), dim=-1)
    prototypes = directions[:, None, :].expand(c, k, d).contiguous()

    fused, category_ids, _, _ = soft_category_prototype_fusion(
        regions,
        logits,
        prototypes,
        category_topk=1,
        category_temperature=1.0,
        prototype_temperature=0.15,
    )

    expected = directions[category_ids.squeeze(-1)]
    torch.testing.assert_close(fused, expected, atol=1e-6, rtol=1e-6)


def test_prototype_ops_propagate_finite_gradients():
    torch.manual_seed(4)
    features = F.normalize(torch.randn(2, 3, 8), dim=-1).requires_grad_()
    prototypes = F.normalize(torch.randn(5, 4, 8), dim=-1).requires_grad_()
    logits = calibrated_logmeanexp_similarity(
        features, prototypes, temperature=0.07, logit_scale=50.0
    )
    logits.square().mean().backward()

    assert features.grad is not None and torch.isfinite(features.grad).all()
    assert prototypes.grad is not None and torch.isfinite(prototypes.grad).all()
