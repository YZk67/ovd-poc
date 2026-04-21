import math
import torch

from lami_dino.models import TextPrototypeAggregator


def test_tpa_output_shape():
    num_classes, num_phrases, dim, num_prototypes = 5, 7, 16, 3
    text_feats = torch.randn(num_classes, num_phrases, dim)

    tpa = TextPrototypeAggregator(dim=dim, num_prototypes=num_prototypes, hidden_dim=32, dropout=0.0)
    prototypes, apr_loss = tpa(text_feats)

    assert prototypes.shape == (num_classes, num_prototypes, dim)
    assert apr_loss is not None


def test_tpa_base_only_apr_respects_content_indices():
    torch.manual_seed(0)
    text_feats = torch.randn(4, 6, 12)
    base_mask = torch.tensor([True, False, True, False])
    novel_mask = ~base_mask
    class_inds = torch.tensor([1, 2, 3])

    tpa = TextPrototypeAggregator(
        dim=12,
        num_prototypes=3,
        hidden_dim=16,
        dropout=0.0,
        apr_on_base_only=True,
    )
    tpa.set_class_splits(base_mask, novel_mask)
    _, apr_loss = tpa(text_feats[class_inds], class_inds=class_inds)

    subset_base = base_mask[class_inds]
    expected_orth = tpa._orthogonality_term(tpa._last_prototypes[subset_base])
    expected_div = tpa._diversity_term(tpa._last_logits[subset_base])

    assert apr_loss is not None
    assert math.isclose(tpa.last_loss_terms["loss_orth"], expected_orth.item(), rel_tol=1e-5, abs_tol=1e-5)
    assert math.isclose(tpa.last_loss_terms["loss_div"], expected_div.item(), rel_tol=1e-5, abs_tol=1e-5)


def test_tpa_novel_anchor_matches_logged_term():
    torch.manual_seed(0)
    text_feats = torch.randn(3, 5, 10)
    base_mask = torch.tensor([True, False, False])
    novel_mask = ~base_mask

    tpa = TextPrototypeAggregator(
        dim=10,
        num_prototypes=2,
        hidden_dim=14,
        dropout=0.0,
        lambda_orth=0.0,
        lambda_div=0.0,
        novel_anchor_weight=1.0,
    )
    tpa.set_class_splits(base_mask, novel_mask)
    _, apr_loss = tpa(text_feats)

    expected_anchor = tpa._novel_anchor_term(tpa._last_prototypes, text_feats)

    assert apr_loss is not None
    assert math.isclose(tpa.last_loss_terms["loss_novel_anchor"], expected_anchor.item(), rel_tol=1e-5, abs_tol=1e-5)
    assert math.isclose(apr_loss.item(), expected_anchor.item(), rel_tol=1e-5, abs_tol=1e-5)
