import torch

from lami_dino.models import TextPrototypeAggregator


def test_tpa_output_shape():
    num_classes, num_phrases, dim, num_prototypes = 5, 7, 16, 3
    text_feats = torch.randn(num_classes, num_phrases, dim)

    tpa = TextPrototypeAggregator(dim=dim, num_prototypes=num_prototypes, hidden_dim=32, dropout=0.0)
    prototypes, apr_loss = tpa(text_feats)

    assert prototypes.shape == (num_classes, num_prototypes, dim)
    assert apr_loss is not None


def test_tpa_step_persists_across_state_dict():
    text_feats = torch.randn(4, 6, 8)

    tpa = TextPrototypeAggregator(
        dim=8,
        num_prototypes=3,
        hidden_dim=16,
        dropout=0.0,
        warmup_steps=100,
    )
    for _ in range(7):
        tpa(text_feats)

    step_before = tpa.step
    lambdas_before = tpa._effective_lambdas()

    restored = TextPrototypeAggregator(
        dim=8,
        num_prototypes=3,
        hidden_dim=16,
        dropout=0.0,
        warmup_steps=100,
    )
    restored.load_state_dict(tpa.state_dict(), strict=False)

    assert restored.step == step_before
    assert restored._effective_lambdas() == lambdas_before
