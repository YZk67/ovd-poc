import torch

from lami_dino.models import TextPrototypeAggregator


def test_tpa_output_shape():
    num_classes, num_phrases, dim, num_prototypes = 5, 7, 16, 3
    text_feats = torch.randn(num_classes, num_phrases, dim)

    tpa = TextPrototypeAggregator(dim=dim, num_prototypes=num_prototypes, hidden_dim=32, dropout=0.0)
    prototypes, apr_loss = tpa(text_feats)

    assert prototypes.shape == (num_classes, num_prototypes, dim)
    assert apr_loss is not None
