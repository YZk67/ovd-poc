import torch
import numpy as np
import pytest

from lami_dino.models import TextPrototypeAggregator


def test_tpa_output_shape():
    num_classes, num_phrases, dim, num_prototypes = 5, 7, 16, 3
    text_feats = torch.randn(num_classes, num_phrases, dim)

    tpa = TextPrototypeAggregator(dim=dim, num_prototypes=num_prototypes, hidden_dim=32, dropout=0.0)
    prototypes, apr_loss = tpa(text_feats)

    assert prototypes.shape == (num_classes, num_prototypes, dim)
    assert apr_loss.ndim == 0


def test_apr_diversity_penalizes_collapsed_usage():
    tpa = TextPrototypeAggregator(dim=8, num_prototypes=3, hidden_dim=16, dropout=0.0)
    uniform_logits = torch.zeros(2, 3, 5)
    collapsed_logits = torch.zeros(2, 3, 5)
    collapsed_logits[:, 0, :] = 10.0

    uniform_loss = tpa._diversity_term(uniform_logits)
    collapsed_loss = tpa._diversity_term(collapsed_logits)

    assert uniform_loss < collapsed_loss


def test_static_multi_prototype_classifier(tmp_path):
    pytest.importorskip("detectron2.layers")
    from lami_dino.modeling.text_classifier import TextClassifier

    num_classes, num_prototypes, dim = 4, 3, 8
    train_path = tmp_path / "train.npy"
    eval_path = tmp_path / "eval.npy"
    embeddings = np.random.randn(num_classes, num_prototypes, dim).astype("float32")
    np.save(train_path, embeddings)
    np.save(eval_path, embeddings)

    classifier = TextClassifier(
        input_shape=dim,
        num_classes=num_classes,
        zs_weight_path=str(train_path),
        eval_zs_weight_path=str(eval_path),
        zs_weight_dim=dim,
        use_tpa=False,
        norm_temperature=1.0,
    )
    x = torch.randn(2, 5, dim)

    logits = classifier(x)
    subset_logits = classifier(x, content_inds=torch.tensor([0, 2]))

    assert logits.shape == (2, 5, num_classes)
    assert subset_logits.shape == (2, 5, 2)
