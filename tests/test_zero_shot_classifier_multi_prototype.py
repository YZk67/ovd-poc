import numpy as np
import torch

from detrex.modeling.classifier.zero_shot_classifier import ZeroShotClassifier


def test_static_multi_prototype_classifier_shapes(tmp_path):
    num_classes, num_prototypes, dim = 4, 3, 8
    train_path = tmp_path / "train.npy"
    eval_path = tmp_path / "eval.npy"
    embeddings = np.random.randn(num_classes, num_prototypes, dim).astype("float32")
    np.save(train_path, embeddings)
    np.save(eval_path, embeddings)

    classifier = ZeroShotClassifier(
        input_shape=dim,
        num_classes=num_classes,
        zs_weight_path=str(train_path),
        eval_zs_weight_path=str(eval_path),
        zs_weight_dim=dim,
        norm_temperature=1.0,
    )
    x = torch.randn(2, 5, dim)

    logits = classifier(x)
    subset_logits = classifier(x, content_inds=torch.tensor([0, 2]))

    assert logits.shape == (2, 5, num_classes)
    assert subset_logits.shape == (2, 5, 2)


def test_static_multi_prototype_score_aggregation(tmp_path):
    train_path = tmp_path / "train.npy"
    eval_path = tmp_path / "eval.npy"
    embeddings = np.array(
        [
            [[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]],
            [[-1.0, 0.0, 0.0], [0.0, -2.0, 0.0], [0.0, 0.0, -3.0]],
        ],
        dtype="float32",
    )
    np.save(train_path, embeddings)
    np.save(eval_path, embeddings)

    x = torch.ones(1, 1, 3)
    expected = {
        "mean": torch.tensor([[[2.0, -2.0]]]),
        "max": torch.tensor([[[3.0, -1.0]]]),
        "logsumexp": torch.logsumexp(
            torch.tensor([[[[1.0, 2.0, 3.0], [-1.0, -2.0, -3.0]]]]),
            dim=-1,
        ),
    }

    for agg, target in expected.items():
        classifier = ZeroShotClassifier(
            input_shape=3,
            num_classes=2,
            zs_weight_path=str(train_path),
            eval_zs_weight_path=str(eval_path),
            zs_weight_dim=3,
            norm_weight=False,
            multi_prototype_score_agg=agg,
        )
        with torch.no_grad():
            classifier.linear.weight.copy_(torch.eye(3))
            classifier.linear.bias.zero_()

        assert torch.allclose(classifier(x), target)
