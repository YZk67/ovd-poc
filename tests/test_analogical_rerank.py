import importlib.util
import sys
import types

import torch
import torch.nn as nn
import torch.nn.functional as F


def _load_dino_class():
    detrex_layers = types.ModuleType("detrex.layers")
    detrex_layers.MLP = type("MLP", (nn.Module,), {})
    detrex_layers.box_cxcywh_to_xyxy = lambda x: x
    detrex_layers.box_xyxy_to_cxcywh = lambda x: x
    sys.modules["detrex.layers"] = detrex_layers

    detrex_utils = types.ModuleType("detrex.utils")
    detrex_utils.inverse_sigmoid = lambda x: x
    detrex_utils.is_dist_avail_and_initialized = lambda: False
    detrex_utils.load_class_freq = lambda *args, **kwargs: None
    detrex_utils.get_fed_loss_inds = lambda *args, **kwargs: None
    detrex_utils.get_cluster_fed_loss_inds = lambda *args, **kwargs: None
    sys.modules["detrex.utils"] = detrex_utils

    d2_modeling = types.ModuleType("detectron2.modeling")
    d2_modeling.detector_postprocess = lambda *args, **kwargs: None
    sys.modules["detectron2.modeling"] = d2_modeling

    d2_structures = types.ModuleType("detectron2.structures")
    d2_structures.Boxes = object
    d2_structures.ImageList = object
    d2_structures.Instances = object
    sys.modules["detectron2.structures"] = d2_structures

    spec = importlib.util.spec_from_file_location(
        "dino_mod", "/home/yi/ovd-poc/lami_dino/modeling/dino.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.DINO


def test_analogical_rerank_preserves_shape_and_base_scores():
    DINO = _load_dino_class()
    model = object.__new__(DINO)
    model.analogical_rerank = True
    model.analogical_topk = 2
    model.analogical_gamma = 0.2
    model.analogical_tau = 0.07
    model.base_idx = torch.tensor([True, True, False, False])
    model.novel_idx = torch.tensor([False, False, True, True])
    model.eval_content_query_embedding = F.normalize(
        torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [0.8, 0.2, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=torch.float32,
        ),
        p=2,
        dim=-1,
    )

    cls_score = torch.tensor(
        [
            [
                [0.40, 0.30, 0.55, 0.45],
                [0.90, 0.80, 0.20, 0.10],
            ]
        ],
        dtype=torch.float32,
    )
    reranked = model._apply_analogical_rerank(cls_score.clone())

    assert reranked.shape == cls_score.shape
    torch.testing.assert_close(reranked[:, :, model.base_idx], cls_score[:, :, model.base_idx])

    assert not torch.equal(reranked[0, 0, model.novel_idx], cls_score[0, 0, model.novel_idx])
    torch.testing.assert_close(reranked[0, 1], cls_score[0, 1])
