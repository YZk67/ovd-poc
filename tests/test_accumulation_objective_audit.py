from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from tools.accumulation_objective_ops import (
    category_overlap, compare_gradients, dn_layout, normalization_plan,
    objective_gradients, paired_fedloss, split_objective,
)
from tools.audit_accumulation_objective import forward_observations, validate_pairing


def losses(x, denominator):
    return {"loss_class": 2*x.square()/denominator,
            "loss_bbox": 3*x.square()/denominator,
            "loss_giou": 5*x.square()/denominator,
            "loss_class_dn": 7*x.square()/(denominator*3),
            "loss_class_dn_0": 11*x.square()/(denominator*3),
            "loss_bbox_enc": 13*x.square()/denominator,
            "loss_apr": 17*x.square(), "loss_rpsa": 19*x.square()}


def test_pooled_gradient_matches_combined_gt_normalization_not_micro_mean():
    # DDP mean over 4 ranks and two forwards versus a declared 8-rank reference.
    plan = normalization_plan([8, 24])
    assert plan["detection_loss_multipliers"] == [.5, 1.5]
    x = torch.nn.Parameter(torch.tensor(.7, dtype=torch.float64))
    micro, pooled = 0, 0
    for n, c in zip(plan["criterion_normalizers"], plan["detection_loss_multipliers"]):
        grads, _ = objective_gradients(losses(x, n), [x], c)
        micro += grads["micro"][0]
        pooled += grads["pooled_gt"][0]
    # Identical local numerator for each virtual rank; pooled mean-GT = 32/8=4.
    expected = torch.autograd.grad(sum(losses(x, 4).values()), x)[0]
    torch.testing.assert_close(pooled.double(), expected, rtol=2e-6, atol=1e-6)
    assert not torch.isclose(micro, pooled)
    assert x.grad is None


@pytest.mark.parametrize("counts,world,reference", [([0, 0],4,8), ([0, 0],4,4), ([0, 20],4,8), ([1, 0],4,8), ([16,16],4,8)])
def test_clamping_and_ddp_world_size_are_included(counts, world, reference):
    plan = normalization_plan(counts, world, reference)
    for n, c in zip(counts, plan["detection_loss_multipliers"]):
        native = 1/(len(counts)*max(n, world))
        assert native*c == pytest.approx(1/max(sum(counts), reference))


def test_equal_counts_equal_gradients_and_no_state_mutation():
    x = torch.nn.Parameter(torch.tensor([.3, .6]))
    unused = torch.nn.Parameter(torch.tensor(2.))
    before = x.detach().clone()
    d = losses(x.sum(), 10.)
    grads, values = objective_gradients(d, [x, unused], 1.)
    for a, b in zip(grads["micro"], grads["pooled_gt"]):
        torch.testing.assert_close(a, b)
    assert grads["micro"][1] == 0
    assert values["micro"] == values["pooled_gt"]
    assert x.grad is None and unused.grad is None
    torch.testing.assert_close(before, x)


def test_regularizers_are_not_gt_reweighted():
    x = torch.nn.Parameter(torch.tensor(.4))
    d = {"loss_class": 0*x, "loss_bbox": 0*x, "loss_giou": 0*x,
         "loss_apr": 9*x, "loss_rpsa": 7*x}
    gradients, _ = objective_gradients(d, [x], 3.)
    assert gradients["micro"][0] == gradients["pooled_gt"][0] == 8


def test_split_fails_closed_on_unknown_or_invalid_loss():
    d = losses(torch.tensor(1.), 1.)
    det, other = split_objective(d)
    assert "loss_class_dn_0" in det and "loss_bbox_enc" in det
    assert set(other) == {"loss_apr", "loss_rpsa"}
    with pytest.raises(ValueError, match="Unknown loss"):
        split_objective({**d, "loss_unknown": torch.tensor(0.)})
    with pytest.raises(ValueError, match="Nonfinite"):
        split_objective({**d, "loss_class": torch.tensor(float("nan"))})


class FakeSampler:
    num_classes = 8

    def filter_content_info(self, data):
        torch.rand(7)  # native sampler's RNG draw must be consumed in both arms
        native = torch.tensor([2, 5, 0, 6])
        lookup = torch.full((8,), -1, dtype=torch.long)
        lookup[native] = torch.arange(4)
        for item in data:
            item["instances"].gt_classes = lookup[item["instances"].gt_classes]
        return native, data


def sample_data():
    return [{"instances": SimpleNamespace(gt_classes=torch.tensor([2, 5]))}]


def test_fedloss_remaps_global_labels_and_preserves_downstream_rng():
    model = FakeSampler()
    source = sample_data()
    original = model.filter_content_info
    torch.manual_seed(42)
    with paired_fedloss(model) as native:
        _, data = model.filter_content_info(deepcopy(source))
        native_labels = data[0]["instances"].gt_classes.tolist()
        rng_native = torch.get_rng_state().clone()
    torch.manual_seed(42)
    with paired_fedloss(model, torch.tensor([5, 2, 1, 7])) as shared:
        _, data = model.filter_content_info(deepcopy(source))
        assert data[0]["instances"].gt_classes.tolist() == [1, 0]
        assert torch.equal(rng_native, torch.get_rng_state())
    assert native_labels == [0, 1]
    assert native["native_indices"] == shared["native_indices"]
    assert model.filter_content_info == original
    assert source[0]["instances"].gt_classes.tolist() == [2, 5]


@pytest.mark.parametrize("indices", [[2,1,0,6], [2,5,6], [2,5,5,6], [2,5,0,8]])
def test_fedloss_fails_closed_and_restores_method(indices):
    model = FakeSampler()
    original = model.filter_content_info
    with pytest.raises(ValueError):
        with paired_fedloss(model, torch.tensor(indices)):
            model.filter_content_info(sample_data())
    assert model.filter_content_info == original


def paired_rows():
    return [[{"micro": m, "inputs": [rank, m], "seed": 42+rank+m,
              "rng_after": "same", "native_indices": [2,5,0,6],
              "selected_indices": [2,5,0,6], "dn": {"dn_num": 2},
              "normalizers": [3.], "matches": [[[[0,1],[0,1]]]]}
             for m in range(2)] for rank in range(4)]


def test_pairing_checks_all_ranks_and_both_microbatches():
    a, b = paired_rows(), paired_rows()
    assert len(validate_pairing(a, b)) == 8
    b[3][1]["rng_after"] = "changed"
    with pytest.raises(ValueError, match="rng_after"):
        validate_pairing(a, b)
    b = paired_rows()
    b[2][1]["selected_indices"] = [5,2,1,7]
    with pytest.raises(ValueError, match="across ranks"):
        validate_pairing(a, b)


def test_matching_changes_are_reported_not_rejected():
    a, b = paired_rows(), paired_rows()
    b[0][0]["matches"] = [[[[1,0],[0,1]]]]
    report = validate_pairing(a, b)
    assert report[0]["changed_image_branch_assignments"] == 1


def test_gradient_comparison_zero_and_opposite_vectors():
    rows = compare_gradients([torch.tensor([1.,0.]),torch.zeros(1)],
                             [torch.tensor([-1.,0.]),torch.zeros(1)], ["decoder_core","upstream_tpa"])
    assert rows["all_trainable"]["cosine"] == -1
    assert rows["all_trainable"]["relative_difference_l2"] == 2
    assert rows["upstream_tpa"]["cosine"] is None
    assert rows["upstream_tpa"]["relative_difference_l2"] is None


def test_native_dn_grouping_formula_and_set_overlap():
    assert dn_layout([0,1,20,5], 100) == {"dn_num":5, "single_padding":40}
    assert dn_layout([0,0,0,0], 100) == {"dn_num":0, "single_padding":0}
    assert dn_layout([101], 100) == {"dn_num":1, "single_padding":202}
    assert category_overlap([1,2,3], [2,3,4])["jaccard"] == .5


class FakeMatcher(torch.nn.Module):
    def forward(self, outputs, targets):
        return [(torch.tensor([0]), torch.tensor([0]))]


class FakeCriterion(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.matcher = FakeMatcher()

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        return num_boxes


def test_forward_observer_records_and_restores_on_invalid_normalizer():
    model = SimpleNamespace(criterion=FakeCriterion(),
                            prepare_for_cdn=lambda: (None,None,None,{"dn_num":2,"single_padding":8}))
    old = model.criterion.get_loss
    with forward_observations(model, 3.) as record:
        model.prepare_for_cdn()
        model.criterion.matcher({}, [])
        assert model.criterion.get_loss("class", {}, [], [], 6.) == 6
    assert record["normalizers"] == [6.]
    assert record["dn"]["dn_num"] == 2
    assert model.criterion.get_loss == old
    with pytest.raises(ValueError, match="normalizers"):
        with forward_observations(model, 3.):
            model.prepare_for_cdn()
            model.criterion.get_loss("class", {}, [], [], 7.)
    assert model.criterion.get_loss == old
    assert not model.criterion.matcher._forward_hooks
