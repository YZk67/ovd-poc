import ast
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest
import torch

from lami_dino.prototype_ops import route_conflicting_task_gradient
from tools.accumulation_objective_ops import compare_gradients, normalization_plan
from tools.accumulation_update_ops import (
    ARMS, add_optional, assign_gradients, capture_gradients, shadow_adamw_step,
    snapshot_gradients, verify_forward,
)
from tools.audit_accumulation_updates import validate_reference
from tools.decoder_aux_ablation_ops import state_digest


def toy_losses(x, bank, denominator):
    return {"loss_class": (x.sum() + bank.sum()).square()/denominator,
            "loss_bbox": x.square().sum()/denominator,
            "loss_giou": 3*x.sum()/denominator,
            "loss_class_dn": 7*bank.sum()/denominator,
            "loss_apr": 2*bank.square().sum(), "loss_rpsa": .7*x.sum()*bank.sum()}


def test_one_graph_two_gt_weights_preserve_apr_and_amp_scaling():
    x = torch.nn.Parameter(torch.tensor([.2, .5]))
    bank = torch.nn.Parameter(torch.tensor([.3, -.7]))
    unused = torch.nn.Parameter(torch.tensor([1.]))
    params = [x, bank, unused]
    before = [p.detach().clone() for p in params]
    grads, present, apr, _ = capture_gradients(toy_losses(x, bank, 5), params, [bank], 1.4, 1024)
    assert present == [True, True, False]
    torch.testing.assert_close(apr[0], 2*bank.detach())
    for arm, multiplier in zip(ARMS, (1., 1.4)):
        losses = toy_losses(x, bank, 5)
        det = sum(v for k, v in losses.items() if k not in ("loss_apr", "loss_rpsa"))
        target = (multiplier*det + losses["loss_apr"] + losses["loss_rpsa"])/2
        expected = torch.autograd.grad(target, (x, bank))
        for a, b in zip(grads[arm][:2], expected):
            torch.testing.assert_close(a/1024, b)
    for p, old in zip(params, before):
        assert p.grad is None
        torch.testing.assert_close(p, old)


def test_regularizers_are_unchanged_and_none_is_not_zero_gradient():
    bank = torch.nn.Parameter(torch.tensor([1., 2.]))
    x = torch.nn.Parameter(torch.tensor([.4]))
    losses = {"loss_class": 0*x.sum(), "loss_bbox": 0*x.sum(), "loss_giou": 0*x.sum(),
              "loss_apr": bank.square().sum(), "loss_rpsa": bank.sum()}
    gradients, flags, _, _ = capture_gradients(losses, [x, bank], [bank], 3., 16)
    assert flags == [True, True]  # a connected zero still gets AdamW momentum/decay
    for a, b in zip(gradients[ARMS[0]], gradients[ARMS[1]]):
        torch.testing.assert_close(a, b)


@pytest.mark.parametrize("multiplier,scale", [(0.,1.), (float('nan'),1.), (1.,0.), (1.,float('inf'))])
def test_invalid_scalars_fail_closed(multiplier, scale):
    p = torch.nn.Parameter(torch.tensor([1.]))
    with pytest.raises(ValueError, match="Invalid"):
        capture_gradients(toy_losses(p, p, 2), [p], [p], multiplier, scale)


def test_optional_accumulation_and_global_presence_assignment():
    a = add_optional(None, [torch.tensor([1.]), None])
    a = add_optional(a, [None, torch.tensor([2.])])
    a = add_optional(a, [torch.tensor([3.]), None])
    p = [torch.nn.Parameter(torch.ones(1)) for _ in range(2)]
    assign_gradients(p, a, [True, True])
    assert [x.grad.item() for x in p] == [4., 2.]
    a[0].add_(100)
    assert p[0].grad.item() == 4  # no aliasing CPU capture buffer
    assign_gradients(p, [torch.zeros(1), torch.zeros(1)], [True, False])
    assert p[0].grad is not None and p[1].grad is None
    with pytest.raises(ValueError, match="Unused"):
        assign_gradients(p, [torch.ones(1), torch.zeros(1)], [False, False])
    p[0].grad.fill_(float('inf'))
    with pytest.raises(FloatingPointError):
        snapshot_gradients(p)


def native_trainer_methods():
    # Exercise the ACTUAL source methods without requiring local CUDA D2/fvcore.
    path = Path(__file__).resolve().parents[1]/"tools/train_net.py"
    tree = ast.parse(path.read_text())
    trainer = next(x for x in tree.body if isinstance(x, ast.ClassDef) and x.name == "Trainer")
    names = {"_get_tpa", "_route_tpa_gradients", "clip_grads", "clip_model_grads"}
    cls = ast.parse("class NativeMethods:\n    pass\n").body[0]
    cls.body = [x for x in trainer.body if isinstance(x, ast.FunctionDef) and x.name in names]
    assert len(cls.body) == 4
    namespace = {"torch": torch, "dist": torch.distributed,
                 "route_conflicting_task_gradient": route_conflicting_task_gradient}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[])), str(path), "exec"), namespace)
    return namespace["NativeMethods"]


def toy_trainer():
    model = torch.nn.Module()
    model.visual = torch.nn.Parameter(torch.tensor([.5, -.3]))
    model.transformer = torch.nn.Module()
    model.transformer.decoder = torch.nn.Module()
    head = torch.nn.Module()
    head.tpa = torch.nn.Linear(2, 1, bias=False)
    model.transformer.decoder.class_embed = torch.nn.ModuleList([head])
    trainer = native_trainer_methods()()
    trainer.model = model
    trainer.separate_tpa_grad_clip = True
    trainer.clip_grad_params = {"max_norm": .5, "norm_type": 2}
    return model, trainer


def test_native_projection_and_independent_clipping_no_weight_mutation():
    model, trainer = toy_trainer()
    before = state_digest(model.state_dict())
    bank = trainer._get_tpa().weight
    model.visual.grad = torch.tensor([3., 4.])
    bank.grad = torch.tensor([[-1., 2.]])  # task = [-2, 2], conflicts with APR [1, 0]
    trainer._route_tpa_gradients((torch.tensor([[1., 0.]]),))
    torch.testing.assert_close(bank.grad, torch.tensor([[1., 2.]]))
    assert trainer._last_tpa_projection_metrics['conflict_projected'] == 1
    norms = trainer.clip_model_grads()
    assert float(norms[0]) == pytest.approx(5.)
    assert float(norms[1]) == pytest.approx(5**.5)
    assert model.visual.grad.norm().item() == pytest.approx(.5, abs=1e-6)
    assert bank.grad.norm().item() == pytest.approx(.5, abs=1e-6)
    assert state_digest(model.state_dict()) == before


def initialized_optimizer(dtype=torch.float32, *, amsgrad=False):
    params = [torch.nn.Parameter(torch.tensor([.7, -1.2], dtype=dtype)) for _ in range(3)]
    optimizer = torch.optim.AdamW([
        {"params": params[:2], "lr": 1e-4, "weight_decay": 1e-4},
        {"params": params[2:], "lr": 1e-3, "weight_decay": 0.},
    ], betas=(.9,.999), amsgrad=amsgrad)
    for i, p in enumerate(params):
        p.grad = torch.tensor([i+.4, -.2], dtype=dtype)
    optimizer.step()
    params[0].grad = torch.tensor([-.2, .5], dtype=dtype)
    params[1].grad = None
    params[2].grad = torch.zeros_like(params[2])
    return params, optimizer


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("amsgrad", [False, True])
def test_shadow_adamw_matches_real_next_step_and_preserves_none(dtype, amsgrad):
    params, optimizer = initialized_optimizer(dtype, amsgrad=amsgrad)
    before = [p.detach().clone() for p in params]
    grads = [None if p.grad is None else p.grad.clone() for p in params]
    state = state_digest(optimizer.state_dict())
    predicted, receipt = shadow_adamw_step(optimizer, params, state)
    assert state_digest(optimizer.state_dict()) == state
    assert receipt['updated_parameter_tensors'] == 2
    assert receipt['unused_parameter_tensors'] == 1
    for p, old, grad in zip(params, before, grads):
        torch.testing.assert_close(p, old, rtol=0, atol=0)
        if grad is None:
            assert p.grad is None
        else:
            torch.testing.assert_close(p.grad, grad, rtol=0, atol=0)
    assert not torch.count_nonzero(predicted[1])
    assert torch.count_nonzero(predicted[2])  # zero-gradient parameter still uses moments
    optimizer.step()
    for p, old, delta in zip(params, before, predicted):
        torch.testing.assert_close(p-old, delta, atol=0, rtol=0)


def test_independent_arms_start_from_identical_moments_and_do_not_call_live_step():
    params, optimizer = initialized_optimizer()
    digest = state_digest(optimizer.state_dict())
    def forbidden(*args, **kwargs):
        raise AssertionError("Live optimizer.step was called")
    optimizer.step = forbidden
    a, _ = shadow_adamw_step(optimizer, params, digest)
    params[0].grad.neg_()
    b, _ = shadow_adamw_step(optimizer, params, digest)
    params[0].grad.neg_()
    again, _ = shadow_adamw_step(optimizer, params, digest)
    assert not torch.equal(a[0], b[0])
    for x, y in zip(a, again):
        torch.testing.assert_close(x, y, rtol=0, atol=0)
    assert state_digest(optimizer.state_dict()) == digest


def test_shadow_rejects_wrong_optimizer_state_missing_moments_and_inventory():
    params, optimizer = initialized_optimizer()
    with pytest.raises(ValueError, match="state changed"):
        shadow_adamw_step(optimizer, params, "wrong")
    with pytest.raises(ValueError, match="inventories differ"):
        shadow_adamw_step(optimizer, params[:-1], state_digest(optimizer.state_dict()))
    del optimizer.state[params[0]]['exp_avg']
    with pytest.raises(ValueError, match="moments"):
        shadow_adamw_step(optimizer, params, state_digest(optimizer.state_dict()))


def test_equal_gt_counts_give_identical_shadow_updates_after_native_routing():
    model, trainer = toy_trainer()
    params = list(model.parameters())
    bank = trainer._get_tpa().weight
    optimizer = torch.optim.AdamW(params, lr=1e-3, weight_decay=1e-4)
    for p in params:
        p.grad = torch.ones_like(p)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    digest = state_digest(optimizer.state_dict())
    before = state_digest(model.state_dict())
    plan = normalization_plan([16,16])
    totals = {arm: None for arm in ARMS}
    apr = None
    for denom, coef in zip(plan['criterion_normalizers'],plan['detection_loss_multipliers']):
        gradients, presence, a, _ = capture_gradients(toy_losses(model.visual, bank, denom), params, [bank], coef, 32)
        for arm in ARMS:
            totals[arm] = add_optional(totals[arm], gradients[arm])
        apr = add_optional(apr, a)
    updates = {}
    for arm in ARMS:
        assign_gradients(params, [g/32 for g in totals[arm]], presence)
        trainer._route_tpa_gradients(apr)
        trainer.clip_model_grads()
        updates[arm], _ = shadow_adamw_step(optimizer, params, digest)
    comparison = compare_gradients(updates[ARMS[0]], updates[ARMS[1]], ['detector','tpa'])
    assert comparison['all_trainable']['difference_l2'] == 0
    assert state_digest(model.state_dict()) == before
    assert state_digest(optimizer.state_dict()) == digest


def test_reference_and_forward_pairing_fail_closed():
    inputs = dict(schema_version=2, world_size=4, physical_batch=16, accumulation=2,
                  effective_batch=32, reference_world_size=8, windows=2)
    ref = dict(complete=True, decision="OBJECTIVE_DIFFERENCES_ONLY_NOT_PERFORMANCE_ATTRIBUTION",
               model_state_unchanged=True, source_checkpoint_unchanged=True, optimizer_steps=0,
               inputs=inputs, windows=[{},{}], protocol={'iteration':56800,'classifier':{'tpa_prototype_mode_strength':0}},
               fingerprint=hashlib.sha256(json.dumps(inputs,sort_keys=True).encode()).hexdigest())
    validate_reference(ref)
    with pytest.raises(ValueError, match="fingerprint"):
        validate_reference({**ref, 'fingerprint':'wrong'})
    row = dict(micro=0, inputs=[], seed=42, rng_after=['x','y'], native_indices=[1,2], selected_indices=[1,2],
               normalizers=[2], matches=[[[[1],[0]]]], dn={'dn_num':2}, weighted_losses={'loss_class':.123})
    verify_forward(row, deepcopy(row))
    for key in ('selected_indices','rng_after','matches','inputs'):
        with pytest.raises(ValueError, match=key):
            verify_forward({**row, key:None}, row)
    with pytest.raises(ValueError, match="loss values"):
        verify_forward({**row,'weighted_losses':{'loss_class':.2}}, row)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Native GradScaler/device replay requires CUDA")
def test_cuda_finish_arm_and_shadow_step_use_native_unscale():
    from tools.audit_accumulation_updates import finish_arm
    model, trainer = toy_trainer()
    model.cuda()
    params = list(model.parameters())
    optimizer = torch.optim.AdamW(params, lr=1e-3)
    for p in params:
        p.grad = torch.ones_like(p)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    saved = torch.cuda.amp.GradScaler(init_scale=128).state_dict()
    digest = state_digest(optimizer.state_dict())
    scaled = [torch.full_like(p,256,device='cpu') for p in params]
    apr = [torch.zeros_like(trainer._get_tpa().weight,device='cpu')]
    stages, receipt = finish_arm(trainer, optimizer, params, scaled, [True]*len(params), apr, saved)
    for g in stages['raw']:
        torch.testing.assert_close(g, torch.full_like(g,2))
    assert receipt['amp_scale'] == 128
    shadow_adamw_step(optimizer, params, digest)
