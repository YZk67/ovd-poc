"""No-radius control without importing the detector/CUDA stack."""

import ast
import copy
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch

from lami_dino.models import TextPrototypeAggregator
from lami_dino.prototype_ops import calibrated_logmeanexp_similarity


CONFIGS = Path(__file__).resolve().parents[1] / "lami_dino" / "configs"
CONFIG = CONFIGS / "dino_convnext_large_4scale_4ep_lvis_no_radius.py"


def make_tpa(strength):
    return TextPrototypeAggregator(
        dim=16, hidden_dim=32, num_prototypes=5, dropout=0.1,
        slot_prior_strength=0.2, prototype_mode_strength=strength,
        identity_value_init=True, warmup_steps=0,
    )


def raw_slots(tpa, prompts):
    keys, values = tpa.key_proj(prompts), tpa.value_proj(prompts)
    logits = tpa._add_slot_prior(torch.einsum("kh,cnh->ckn", tpa.prototype_queries, keys))
    return torch.einsum("ckn,cnd->ckd", (logits / tpa.attention_scale).softmax(-1), values), logits


def test_radius_changes_no_learned_initial_tensor_or_rng_consumption():
    torch.manual_seed(42)
    native = make_tpa(1.5)
    native_rng = torch.get_rng_state()
    torch.manual_seed(42)
    control = make_tpa(0.0)
    assert torch.equal(native_rng, torch.get_rng_state())
    differing = {name for name, value in native.state_dict().items()
                 if not torch.equal(value, control.state_dict()[name])}
    assert differing == {"prototype_mode_strength"}
    assert all(torch.equal(p, dict(control.named_parameters())[name])
               for name, p in native.named_parameters())
    assert control._effective_lambdas() == native._effective_lambdas() == (0.10, 0.03)
    assert control.dropout.p == native.dropout.p == 0.1


@pytest.mark.parametrize("training", [False, True])
def test_zero_radius_retains_raw_five_slots_not_their_center(training):
    torch.manual_seed(3)
    tpa = make_tpa(0.0).train(training)
    prompts = torch.randn(7, 8, 16)
    expected, logits = raw_slots(tpa, prompts)
    actual, apr = tpa(prompts, with_loss=True, advance_step=False, apply_dropout=False)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert actual.shape == (7, 5, 16)
    assert not torch.allclose(actual, actual.mean(1, keepdim=True).expand_as(actual))
    torch.testing.assert_close(apr, tpa.compute_apr_loss(expected, logits))
    assert apr.requires_grad and apr.item() > 0


def test_no_radius_preserves_dropout_and_apr_uses_clean_prototypes():
    torch.manual_seed(4)
    tpa = make_tpa(0.0).train()
    prompts = torch.randn(7, 8, 16)
    expected, logits = raw_slots(tpa, prompts)
    torch.manual_seed(10)
    dropped = tpa.dropout(expected)
    torch.manual_seed(10)
    actual, apr = tpa(prompts, advance_step=False)
    torch.testing.assert_close(actual, dropped, rtol=0, atol=0)
    torch.testing.assert_close(apr, tpa.compute_apr_loss(expected, logits))


def test_no_radius_task_and_apr_gradients_match_raw_attention_reference():
    torch.manual_seed(5)
    tpa = make_tpa(0.0).train()
    prompts = torch.randn(7, 8, 16)
    queries = torch.nn.functional.normalize(torch.randn(2, 3, 16), dim=-1)

    def task_loss(prototypes):
        prototypes = torch.nn.functional.normalize(prototypes, dim=-1)
        logits = calibrated_logmeanexp_similarity(
            queries, prototypes, temperature=0.07, logit_scale=50.0)
        return logits.square().mean()

    actual, apr = tpa(prompts, advance_step=False, apply_dropout=False)
    actual_grads = torch.autograd.grad(task_loss(actual) + apr, tuple(tpa.parameters()))
    expected, logits = raw_slots(tpa, prompts)
    expected_grads = torch.autograd.grad(
        task_loss(expected) + tpa.compute_apr_loss(expected, logits), tuple(tpa.parameters()))
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        assert torch.isfinite(actual_grad).all()
        torch.testing.assert_close(actual_grad, expected_grad)
    assert all(p.grad is None for p in tpa.parameters())


def test_no_radius_checkpoint_restores_radius_and_live_monitor(tmp_path):
    torch.manual_seed(6)
    control = make_tpa(0.0).eval()
    path = tmp_path / "control.pth"
    torch.save(control.state_dict(), path)
    restored = make_tpa(1.5).eval()
    restored.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    prompts = torch.randn(7, 8, 16)
    expected, _ = control(prompts, with_loss=False)
    actual, _ = restored(prompts, with_loss=False)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert restored.prototype_mode_strength.item() == 0.0
    assert restored.get_monitor_dict()["fixed_radius_enabled"] == 0.0
    assert restored.get_monitor_dict()["prototype_mode_strength"] == 0.0
    restored.prototype_mode_strength.fill_(1.5)
    assert restored.get_monitor_dict()["fixed_radius_enabled"] == 1.0
    assert restored.get_monitor_dict()["prototype_mode_strength"] == 1.5


def test_ablation_config_changes_only_radius_and_output_paths(monkeypatch):
    # Load the real child config against a sentinel base. Anything else changed
    # in the inherited model/train/data/optimizer settings fails equality below.
    module_name = "_radius_test_package.dino_convnext_large_4scale_4ep_lvis_screen"
    base = ModuleType(module_name)
    base.model = SimpleNamespace(classifier=SimpleNamespace(tpa_prototype_mode_strength=1.5), untouched=object())
    base.train = SimpleNamespace(output_dir="baseline", max_iter=28400, lr_scheduler_max_iter=85200)
    base.dataloader = SimpleNamespace(evaluator=SimpleNamespace(output_dir="baseline"))
    base.optimizer = SimpleNamespace(lr=1e-4)
    base.lr_multiplier = object()
    base.iterations_per_epoch = 7100
    monkeypatch.setitem(sys.modules, module_name, base)
    namespace = {"__package__": "_radius_test_package", "__name__": "_radius_test_package.no_radius"}
    before_model_untouched = base.model.untouched
    before_train = copy.deepcopy(vars(base.train))
    exec(compile(CONFIG.read_text(), str(CONFIG), "exec"), namespace)
    assert namespace["model"] is base.model
    assert base.model.untouched is before_model_untouched
    assert base.model.classifier.tpa_prototype_mode_strength == 0.0
    assert {k: v for k, v in vars(base.train).items() if k != "output_dir"} == {
        k: v for k, v in before_train.items() if k != "output_dir"}
    assert base.dataloader.evaluator.output_dir == base.train.output_dir != "baseline"
    assert namespace["optimizer"] is base.optimizer and base.optimizer.lr == 1e-4
    assert namespace["lr_multiplier"] is base.lr_multiplier
    statements = ast.parse(CONFIG.read_text()).body
    # Restrict *all* child assignments; no hidden schedule/loss/initialization edits.
    assert [ast.unparse(s.targets[0]) for s in statements if isinstance(s, ast.Assign)] == [
        "model.classifier.tpa_prototype_mode_strength", "train.output_dir",
        "dataloader.evaluator.output_dir"]
    assert all(isinstance(s, (ast.Expr, ast.ImportFrom, ast.Assign)) for s in statements)


def test_inherited_protocol_is_locked_to_the_existing_bs32_reference():
    # Check the selected literal settings in actual source without importing the
    # detector. Child-only equality above guards the rest of the inherited cfg.
    files = ["dino_convnext_large_4scale_12ep_lvis.py", "dino_convnext_large_4scale_4ep_lvis_screen.py"]
    assignments = {}
    for name in files:
        for node in ast.parse((CONFIGS / name).read_text()).body:
            if isinstance(node, ast.Assign) and len(node.targets) == 1:
                assignments[ast.unparse(node.targets[0])] = node.value
    expected = {
        "train.max_iter": 28400, "train.lr_scheduler_max_iter": 85200,
        "train.eval_period": 28400, "train.checkpointer.period": 14200,
        "train.gradient_accumulation_steps": 2, "dataloader.train.total_batch_size": 16,
        "train.seed": 42, "train.init_checkpoint_scope": "backbone_only",
        "train.init_checkpoint": "./pretrained_models/clip_convnext_large_trans.pth",
        "train.backbone_trainable_scope": "output_norm_only",
        "optimizer.lr": 1e-4, "train.tpa_lr_multiplier": 10.0,
        "train.tpa_conflict_projection": True, "train.separate_tpa_grad_clip": True,
        "train.clip_grad.params.max_norm": 0.5,
        "model.classifier.tpa_num_prototypes": 5,
        "model.classifier.tpa_prototype_mode_strength": 1.5,
        "model.classifier.tpa_slot_prior_strength": 0.2,
        "model.classifier.tpa_identity_value_init": True,
        "model.classifier.tpa_tau": 0.004375, "model.classifier.tpa_cls_tau": 0.07,
        "model.classifier.tpa_dropout": 0.1, "model.classifier.tpa_warmup_steps": 0,
        "model.tpa_stabilization_steps": 0, "model.tpa_task_gradient_scale": 1.0,
        "model.transformer.rpsa_warmup_start": 20000,
        "model.transformer.rpsa_warmup_iters": 8000,
        "model.criterion.weight_dict['loss_rpsa']": 0.05,
        "model.alpha": 0.0, "model.beta": 0.3, "model.novel_scale": 3.0,
        "model.soft_category_topk": 3,
    }
    scope = {"iterations_per_epoch": 7100, "base_lr": 1e-4, "tpa_lr_multiplier": 10.0}
    for key, value in expected.items():
        expr = ast.Expression(assignments[key])
        assert eval(compile(expr, "<config-setting>", "eval"), {"__builtins__": {}}, scope) == value, key
