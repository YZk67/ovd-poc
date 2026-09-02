import torch
import torch.nn.functional as F

from lami_dino.models import TextPrototypeAggregator
from lami_dino.models.text_prototype_aggregator import (
    compute_prototype_similarity,
    compute_usage_entropy,
)


def _pairwise_cos(prototypes):
    """Mean off-diagonal cosine between the K prototypes of each class."""
    normed = F.normalize(prototypes, p=2, dim=-1)
    gram = torch.einsum("ckd,cmd->ckm", normed, normed)
    k = gram.size(-1)
    off_sum = gram.sum(dim=(-2, -1)) - gram.diagonal(dim1=-2, dim2=-1).sum(-1)
    return (off_sum / (k * k - k)).mean()


def test_tpa_output_shape():
    num_classes, num_phrases, dim, num_prototypes = 5, 7, 16, 3
    text_feats = torch.randn(num_classes, num_phrases, dim)

    tpa = TextPrototypeAggregator(dim=dim, num_prototypes=num_prototypes, hidden_dim=32, dropout=0.0)
    prototypes, apr_loss = tpa(text_feats)

    assert prototypes.shape == (num_classes, num_prototypes, dim)
    assert apr_loss.ndim == 0


def test_prototypes_are_distinct_at_init():
    """Guard against the degenerate init that made TPA equivalent to K=1.

    Eq. (1)'s sqrt(hidden_dim) factor requires tau_p to be calibrated jointly
    with the hidden dimension. A denominator near 1 made attention almost
    uniform and every prototype landed on the same mean-of-values vector.
    """
    torch.manual_seed(0)
    num_classes, num_phrases, dim, num_prototypes = 64, 12, 128, 5

    # Phrases of one class share a direction but are not identical, which is how
    # real CLIP prompt banks look (intra-class cosine ~0.75).
    base = F.normalize(torch.randn(num_classes, 1, dim), p=2, dim=-1)
    spread = F.normalize(torch.randn(num_classes, num_phrases, dim), p=2, dim=-1)
    text_feats = F.normalize(base + 0.58 * spread, p=2, dim=-1)

    tpa = TextPrototypeAggregator(
        dim=dim,
        num_prototypes=num_prototypes,
        hidden_dim=256,
        dropout=0.0,
        tau=0.004375,
    )
    tpa.eval()
    with torch.no_grad():
        prototypes, _ = tpa(text_feats, with_loss=False)

    assert _pairwise_cos(prototypes) < 0.999, (
        "TPA prototypes are collapsed at initialization: the K prototypes are "
        "effectively one vector, so num_prototypes>1 buys nothing."
    )


def test_eq1_scaled_temperature_controls_attention_sharpness():
    """Eq. (1) uses sqrt(d_h) * tau_p and lower tau_p is sharper."""
    torch.manual_seed(0)
    text_feats = torch.randn(32, 12, 128)

    def entropy_at(tau):
        tpa = TextPrototypeAggregator(dim=128, num_prototypes=5, hidden_dim=256, dropout=0.0, tau=tau)
        torch.manual_seed(0)
        tpa._reset_parameters()
        tpa.eval()
        with torch.no_grad():
            tpa(text_feats, with_loss=False)
            attn = F.softmax(tpa._last_logits / tpa.attention_scale, dim=-1)
        return -(attn * attn.clamp_min(1e-9).log()).sum(-1).mean()

    assert entropy_at(0.001) < entropy_at(0.1)


def test_eq1_attention_scale_is_sqrt_hidden_dim_times_tau():
    tpa = TextPrototypeAggregator(dim=16, hidden_dim=64, tau=0.0125)
    assert tpa.attention_scale == 0.1


def _make_tpa(**kwargs):
    kwargs.setdefault("dim", 16)
    kwargs.setdefault("num_prototypes", 3)
    kwargs.setdefault("hidden_dim", 32)
    kwargs.setdefault("dropout", 0.0)
    return TextPrototypeAggregator(**kwargs)


def test_warmup_clock_ignores_eval_forwards():
    """Only training iterations may advance the APR warmup.

    Counting eval forwards fast-forwards the schedule by however many validation
    passes have run, so "ramp in over the first 5% of training" silently becomes
    something else entirely.
    """
    tpa = _make_tpa(warmup_steps=100)
    text_feats = torch.randn(4, 6, 16)

    tpa.train()
    for _ in range(3):
        tpa(text_feats)
    assert int(tpa._step) == 3

    tpa.eval()
    with torch.no_grad():
        for _ in range(10):
            tpa(text_feats, with_loss=False)
    assert int(tpa._step) == 3, "eval forwards must not advance the warmup clock"


def test_warmup_state_survives_checkpoint_roundtrip():
    """_step must be a persistent buffer, or resuming restarts the warmup at 0
    and re-suppresses the regularizer for another full warmup window."""
    tpa = _make_tpa(warmup_steps=100)
    text_feats = torch.randn(4, 6, 16)
    tpa.train()
    for _ in range(25):
        tpa(text_feats)

    assert "_step" in tpa.state_dict(), "_step must be saved in the state dict"

    restored = _make_tpa(warmup_steps=100)
    restored.load_state_dict(tpa.state_dict())
    assert int(restored._step) == int(tpa._step) == 25

    lam_before = tpa._effective_lambdas()
    lam_after = restored._effective_lambdas()
    assert lam_before == lam_after


def test_warmup_ramps_lambdas_from_zero_to_base():
    lambda_orth, lambda_div = 0.10, 0.03
    tpa = _make_tpa(warmup_steps=100, lambda_orth=lambda_orth, lambda_div=lambda_div)
    text_feats = torch.randn(4, 6, 16)
    tpa.train()

    start_orth, start_div = tpa._effective_lambdas()
    assert start_orth < lambda_orth and start_div < lambda_div

    for _ in range(120):
        tpa(text_feats)

    end_orth, end_div = tpa._effective_lambdas()
    assert end_orth == lambda_orth and end_div == lambda_div


def test_warmup_steps_is_configurable():
    """A run with a different max_iter must be able to size its own warmup."""
    short = _make_tpa(warmup_steps=10)
    long = _make_tpa(warmup_steps=10_000)
    text_feats = torch.randn(4, 6, 16)
    for tpa in (short, long):
        tpa.train()
        for _ in range(20):
            tpa(text_feats)

    assert short._effective_lambdas()[0] > long._effective_lambdas()[0]


def test_both_paper_apr_terms_are_on_by_default():
    """Eq. (5) includes directional diversity and usage balance."""
    tpa = _make_tpa(warmup_steps=0)
    assert tpa.lambda_div_base > 0.0
    assert tpa.lambda_orth_base > 0.0


def test_usage_entropy_is_blind_to_collapse():
    """The reason the collapse went unnoticed: identical prototypes are 'used'
    perfectly evenly, so usage entropy reports its healthiest possible value.
    Any collapse alarm has to come from the similarity metrics instead."""
    C, K, N = 8, 4, 6
    collapsed_logits = torch.zeros(C, K, N)  # k-independent => identical prototypes
    tpa = _make_tpa(num_prototypes=K, warmup_steps=0)

    assert compute_usage_entropy(collapsed_logits, tau=tpa.attention_scale) > 0.99
    assert tpa._balance_term(collapsed_logits).abs() < 1e-6

    collapsed = torch.randn(C, 1, 16).expand(C, K, 16).contiguous()
    cos, rank = compute_prototype_similarity(collapsed)
    assert cos > 0.999 and rank < 1.01, "similarity metrics must flag the collapse"


def test_balance_term_is_kl_to_uniform_usage():
    C, K, N = 4, 3, 6
    tpa = _make_tpa(num_prototypes=K, warmup_steps=0)

    uniform_logits = torch.zeros(C, K, N)
    monopolized_logits = torch.full((C, K, N), -10.0)
    monopolized_logits[:, 0] = 10.0

    assert tpa._balance_term(uniform_logits).abs() < 1e-6
    assert tpa._balance_term(monopolized_logits) > 0.9 * torch.log(torch.tensor(float(K)))


def test_prototype_similarity_separates_collapsed_from_distinct():
    C, K, D = 8, 4, 16
    collapsed = torch.randn(C, 1, D).expand(C, K, D).contiguous()
    distinct = torch.eye(K, D).unsqueeze(0).expand(C, K, D).contiguous()

    cos_c, rank_c = compute_prototype_similarity(collapsed)
    cos_d, rank_d = compute_prototype_similarity(distinct)

    assert cos_c > 0.999 and rank_c < 1.01
    assert abs(cos_d) < 1e-5 and rank_d > K - 0.01


def test_monitor_dict_exposes_collapse_metrics():
    tpa = _make_tpa(warmup_steps=0)
    tpa.eval()
    with torch.no_grad():
        tpa(torch.randn(8, 6, 16), with_loss=False)

    monitor = tpa.get_monitor_dict()
    for key in ("proto_pairwise_cos", "proto_effective_rank", "usage_entropy"):
        assert key in monitor, f"{key} missing from the monitor dict"
        assert isinstance(monitor[key], float)


def test_single_prototype_apr_loss_is_finite():
    """K=1 is the no-op control for the collapse fix, so it must not NaN.

    Both APR terms normalise by a quantity that vanishes at K=1 -- the
    orthogonality term divides by (K*K - K) after masking its numerator to zero,
    and the diversity term divides by log(K). Each evaluated to 0/0, and since
    lambda_div=0 does not rescue a NaN (0.0 * nan is nan), the whole apr_loss
    came back nan and would have poisoned the total loss within a few steps.
    """
    torch.manual_seed(0)
    text_feats = torch.randn(9, 6, 32)

    tpa = TextPrototypeAggregator(dim=32, num_prototypes=1, hidden_dim=16, warmup_steps=10)
    tpa.train()
    for _ in range(3):
        prototypes, apr_loss = tpa(text_feats)

    assert prototypes.shape == (9, 1, 32)
    assert torch.isfinite(apr_loss), f"apr_loss must be finite at K=1, got {apr_loss}"
    assert torch.isfinite(torch.tensor(tpa.last_loss_terms["loss_orth"]))
    assert torch.isfinite(torch.tensor(tpa.last_loss_terms["loss_div"]))

    # Still has a grad_fn: callers put it straight into the loss dict.
    (apr_loss + prototypes.sum()).backward()
    for name, param in tpa.named_parameters():
        if param.grad is not None:
            assert torch.isfinite(param.grad).all(), f"non-finite grad on {name}"
