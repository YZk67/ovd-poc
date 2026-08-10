import torch
import torch.nn.functional as F

from lami_dino.models import TextPrototypeAggregator


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

    The aggregator once divided the attention logits by both sqrt(hidden_dim)
    and tau, which cancelled the temperature out. Attention over the phrases
    came out near-uniform, every prototype landed on the same mean-of-values
    vector (pairwise cosine > 0.9999), and because the task loss hands an
    identical gradient to identical prototypes, training could never separate
    them again. Diversity has to exist at init; nothing downstream creates it.
    """
    torch.manual_seed(0)
    num_classes, num_phrases, dim, num_prototypes = 64, 12, 128, 5

    # Phrases of one class share a direction but are not identical, which is how
    # real CLIP prompt banks look (intra-class cosine ~0.75).
    base = F.normalize(torch.randn(num_classes, 1, dim), p=2, dim=-1)
    spread = F.normalize(torch.randn(num_classes, num_phrases, dim), p=2, dim=-1)
    text_feats = F.normalize(base + 0.58 * spread, p=2, dim=-1)

    tpa = TextPrototypeAggregator(
        dim=dim, num_prototypes=num_prototypes, hidden_dim=256, dropout=0.0, tau=0.07
    )
    tpa.eval()
    with torch.no_grad():
        prototypes, _ = tpa(text_feats, with_loss=False)

    assert _pairwise_cos(prototypes) < 0.999, (
        "TPA prototypes are collapsed at initialization: the K prototypes are "
        "effectively one vector, so num_prototypes>1 buys nothing."
    )


def test_tau_controls_attention_sharpness():
    """tau must be the only temperature knob: lower tau => sharper attention.

    If some other constant divides the logits, tau stops having a meaningful
    effect, which is exactly how the collapse above went unnoticed.
    """
    torch.manual_seed(0)
    text_feats = torch.randn(32, 12, 128)

    def entropy_at(tau):
        tpa = TextPrototypeAggregator(dim=128, num_prototypes=5, hidden_dim=256, dropout=0.0, tau=tau)
        torch.manual_seed(0)
        tpa._reset_parameters()
        tpa.eval()
        with torch.no_grad():
            tpa(text_feats, with_loss=False)
            attn = F.softmax(tpa._last_logits / tau, dim=-1)
        return -(attn * attn.clamp_min(1e-9).log()).sum(-1).mean()

    assert entropy_at(0.01) < entropy_at(1.0)
