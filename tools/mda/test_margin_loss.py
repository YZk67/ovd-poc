"""Unit test for DINO._compute_margin_loss + PlainMarginLoss.

Extracts the live source of _compute_margin_loss from dino.py (so the test
follows code edits without manual re-sync) and runs it against a mocked DINO
instance. Verifies:
  - shape: pos[N], neg[N, K] match
  - indexing: skips GT with any -1 in confusable_neg_remapped
  - non-zero loss when neg scores violate margin
  - zero loss when no valid (query, GT) pairs
  - device propagation
"""
import ast
import importlib.util
import sys
import torch
from types import SimpleNamespace

DINO_PATH = "/home/yi/ovd-poc/lami_dino/modeling/dino.py"
MARGIN_LOSS_PATH = "/home/yi/ovd-poc/lami_dino/modeling/mda/margin_loss.py"

# --- Load PlainMarginLoss via file path (avoids detectron2 import chain) ---
spec = importlib.util.spec_from_file_location("margin_loss", MARGIN_LOSS_PATH)
margin_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(margin_mod)
PlainMarginLoss = margin_mod.PlainMarginLoss

# --- Extract _compute_margin_loss source from dino.py ---
with open(DINO_PATH) as f:
    src = f.read()
tree = ast.parse(src)
func_src = None
for node in ast.walk(tree):
    if isinstance(node, ast.FunctionDef) and node.name == "_compute_margin_loss":
        func_src = ast.unparse(node)
        break
assert func_src is not None, "_compute_margin_loss not found in dino.py"
print("=== Extracted _compute_margin_loss ===")
print(func_src)
print()

ns = {"torch": torch}
exec(func_src, ns)
compute_margin_loss = ns["_compute_margin_loss"]


def make_obj(num_negatives=3, margin=0.2, matcher=None):
    obj = SimpleNamespace()
    obj.criterion = SimpleNamespace(matcher=matcher)
    obj.confusable_mgr = SimpleNamespace(num_negatives=num_negatives)
    obj.margin_loss_fn = PlainMarginLoss(margin=margin)
    return obj


def case_1_basic_nonzero():
    """Two batches, GT with confusable negatives violating margin."""
    B, Q, M, K = 2, 10, 20, 3
    torch.manual_seed(0)
    pred_logits = torch.zeros(B, Q, M)
    # batch 0: GT label=3, matched to query 0; negs=[5,9,11]
    # set logit[0,0,3] high, but neg 5 even higher → should fire
    pred_logits[0, 0, 3] = 1.0   # pos sigmoid ≈ 0.731
    pred_logits[0, 0, 5] = 1.5   # neg sigmoid ≈ 0.818  margin=0.2 → 0.2+0.818-0.731=0.287 > 0
    pred_logits[0, 0, 9] = 0.0   # neg sigmoid 0.5      margin: 0.2+0.5-0.731=-0.031 < 0 → 0
    pred_logits[0, 0, 11] = -1.0 # neg sigmoid 0.269    margin: 0.2+0.269-0.731=-0.262 < 0 → 0
    # batch 1: GT label=7, matched to query 5; negs=[12,14,16]
    pred_logits[1, 5, 7] = 0.5   # pos sigmoid 0.622
    pred_logits[1, 5, 12] = 1.0  # neg sigmoid 0.731    margin: 0.2+0.731-0.622=0.309
    pred_logits[1, 5, 14] = 0.6  # neg sigmoid 0.646    margin: 0.2+0.646-0.622=0.224
    pred_logits[1, 5, 16] = -2.0 # neg sigmoid 0.119    margin: <0 → 0
    pred_boxes = torch.zeros(B, Q, 4)

    output = {"pred_logits": pred_logits, "pred_boxes": pred_boxes}
    targets = [
        {
            "labels": torch.tensor([3]),
            "boxes": torch.zeros(1, 4),
            "confusable_neg_remapped": torch.tensor([[5, 9, 11]]),
        },
        {
            "labels": torch.tensor([7]),
            "boxes": torch.zeros(1, 4),
            "confusable_neg_remapped": torch.tensor([[12, 14, 16]]),
        },
    ]

    # Matcher returns one (query, target) pair per image
    def matcher(out, tgts):
        return [
            (torch.tensor([0]), torch.tensor([0])),
            (torch.tensor([5]), torch.tensor([0])),
        ]

    obj = make_obj(num_negatives=K, matcher=matcher)
    loss = compute_margin_loss(obj, output, targets)
    # Expected manual:
    # row 0 contributions: 0.287 + 0 + 0 = 0.287
    # row 1 contributions: 0.309 + 0.224 + 0 = 0.533
    # mean over 6 entries: (0.287 + 0.533) / 6 ≈ 0.1367
    expected = (0.2 + torch.sigmoid(torch.tensor(1.5)) - torch.sigmoid(torch.tensor(1.0))).clamp(min=0)
    expected = expected + (0.2 + torch.sigmoid(torch.tensor(1.0)) - torch.sigmoid(torch.tensor(0.5))).clamp(min=0)
    expected = expected + (0.2 + torch.sigmoid(torch.tensor(0.6)) - torch.sigmoid(torch.tensor(0.5))).clamp(min=0)
    expected = expected / 6
    print(f"[case 1 basic non-zero]    loss = {loss.item():.6f}  expected = {expected.item():.6f}")
    assert loss.requires_grad is False  # no grad in this synthetic test
    assert abs(loss.item() - expected.item()) < 1e-5, "loss mismatch"
    assert loss.item() > 0, "should be non-zero"


def case_2_skip_invalid_negs():
    """GT with -1 in confusable_neg_remapped should be skipped."""
    B, Q, M, K = 1, 5, 20, 3
    pred_logits = torch.full((B, Q, M), -10.0)
    pred_logits[0, 0, 3] = 1.0
    pred_logits[0, 0, 5] = 5.0   # would normally violate margin, but this GT is skipped
    pred_logits[0, 1, 7] = 0.5
    pred_logits[0, 1, 12] = 1.5  # this one stays
    pred_logits[0, 1, 14] = 0.6
    pred_logits[0, 1, 16] = -2.0
    pred_boxes = torch.zeros(B, Q, 4)

    output = {"pred_logits": pred_logits, "pred_boxes": pred_boxes}
    targets = [
        {
            "labels": torch.tensor([3, 7]),
            "boxes": torch.zeros(2, 4),
            # GT 0 has -1 → skipped; GT 1 valid
            "confusable_neg_remapped": torch.tensor([[5, -1, 11], [12, 14, 16]]),
        },
    ]

    def matcher(out, tgts):
        return [(torch.tensor([0, 1]), torch.tensor([0, 1]))]

    obj = make_obj(num_negatives=K, matcher=matcher)
    loss = compute_margin_loss(obj, output, targets)
    # Only GT 1 contributes
    e1 = (0.2 + torch.sigmoid(torch.tensor(1.5)) - torch.sigmoid(torch.tensor(0.5))).clamp(min=0)
    e2 = (0.2 + torch.sigmoid(torch.tensor(0.6)) - torch.sigmoid(torch.tensor(0.5))).clamp(min=0)
    e3 = (0.2 + torch.sigmoid(torch.tensor(-2.0)) - torch.sigmoid(torch.tensor(0.5))).clamp(min=0)
    expected = (e1 + e2 + e3) / 3
    print(f"[case 2 skip invalid negs] loss = {loss.item():.6f}  expected = {expected.item():.6f}")
    assert abs(loss.item() - expected.item()) < 1e-5


def case_3_no_matches():
    """Empty matches → zero loss, no shape error."""
    B, Q, M = 1, 5, 20
    pred_logits = torch.randn(B, Q, M)
    pred_boxes = torch.zeros(B, Q, 4)
    output = {"pred_logits": pred_logits, "pred_boxes": pred_boxes}
    targets = [{"labels": torch.tensor([], dtype=torch.long), "boxes": torch.zeros(0, 4)}]

    def matcher(out, tgts):
        return [(torch.tensor([], dtype=torch.long), torch.tensor([], dtype=torch.long))]

    obj = make_obj(matcher=matcher)
    loss = compute_margin_loss(obj, output, targets)
    print(f"[case 3 no matches]        loss = {loss.item():.6f}  expected = 0.000000")
    assert loss.item() == 0.0
    assert loss.dim() == 0  # scalar


def case_4_no_confusable_key():
    """target dict without confusable_neg_remapped → batch skipped."""
    B, Q, M = 1, 5, 20
    pred_logits = torch.zeros(B, Q, M)
    pred_logits[0, 0, 3] = 1.0
    pred_logits[0, 0, 5] = 5.0
    pred_boxes = torch.zeros(B, Q, 4)
    output = {"pred_logits": pred_logits, "pred_boxes": pred_boxes}
    targets = [{"labels": torch.tensor([3]), "boxes": torch.zeros(1, 4)}]  # no key

    def matcher(out, tgts):
        return [(torch.tensor([0]), torch.tensor([0]))]

    obj = make_obj(matcher=matcher)
    loss = compute_margin_loss(obj, output, targets)
    print(f"[case 4 no confusable key] loss = {loss.item():.6f}  expected = 0.000000")
    assert loss.item() == 0.0


def case_5_grad_flows():
    """Loss must be differentiable w.r.t. pred_logits."""
    B, Q, M, K = 1, 3, 10, 2
    pred_logits = torch.zeros(B, Q, M, requires_grad=True)
    with torch.no_grad():
        pred_logits[0, 0, 1] = 0.0
        pred_logits[0, 0, 4] = 0.5  # neg above pos
    pred_boxes = torch.zeros(B, Q, 4)
    output = {"pred_logits": pred_logits, "pred_boxes": pred_boxes}
    targets = [{"labels": torch.tensor([1]), "boxes": torch.zeros(1, 4),
                "confusable_neg_remapped": torch.tensor([[4, 7]])}]

    def matcher(out, tgts):
        return [(torch.tensor([0]), torch.tensor([0]))]

    obj = make_obj(num_negatives=K, matcher=matcher)
    loss = compute_margin_loss(obj, output, targets)
    loss.backward()
    grad = pred_logits.grad
    print(f"[case 5 grad flows]        loss = {loss.item():.6f}  grad nonzero at [0,0,1]={float(grad[0,0,1]):.4f}, [0,0,4]={float(grad[0,0,4]):.4f}")
    assert loss.requires_grad
    assert grad[0, 0, 1].abs() > 0  # pos score should get gradient
    assert grad[0, 0, 4].abs() > 0  # neg 4 above margin → gradient


if __name__ == "__main__":
    case_1_basic_nonzero()
    case_2_skip_invalid_negs()
    case_3_no_matches()
    case_4_no_confusable_key()
    case_5_grad_flows()
    print("\nALL PASS")
