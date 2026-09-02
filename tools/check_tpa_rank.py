"""Effective rank of a checkpoint's TPA prototypes, standalone.

Reads the TPA tensors straight out of the .pth and replays the aggregator by
hand, so it works for any Kp without needing a matching config -- instantiating
the model from a Kp=5 config against a Kp=3 checkpoint would silently skip
prototype_queries on shape mismatch and measure random weights instead.

Replays the ORIGINAL forward, sqrt(d) division included, because the question is
what the prototypes were under the code that trained this checkpoint.

    python tools/check_tpa_rank.py <ckpt.pth> <prompt_bank.npy> [tau_p]
"""
import math
import sys

import numpy as np
import torch
import torch.nn.functional as F

ck = torch.load(sys.argv[1], map_location="cpu")
sd = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
hits = sorted(k for k in sd if k.endswith("tpa.key_proj.weight"))
if not hits:
    sys.exit(f"[!] no TPA weights in {sys.argv[1]} -- was this run trained with use_tpa=True?")
pre = hits[0][: -len("key_proj.weight")]
tau = float(sys.argv[3]) if len(sys.argv) > 3 else 0.004375

T = torch.from_numpy(np.load(sys.argv[2])).float()
q = sd[pre + "prototype_queries"].float()
kw, kb = sd[pre + "key_proj.weight"].float(), sd[pre + "key_proj.bias"].float()
vw, vb = sd[pre + "value_proj.weight"].float(), sd[pre + "value_proj.bias"].float()

attention_scale = math.sqrt(kw.shape[0]) * tau
logits = torch.einsum("kh,cnh->ckn", q, F.linear(T, kw, kb))
P = torch.einsum(
    "ckn,cnd->ckd",
    torch.softmax(logits / attention_scale, dim=-1),
    F.linear(T, vw, vb),
)

Pn = F.normalize(P, dim=-1)
G = torch.einsum("ckd,cmd->ckm", Pn, Pn)
Kp = G.size(-1)
if Kp < 2:
    print(f"Kp=1 -- effective rank is 1.000 by definition")
    sys.exit()
off = (G.sum((-2, -1)) - G.diagonal(dim1=-2, dim2=-1).sum(-1)) / (Kp * Kp - Kp)
sv = torch.linalg.eigvalsh(G.float()).clamp_min(0.0).sqrt()
fr = sv / sv.sum(-1, keepdim=True).clamp_min(1e-12)
rank = torch.exp(-(fr * fr.clamp_min(1e-12).log()).sum(-1))
print(f"Kp={Kp}  tau={tau}  pairwise_cos={off.mean():.5f}  eff_rank={rank.mean():.4f}  "
      f"(collapsed ~1.0, ceiling {min(Kp, T.shape[1], P.shape[-1])})")
