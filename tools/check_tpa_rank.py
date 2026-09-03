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
slot_prior_strength = float(sd.get(pre + "slot_prior_strength", torch.tensor(0.0)))
prototype_mode_strength = float(
    sd.get(pre + "prototype_mode_strength", torch.tensor(0.0))
)

attention_scale = math.sqrt(kw.shape[0]) * tau
logits = torch.einsum("kh,cnh->ckn", q, F.linear(T, kw, kb))
if slot_prior_strength > 0.0:
    Kp, num_prompts = q.shape[0], T.shape[1]
    if Kp > num_prompts:
        sys.exit(
            "[!] slot-prior checkpoint has more prototypes than prompts: "
            f"Kp={Kp}, N={num_prompts}"
        )
    positions = torch.linspace(0, num_prompts - 1, steps=Kp).round().long()
    prior = logits.new_zeros((Kp, num_prompts))
    prior.scatter_(1, positions[:, None], slot_prior_strength)
    logits = logits + prior.unsqueeze(0)

values = F.linear(T, vw, vb)
P = torch.einsum(
    "ckn,cnd->ckd",
    torch.softmax(logits / attention_scale, dim=-1),
    values,
)
if prototype_mode_strength > 0.0:
    centroid = values.mean(dim=1, keepdim=True)
    residual = P - centroid
    unit_residual = residual / residual.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    mode_radius = prototype_mode_strength * centroid.norm(
        dim=-1, keepdim=True
    ).clamp_min(1e-6)
    P = centroid + mode_radius * unit_residual

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
print(f"Kp={Kp}  tau={tau}  slot_prior={slot_prior_strength:g}  "
      f"mode_strength={prototype_mode_strength:g}  pairwise_cos={off.mean():.5f}  "
      f"eff_rank={rank.mean():.4f}  "
      f"(collapsed ~1.0, ceiling {min(Kp, T.shape[1], P.shape[-1])})")
