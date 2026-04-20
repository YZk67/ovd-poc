"""
Break TPA prototype_queries collinearity saddle by injecting noise.
Only perturbs decoder-level TPAs (class_embed.0 .. class_embed.5).
Encoder TPA (class_embed.6) is left alone — it was already diverse (|cos|~0.35).
"""
import argparse
import torch
import torch.nn.functional as F


def pairwise_cos_stats(w: torch.Tensor):
    K = w.shape[0]
    Pn = F.normalize(w, p=2, dim=-1)
    G = Pn @ Pn.t()
    off = G - torch.eye(K, device=G.device)
    off_abs = off.abs()
    off_mean = off_abs.sum().item() / (K * K - K)
    off_max = off_abs.max().item()
    return off_mean, off_max


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt-in", required=True)
    p.add_argument("--ckpt-out", required=True)
    p.add_argument("--noise-scale", type=float, default=0.3,
                   help="noise std as fraction of each prototype's norm")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    state = torch.load(args.ckpt_in, map_location="cpu")
    model_sd = state["model"] if isinstance(state, dict) and "model" in state else state

    touched = []
    for k in list(model_sd.keys()):
        if "tpa.prototype_queries" not in k:
            continue
        if "class_embed.6" in k:
            continue  # encoder TPA — leave alone
        w = model_sd[k].float()
        mean_norm = w.norm(dim=-1).mean().item()
        noise = torch.randn_like(w) * mean_norm * args.noise_scale

        before_off, before_max = pairwise_cos_stats(w)
        w_new = w + noise
        after_off, after_max = pairwise_cos_stats(w_new)

        model_sd[k] = w_new.to(model_sd[k].dtype)
        touched.append((k, before_off, before_max, after_off, after_max))

    print(f"Perturbed {len(touched)} tensors (noise_scale={args.noise_scale}):")
    print(f"{'name':<60} {'|cos|_before':>14} {'max_before':>11} {'|cos|_after':>13} {'max_after':>10}")
    for k, bo, bm, ao, am in touched:
        print(f"{k:<60} {bo:>14.4f} {bm:>11.4f} {ao:>13.4f} {am:>10.4f}")

    torch.save(state, args.ckpt_out)
    print(f"\nSaved perturbed ckpt -> {args.ckpt_out}")


if __name__ == "__main__":
    main()
