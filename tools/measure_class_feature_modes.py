"""Measure how multi-modal the detector's per-class box features actually are.

The multi-prototype premise is that a class occupies several distinct regions of
the embedding space, so K prototypes can each own one. Nothing has ever checked
that. This collects the features that reach the classifier -- `linear(x)`, the
exact vector compared against the prototypes -- groups them by predicted class,
and measures the within-class spread.

The headline number is `gain above null`: how much better 5 spherical k-means
centroids cover the class's boxes than a single one, minus the same gain on a
one-mode cloud matched to the same concentration. Calibrated on synthetic data
it reads -0.000 for a single mode at any width and +0.36 to +0.41 for 3-5
separated modes.

Near zero means the features carry no structure for a second prototype to
capture, and no amount of aggregation/temperature/regulariser work on the text
side can help -- the multi-prototype premise would simply not hold for this
detector. `rank8` is reported alongside for comparison with the text bank's
5.462, but it is NOT the judge: it ranks a loose single mode (7.82) above a
tight three-mode cloud (5.68).

    python tools/measure_class_feature_modes.py \
        --ckpt /root/autodl-tmp/tpa_fix_k5_2ep/model_final.pth \
        --num-images 500
"""
import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def effective_rank(feats: torch.Tensor) -> float:
    """exp(entropy of the normalised singular values) of L2-normalised rows.

    Same definition as compute_prototype_similarity, so the numbers are directly
    comparable to the prototype/text-bank ranks quoted elsewhere.
    """
    x = F.normalize(feats.float(), dim=-1)
    gram = x @ x.t()
    evals = torch.linalg.eigvalsh(gram).clamp_min(0.0)
    svals = evals.sqrt()
    frac = svals / svals.sum().clamp_min(1e-12)
    return float(torch.exp(-(frac * frac.clamp_min(1e-12).log()).sum()))


def cover_cos(feats: torch.Tensor, k: int, iters: int = 30, seed: int = 0) -> float:
    """Mean cosine from each box to its nearest of k spherical k-means centroids.

    This is the quantity the classifier actually maximises, so `cover_cos(k=5) -
    cover_cos(k=1)` is exactly what K=5 prototypes could buy over K=1 for this
    class -- an upper bound, since real prototypes are built from text and are
    not free to sit on the visual centroids.

    Validated on synthetic clouds in 768d: the k=5 minus k=1 gain is 0.001-0.021
    for a single mode at any width, and 0.39-0.42 for 3-5 separated modes. Two
    orders of magnitude apart, so the statistic is decisive.

    Deliberately not PCA: the fraction of variance on the leading component does
    NOT separate one mode from several. Measured on the same synthetic data, one
    isotropic mode gives 0.036 and three well-separated modes give 0.131 -- the
    multi-modal case scores HIGHER, so reading it as a unimodality score inverts
    the answer. `effective_rank` is no good for this either: a loose single mode
    scores 7.82 and a tight 3-mode cloud scores 5.68, i.e. the wrong way round.
    """
    x = F.normalize(feats.float(), dim=-1)
    if k == 1:
        return float((x @ F.normalize(x.mean(0), dim=-1)).mean())
    g = torch.Generator().manual_seed(seed)
    c = x[torch.randperm(len(x), generator=g)[:k]].clone()
    for _ in range(iters):
        assign = (x @ c.t()).argmax(dim=1)
        for j in range(k):
            sel = x[assign == j]
            if len(sel):
                c[j] = F.normalize(sel.mean(0), dim=-1)
    return float((x @ c.t()).max(dim=1).values.mean())


def matched_unimodal_null(feats: torch.Tensor, seed: int = 0) -> torch.Tensor:
    """One mode, same shape as `feats`, bisected to the same cos@1.

    k-means gains something on any cloud in 768d, and how much depends almost
    entirely on how concentrated the cloud is -- an isotropic null scores 0.102,
    higher than any genuinely single-mode cloud. So the null has to be matched on
    concentration or it flatters the data it is supposed to challenge.
    """
    target = cover_cos(feats, 1)
    g = torch.Generator().manual_seed(seed)
    n, d = feats.shape
    direction = F.normalize(torch.randn(d, generator=g), dim=-1)
    noise = torch.randn(n, d, generator=g) / d ** 0.5
    lo, hi = 1e-3, 50.0
    for _ in range(30):                      # cos@1 falls monotonically in sigma
        mid = 0.5 * (lo + hi)
        if cover_cos(F.normalize(direction + mid * noise, dim=-1), 1) > target:
            lo = mid
        else:
            hi = mid
    return F.normalize(direction + 0.5 * (lo + hi) * noise, dim=-1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="lami_dino/configs/dino_convnext_large_4scale_12ep_lvis.py")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--bank", default="dataset/metadata/lvis_claude_prompts_convnextl.npy")
    p.add_argument("--num-images", type=int, default=500)
    p.add_argument("--score-thresh", type=float, default=0.3)
    p.add_argument("--min-boxes", type=int, default=50, help="classes with fewer boxes are skipped")
    p.add_argument("--max-boxes", type=int, default=300, help="cap kept per class, to bound memory")
    p.add_argument("--draws", type=int, default=20, help="random 8-box draws averaged per class")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", default=None, help="optional .npz dump of the per-class features")
    args = p.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from detectron2.checkpoint import DetectionCheckpointer
    from detectron2.config import LazyConfig, instantiate

    cfg = LazyConfig.load(args.config)
    cfg.dataloader.test.num_workers = 0
    model = instantiate(cfg.model).to(args.device).eval()
    DetectionCheckpointer(model).load(args.ckpt)

    # Hook every class_embed rather than [-1]: there are 7 of them for 6 decoder
    # layers, so [-1] is most likely the two-stage encoder head, not the final
    # decoder layer. Each hook overwrites `captured`, so after the forward pass
    # it holds whichever one ran last -- the layer the detections come from.
    heads = model.transformer.decoder.class_embed
    captured = {}

    def make_hook(idx):
        def hook(module, inputs, output):
            # forward(x, classifier=None, content_inds=None, additional_class=None):
            # the vector actually compared against the prototypes is linear(x).
            with torch.no_grad():
                captured["feat"] = module.linear(inputs[0]).detach()
            captured["logits"] = output.detach()
            captured["idx"] = idx
        return hook

    handles = [h.register_forward_hook(make_hook(i)) for i, h in enumerate(heads)]

    loader = instantiate(cfg.dataloader.test)
    per_class = defaultdict(list)
    seen = 0
    with torch.no_grad():
        for batch in loader:
            captured.clear()
            model(batch)
            if "feat" not in captured:
                sys.exit("[!] no class_embed fired -- check model.transformer.decoder.class_embed")
            if seen == 0:
                print(f"[hook] features taken from class_embed[{captured['idx']}] "
                      f"(last of {len(heads)} to run), shape {tuple(captured['feat'].shape)}")
            feat, logits = captured["feat"], captured["logits"]
            if feat.ndim == 3:                      # [B, Q, D] -> [B*Q, D]
                feat, logits = feat.flatten(0, 1), logits.flatten(0, 1)
            scores, labels = logits.sigmoid().max(dim=-1)
            keep = scores >= args.score_thresh
            for f, c in zip(feat[keep].cpu(), labels[keep].cpu().tolist()):
                if len(per_class[c]) < args.max_boxes:
                    per_class[c].append(f)
            seen += len(batch)
            if seen % 50 < len(batch):
                enough = sum(1 for v in per_class.values() if len(v) >= args.min_boxes)
                print(f"  [{seen}/{args.num_images}] classes with >={args.min_boxes} boxes: {enough}")
            if seen >= args.num_images:
                break
    for h in handles:
        h.remove()

    usable = {c: torch.stack(v) for c, v in per_class.items() if len(v) >= args.min_boxes}
    if not usable:
        sys.exit(f"[!] no class reached {args.min_boxes} boxes -- lower --score-thresh or raise --num-images")

    bank = torch.from_numpy(np.load(args.bank)).float()
    bank_rank = float(np.mean([effective_rank(bank[c]) for c in range(bank.shape[0])]))

    g = torch.Generator().manual_seed(0)
    rows = []
    for c, feats in usable.items():
        r8 = float(np.mean([
            effective_rank(feats[torch.randperm(len(feats), generator=g)[:8]])
            for _ in range(args.draws)
        ]))
        c1, c5 = cover_cos(feats, 1), cover_cos(feats, 5)
        null = matched_unimodal_null(feats, seed=c)
        n1, n5 = cover_cos(null, 1), cover_cos(null, 5)
        rows.append((c, len(feats), r8, effective_rank(feats), c1, c5, c5 - c1, n5 - n1))
    rows.sort(key=lambda r: -r[1])

    print(f"\n{len(rows)} classes with >={args.min_boxes} boxes, from {seen} images\n")
    print(f"{'class':>6s} {'n':>5s} {'rank8':>7s} {'rank_all':>9s} "
          f"{'cos@1':>7s} {'cos@5':>7s} {'gain':>7s} {'null':>7s}")
    for c, n, r8, rf, c1, c5, gain, ngain in rows[:20]:
        print(f"{c:6d} {n:5d} {r8:7.3f} {rf:9.3f} {c1:7.3f} {c5:7.3f} {gain:7.3f} {ngain:7.3f}")
    if len(rows) > 20:
        print(f"   ... {len(rows) - 20} more")

    r8s = np.array([r[2] for r in rows])
    gains = np.array([r[6] for r in rows])
    nulls = np.array([r[7] for r in rows])
    print(f"\n{'':24s}{'rank8':>8s}")
    print(f"{'box features (mean)':24s}{r8s.mean():8.3f}")
    print(f"{'box features (p10/p90)':24s}{np.percentile(r8s, 10):8.3f} / {np.percentile(r8s, 90):.3f}")
    print(f"{'text bank, 8 prompts':24s}{bank_rank:8.3f}   <- reference (max possible 8.0)")
    print(f"\nk=5 vs k=1 cosine gain: {gains.mean():.4f}   matched 1-mode null: {nulls.mean():.4f}")
    print(f"gain above null:        {gains.mean() - nulls.mean():+.4f}")
    print("\nGain at or below the null => 5 prototypes buy nothing over 1 for these\n"
          "features, and no aggregation/temperature/regulariser change can fix that.")

    if args.out:
        np.savez_compressed(args.out, **{str(c): v.numpy() for c, v in usable.items()})
        print(f"[save] {args.out}")


if __name__ == "__main__":
    main()
