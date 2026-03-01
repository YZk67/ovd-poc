"""Calibrate OLN pseudo_weight using a 2-component Weibull mixture model.

OLN generates pseudo proposals with raw confidence scores (0.3–1.0).
These are stored as `pseudo_weight` in the training JSON.
A 2-component Weibull mixture (background + foreground) is fitted via EM
to convert raw scores into calibrated P(foreground | score).

The model already uses `pseudo_weight` for loss weighting in
two_stage_criterion._get_matched_weights() — this script just improves quality.
No model code changes needed.

Usage (run from project root on remote server):
    python tools/calibrate_weibull_weights.py \\
        --input  dataset/metadata/train_with_pseudo_unknown.json \\
        --output dataset/metadata/train_with_pseudo_weibull.json

Then update the training config:
    dataloader.train.dataset.names = "ovd_coco_train_with_pseudo_weibull"
and register the new dataset name in detectron2/.../builtin.py.

Requirements: scipy (pip install scipy)
"""

import argparse
import json

import numpy as np


def _wpdf(x, c, scale):
    """Weibull PDF (2-param, loc=0) clipped to avoid log(0)."""
    from scipy.stats import weibull_min

    return weibull_min.pdf(x, c, scale=scale, loc=0).clip(1e-300)


def fit_weibull_mixture(scores, n_iter=80):
    """Fit 2-component Weibull mixture via EM and return P(fg | score).

    Component 0: background (low scores, typically c~1, scale~0.3)
    Component 1: foreground (high scores, typically c~3-5, scale~0.7)
    """
    from scipy.optimize import minimize

    scores = np.asarray(scores, dtype=np.float64).clip(1e-6, 1 - 1e-6)

    # Init: separate low/high scores as background/foreground
    median = float(np.median(scores))
    pi = np.array([0.5, 0.5])
    params = [[1.0, min(median, 0.4)], [2.0, max(median, 0.6)]]  # [c, scale] each

    for iteration in range(n_iter):
        # ---------- E-step: compute responsibilities ----------
        r = np.stack([pi[k] * _wpdf(scores, *params[k]) for k in range(2)], axis=1)
        row_sum = r.sum(axis=1, keepdims=True)
        # Guard against zero row_sum (numerical underflow)
        row_sum = np.where(row_sum == 0, 1e-300, row_sum)
        r /= row_sum

        # ---------- M-step: update mixing weights ----------
        pi = r.mean(axis=0)
        pi = pi.clip(1e-6, 1 - 1e-6)

        # ---------- M-step: update Weibull params via weighted MLE ----------
        for k in range(2):
            w = r[:, k]

            def neg_ll(p, w=w):
                c = max(p[0], 0.05)
                s = max(p[1], 0.01)
                return -(w * np.log(_wpdf(scores, c, s))).sum()

            res = minimize(
                neg_ll,
                params[k],
                method="Nelder-Mead",
                options={"maxiter": 200, "xatol": 1e-5, "fatol": 1e-7},
            )
            params[k] = [max(res.x[0], 0.05), max(res.x[1], 0.01)]

    # Identify foreground component by higher Weibull mean
    from scipy.stats import weibull_min

    means = [weibull_min.mean(*p) for p in params]
    fg_idx = int(np.argmax(means))
    bg_idx = 1 - fg_idx

    print(
        f"  BG component: c={params[bg_idx][0]:.3f}, scale={params[bg_idx][1]:.3f}, "
        f"mean={means[bg_idx]:.3f}, weight={pi[bg_idx]:.3f}"
    )
    print(
        f"  FG component: c={params[fg_idx][0]:.3f}, scale={params[fg_idx][1]:.3f}, "
        f"mean={means[fg_idx]:.3f}, weight={pi[fg_idx]:.3f}"
    )

    # P(fg | score) = pi_fg * pdf_fg / (pi_bg * pdf_bg + pi_fg * pdf_fg)
    pdf_bg = _wpdf(scores, *params[bg_idx])
    pdf_fg = _wpdf(scores, *params[fg_idx])
    p_fg = (pi[fg_idx] * pdf_fg) / (pi[bg_idx] * pdf_bg + pi[fg_idx] * pdf_fg)
    return p_fg.clip(0.0, 1.0)


def main():
    parser = argparse.ArgumentParser(
        description="Calibrate OLN pseudo_weight via Weibull mixture model"
    )
    parser.add_argument(
        "--input",
        default="dataset/metadata/train_with_pseudo_unknown.json",
        help="Input COCO JSON with pseudo annotations",
    )
    parser.add_argument(
        "--output",
        default="dataset/metadata/train_with_pseudo_weibull.json",
        help="Output COCO JSON with updated pseudo_weight",
    )
    parser.add_argument(
        "--n-iter",
        type=int,
        default=80,
        help="Number of EM iterations",
    )
    args = parser.parse_args()

    try:
        import scipy  # noqa: F401
    except ImportError:
        raise ImportError("scipy not found. Run: pip install scipy")

    print(f"Loading JSON from {args.input} ...")
    with open(args.input) as f:
        data = json.load(f)

    pseudo_indices = [
        i for i, a in enumerate(data["annotations"]) if a.get("is_pseudo")
    ]
    scores = [data["annotations"][i]["score"] for i in pseudo_indices]

    print(
        f"Found {len(scores)} pseudo annotations. "
        f"Score range: [{min(scores):.3f}, {max(scores):.3f}]"
    )
    print(f"Fitting 2-component Weibull mixture (n_iter={args.n_iter}) ...")

    p_fg = fit_weibull_mixture(scores, n_iter=args.n_iter)

    print(
        f"P(fg) calibrated weights: "
        f"mean={p_fg.mean():.3f}, std={p_fg.std():.3f}, "
        f"min={p_fg.min():.3f}, max={p_fg.max():.3f}"
    )
    pct_high = (p_fg > 0.7).mean() * 100
    pct_low = (p_fg < 0.3).mean() * 100
    print(f"  P(fg) > 0.7: {pct_high:.1f}%  |  P(fg) < 0.3: {pct_low:.1f}%")

    for i, ann_idx in enumerate(pseudo_indices):
        data["annotations"][ann_idx]["pseudo_weight"] = float(p_fg[i])

    print(f"Saving calibrated JSON to {args.output} ...")
    with open(args.output, "w") as f:
        json.dump(data, f)
    print("Done.")


if __name__ == "__main__":
    main()
