#!/usr/bin/env python3
"""VSAS: Visual Similarity Attribute Selection.

Iteratively selects attributes that generalize to unknown objects by
balancing distribution similarity (JSD) and diversity (cosine distance).

Input: distribution .pth from training + full attribute embeddings .pth
Output: selected attribute embeddings .pth

Usage:
    python tools/owovd/select_attributes_vsas.py \
        --distributions dataset/metadata/att_distributions.pth \
        --att-embeddings dataset/metadata/coco_att_embeddings.pth \
        --output dataset/metadata/coco_vsas_selected.pth \
        --per-class 25 --num-classes 20 --thr 0.5 --alpha 0.2
"""

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F


def jensen_shannon_divergence(p, q):
    """Compute JSD between two distributions."""
    m = 0.5 * (p + q)
    m = m.clamp(min=1e-6)
    jsd = 0.5 * (
        (p * (p / m).clamp(min=1e-6).log()).sum()
        + (q * (q / m).clamp(min=1e-6).log()).sum()
    )
    return jsd


def compute_distribution_similarity(positive_dists, negative_dists):
    """Compute JSD between positive and negative distributions for each attribute.

    Lower JSD = more similar distributions = attribute generalizes to unknowns.
    """
    sims = []
    for i in range(len(positive_dists)):
        pos = positive_dists[i]
        neg = negative_dists[i]
        # Normalize to probability distributions
        pos = pos / pos.sum().clamp(min=1e-8)
        neg = neg / neg.sum().clamp(min=1e-8)
        sims.append(jensen_shannon_divergence(pos, neg))
    return torch.stack(sims)


def select_attributes(
    distribution_sim,
    att_embeddings,
    num_select,
    alpha=0.2,
    use_sigmoid=True,
):
    """Greedy attribute selection balancing JSD similarity and diversity.

    Args:
        distribution_sim: [N] JSD scores (lower = more generalizable).
        att_embeddings: [N, dim] attribute embeddings.
        num_select: Total number of attributes to select.
        alpha: Weight for JSD vs cosine diversity (alpha * JSD + (1-alpha) * cosine).
        use_sigmoid: Use sigmoid on cosine sim matrix.

    Returns:
        selected_indices: List of selected attribute indices.
    """
    att_norm = F.normalize(att_embeddings, p=2, dim=1)
    cosine_sim = torch.matmul(att_norm, att_norm.t())
    if use_sigmoid:
        cosine_sim = cosine_sim.sigmoid()
    else:
        cosine_sim = cosine_sim.abs()

    n_total = len(distribution_sim)
    selected = []

    for step in range(min(num_select, n_total)):
        if len(selected) == 0:
            # First pick: lowest JSD
            idx = distribution_sim.argmin().item()
        else:
            unselected = list(set(range(n_total)) - set(selected))
            # Cosine similarity to already-selected (mean)
            cos_with_selected = cosine_sim[unselected][:, selected].mean(dim=1)
            # JSD for unselected
            jsd_unselected = distribution_sim[unselected]
            # Combined score (lower is better)
            score = alpha * jsd_unselected + (1 - alpha) * cos_with_selected
            idx = unselected[score.argmin().item()]

        selected.append(idx)

    return selected


def main():
    parser = argparse.ArgumentParser(description="VSAS attribute selection")
    parser.add_argument("--distributions", required=True, help="Distribution .pth from training")
    parser.add_argument("--att-embeddings", required=True, help="Full attribute embeddings .pth")
    parser.add_argument("--output", required=True, help="Output .pth with selected attributes")
    parser.add_argument("--per-class", type=int, default=25, help="Attributes per class to select")
    parser.add_argument("--num-classes", type=int, default=20, help="Number of known classes")
    parser.add_argument("--thr", type=float, default=0.5, help="Assignment threshold for distributions")
    parser.add_argument("--alpha", type=float, default=0.2, help="JSD vs diversity weight")
    args = parser.parse_args()

    # Load distributions
    dists = torch.load(args.distributions, map_location="cpu")
    thresholds = dists["thresholds"]
    thr_idx = thresholds.index(args.thr)
    pos_dists = dists["positive_distributions"][thr_idx]
    neg_dists = dists["negative_distributions"][thr_idx]

    # Load full attribute embeddings
    att_data = torch.load(args.att_embeddings, map_location="cpu")
    all_texts = att_data["att_text"]
    all_embeddings = att_data["att_embedding"]

    print(f"Total candidate attributes: {len(all_texts)}")
    print(f"Threshold: {args.thr}, Alpha: {args.alpha}")

    # Compute JSD similarity
    dist_sim = compute_distribution_similarity(pos_dists, neg_dists)
    print(f"Distribution similarity range: [{dist_sim.min():.4f}, {dist_sim.max():.4f}]")

    # Select attributes
    num_select = args.per_class * args.num_classes
    print(f"Selecting {num_select} attributes ({args.per_class} per class x {args.num_classes} classes)...")

    selected_indices = select_attributes(
        dist_sim, all_embeddings, num_select, alpha=args.alpha
    )

    selected_texts = [all_texts[i] for i in selected_indices]
    selected_embeddings = all_embeddings[selected_indices]

    print(f"\nSelected {len(selected_indices)} attributes")
    print(f"Top 10: {selected_texts[:10]}")

    # Save
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "att_text": selected_texts,
            "att_embedding": selected_embeddings,
            "selected_indices": selected_indices,
            "alpha": args.alpha,
            "threshold": args.thr,
            "per_class": args.per_class,
        },
        out_path,
    )
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
