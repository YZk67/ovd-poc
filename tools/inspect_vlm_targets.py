"""检查 VLM soft targets 中 novel 类的分布情况。

用法:
    python tools/inspect_vlm_targets.py dataset/vlm_targets_crop_ab_backup/vlm_soft_targets.pth
"""
import sys
import torch
import numpy as np
from collections import defaultdict

ALL_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "kite", "skateboard", "surfboard", "bottle", "cup", "fork", "knife", "spoon", "bowl", "banana",
    "apple", "sandwich", "orange", "broccoli", "carrot", "pizza", "donut", "cake", "chair", "couch",
    "bed", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "microwave", "oven", "toaster",
    "sink", "refrigerator", "book", "clock", "vase", "scissors", "toothbrush"
]

SEEN_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "train", "truck", "boat", "bench", "bird",
    "horse", "sheep", "bear", "zebra", "giraffe", "backpack", "handbag", "suitcase", "frisbee",
    "skis", "kite", "surfboard", "bottle", "fork", "spoon", "bowl", "banana", "apple", "sandwich",
    "orange", "broccoli", "carrot", "pizza", "donut", "chair", "bed", "toilet", "tv", "laptop",
    "mouse", "remote", "microwave", "oven", "toaster", "refrigerator", "book", "clock", "vase",
    "toothbrush"
]

NOVEL_CLASSES = [c for c in ALL_CLASSES if c not in SEEN_CLASSES]
NOVEL_INDICES = [ALL_CLASSES.index(c) for c in NOVEL_CLASSES]
BASE_INDICES = [ALL_CLASSES.index(c) for c in SEEN_CLASSES]


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "dataset/vlm_targets_crop_ab_backup/vlm_soft_targets.pth"
    print(f"Loading: {path}")
    data = torch.load(path, map_location="cpu", weights_only=False)

    print(f"Images: {len(data)}")

    # Collect all similarity distributions
    all_dists = []
    n_none = 0
    n_total = 0
    # Per-annotation argmax stats
    argmax_counts = np.zeros(len(ALL_CLASSES), dtype=np.int64)
    # Novel probability mass stats
    novel_mass_list = []
    # Top class for each annotation
    top_novel_probs = []  # max novel prob per annotation

    for img_id, targets in data.items():
        if targets is None:
            continue
        for t in targets:
            n_total += 1
            if t is None:
                n_none += 1
                continue
            dist = t["similarity_distribution"]
            if isinstance(dist, torch.Tensor):
                dist = dist.numpy()
            all_dists.append(dist)
            argmax_counts[np.argmax(dist)] += 1
            novel_mass = dist[NOVEL_INDICES].sum()
            novel_mass_list.append(novel_mass)
            top_novel_probs.append(dist[NOVEL_INDICES].max())

    all_dists = np.array(all_dists)
    novel_mass_arr = np.array(novel_mass_list)
    top_novel_arr = np.array(top_novel_probs)

    print(f"Total annotations: {n_total}, None: {n_none}, Valid: {len(all_dists)}")
    print()

    # 1. Average distribution across all annotations
    mean_dist = all_dists.mean(axis=0)
    print("=" * 70)
    print("平均 similarity distribution (所有 annotation 的均值)")
    print("=" * 70)
    sorted_idx = np.argsort(-mean_dist)
    for i, idx in enumerate(sorted_idx[:20]):
        is_novel = " [NOVEL]" if ALL_CLASSES[idx] in NOVEL_CLASSES else ""
        print(f"  {i+1:2d}. {ALL_CLASSES[idx]:20s}: {mean_dist[idx]:.4f}{is_novel}")
    print(f"  ...")
    print(f"  Novel 类总概率质量 (均值): {mean_dist[NOVEL_INDICES].sum():.4f}")
    print(f"  Base  类总概率质量 (均值): {mean_dist[BASE_INDICES].sum():.4f}")
    print()

    # 2. Argmax distribution
    print("=" * 70)
    print("Argmax 分布 (每个 annotation 的最高概率类)")
    print("=" * 70)
    novel_argmax = sum(argmax_counts[i] for i in NOVEL_INDICES)
    base_argmax = sum(argmax_counts[i] for i in BASE_INDICES)
    print(f"  Argmax 是 novel 类: {novel_argmax} ({novel_argmax/len(all_dists)*100:.2f}%)")
    print(f"  Argmax 是 base  类: {base_argmax} ({base_argmax/len(all_dists)*100:.2f}%)")
    print()
    print("  Top 20 argmax 类:")
    sorted_argmax = np.argsort(-argmax_counts)
    for i, idx in enumerate(sorted_argmax[:20]):
        if argmax_counts[idx] == 0:
            break
        is_novel = " [NOVEL]" if ALL_CLASSES[idx] in NOVEL_CLASSES else ""
        print(f"    {ALL_CLASSES[idx]:20s}: {argmax_counts[idx]:6d}{is_novel}")
    print()

    # 3. Novel probability mass distribution
    print("=" * 70)
    print("每个 annotation 中 novel 类的总概率质量")
    print("=" * 70)
    print(f"  Mean:   {novel_mass_arr.mean():.4f}")
    print(f"  Median: {np.median(novel_mass_arr):.4f}")
    print(f"  Std:    {novel_mass_arr.std():.4f}")
    print(f"  Min:    {novel_mass_arr.min():.4f}")
    print(f"  Max:    {novel_mass_arr.max():.4f}")
    print(f"  >0.1:   {(novel_mass_arr > 0.1).sum()} ({(novel_mass_arr > 0.1).mean()*100:.2f}%)")
    print(f"  >0.3:   {(novel_mass_arr > 0.3).sum()} ({(novel_mass_arr > 0.3).mean()*100:.2f}%)")
    print(f"  >0.5:   {(novel_mass_arr > 0.5).sum()} ({(novel_mass_arr > 0.5).mean()*100:.2f}%)")
    print()

    # 4. Per novel class average probability
    print("=" * 70)
    print("每个 novel 类的平均概率")
    print("=" * 70)
    for cls_name in NOVEL_CLASSES:
        idx = ALL_CLASSES.index(cls_name)
        avg_prob = all_dists[:, idx].mean()
        max_prob = all_dists[:, idx].max()
        argmax_n = argmax_counts[idx]
        print(f"  {cls_name:20s}: avg={avg_prob:.5f}  max={max_prob:.4f}  argmax_count={argmax_n}")
    print()

    # 5. Sample a few high-novel-mass annotations
    print("=" * 70)
    print("Novel 概率质量最高的 5 个 annotation 示例")
    print("=" * 70)
    top5_idx = np.argsort(-novel_mass_arr)[:5]
    sample_i = 0
    for img_id, targets in data.items():
        if targets is None:
            continue
        for t in targets:
            if t is None:
                continue
            if sample_i in top5_idx:
                dist = t["similarity_distribution"]
                if isinstance(dist, torch.Tensor):
                    dist = dist.numpy()
                novel_m = dist[NOVEL_INDICES].sum()
                print(f"\n  img_id={img_id}, ann_id={t.get('ann_id', '?')}, novel_mass={novel_m:.4f}")
                if "description" in t:
                    print(f"  description: {t['description'][:100]}")
                top3 = np.argsort(-dist)[:5]
                for j in top3:
                    is_novel = " [NOVEL]" if ALL_CLASSES[j] in NOVEL_CLASSES else ""
                    print(f"    {ALL_CLASSES[j]:20s}: {dist[j]:.4f}{is_novel}")
            sample_i += 1


if __name__ == "__main__":
    main()
