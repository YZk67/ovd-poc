"""
Generate Figure 4 of the paper: RPSA region-cluster overlays on LVIS images.

Visualizes the soft k-means cluster assignments that RPSA computes on the
encoder's region features. RPSA itself is only invoked during training, but
its first step (soft k-means over encoder tokens) is fully deterministic
given the same trained encoder, so we replicate it at eval time on the
captured `output_memory` to produce overlays.

Pipeline:
  1. Load trained model (eval mode).
  2. Monkey-patch `transformer.gen_encoder_output_proposals` to capture the
     post-encoder output_memory [B,N,D] and per-level spatial_shapes [L,2].
  3. Forward N images.
  4. Run soft_kmeans_assign(output_memory, K, sigma, em_iters) -> [B,N,K].
  5. Slice tokens by level, reshape to [H_l, W_l], take argmax-K, color and
     overlay on the resized model-input image.

Run on the training server (1 GPU enough; ~15s per image):
    python tools/plot_rpsa_clusters.py \
        --config lami_dino/configs/dino_convnext_large_4scale_12ep_lvis.py \
        --ckpt   /root/autodl-tmp/model_final.pth \
        --num-images 6 \
        --feature-level 0 \
        --output-dir /root/autodl-tmp/rpsa_figs

Then scp the figures locally for paper inclusion.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch


def install_capture_hook(model, capture):
    """Replace transformer.gen_encoder_output_proposals so each call records
    the post-encoder features and spatial shapes."""
    transformer = model.transformer
    original = transformer.gen_encoder_output_proposals

    def patched(memory, mask, spatial_shapes):
        output_memory, output_proposals = original(memory, mask, spatial_shapes)
        capture["output_memory"] = output_memory.detach()
        capture["spatial_shapes"] = spatial_shapes.detach().cpu()
        return output_memory, output_proposals

    transformer.gen_encoder_output_proposals = patched


def slice_levels(tokens, spatial_shapes):
    """Split [B,N,C] into list of [B,H_l,W_l,C] per encoder level."""
    levels = []
    cursor = 0
    B = tokens.size(0)
    C = tokens.size(-1)
    for H, W in spatial_shapes.tolist():
        n = int(H) * int(W)
        slc = tokens[:, cursor : cursor + n, :].reshape(B, int(H), int(W), C)
        levels.append(slc)
        cursor += n
    return levels


def overlay_clusters_on_image(img_rgb, cluster_grid, K, alpha):
    """img_rgb: HxWx3 uint8. cluster_grid: H_l x W_l int."""
    import matplotlib.pyplot as plt
    from PIL import Image

    H, W = img_rgb.shape[:2]
    cmap = plt.cm.get_cmap("tab10", K)
    palette = (np.array([cmap(i)[:3] for i in range(K)]) * 255).astype(np.uint8)
    cluster_rgb = palette[cluster_grid]  # H_l x W_l x 3
    cluster_rgb_resized = np.array(
        Image.fromarray(cluster_rgb).resize((W, H), Image.NEAREST)
    )
    overlay = (
        (1.0 - alpha) * img_rgb.astype(np.float32)
        + alpha * cluster_rgb_resized.astype(np.float32)
    )
    return overlay.clip(0, 255).astype(np.uint8)


def fig_for_image(img_rgb, all_level_grids, K, alpha, file_label, out_pdf):
    """Compose: original | overlay@level0 | overlay@level1 | ... in one row."""
    import matplotlib.pyplot as plt

    n_panels = 1 + len(all_level_grids)
    fig, axes = plt.subplots(1, n_panels, figsize=(2.4 * n_panels, 2.4))
    if n_panels == 1:
        axes = [axes]

    axes[0].imshow(img_rgb)
    axes[0].set_title(f"input ({file_label})", fontsize=7)
    axes[0].axis("off")

    for j, grid in enumerate(all_level_grids):
        overlay = overlay_clusters_on_image(img_rgb, grid, K, alpha)
        H_l, W_l = grid.shape
        axes[j + 1].imshow(overlay)
        axes[j + 1].set_title(f"level {j} ({H_l}x{W_l}, K={K})", fontsize=7)
        axes[j + 1].axis("off")

    fig.tight_layout()
    fig.savefig(out_pdf, bbox_inches="tight", dpi=180)
    plt.close(fig)


def reconstruct_image_from_input(batched_input):
    """detectron2 mapper outputs CHW tensor in RGB (img_format='RGB' in this repo)."""
    img = batched_input["image"]
    if isinstance(img, torch.Tensor):
        arr = img.permute(1, 2, 0).cpu().numpy()
        if arr.dtype != np.uint8:
            arr = arr.clip(0, 255).astype(np.uint8)
        return arr
    return np.asarray(img)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config",
                   default="lami_dino/configs/dino_convnext_large_4scale_12ep_lvis.py")
    p.add_argument("--ckpt", default="/root/autodl-tmp/model_final.pth")
    p.add_argument("--num-images", type=int, default=6)
    p.add_argument("--K", type=int, default=8,
                   help="num clusters (match training: model.transformer.rpsa_module.K)")
    p.add_argument("--sigma", type=float, default=1.0)
    p.add_argument("--em-iters", type=int, default=1)
    p.add_argument("--feature-level", type=int, default=-1,
                   help="if >=0, render only this level; else render all levels in one row")
    p.add_argument("--alpha", type=float, default=0.55, help="overlay opacity")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--skip-images", type=int, default=0,
                   help="skip first N images (useful to find diverse examples)")
    p.add_argument("--output-dir", default="/root/autodl-tmp/rpsa_figs")
    args = p.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(repo_root))

    from detectron2.checkpoint import DetectionCheckpointer
    from detectron2.config import LazyConfig, instantiate
    from detectron2.data import (
        build_detection_test_loader,
        get_detection_dataset_dicts,
    )

    from lami_dino.models.rpsa import soft_kmeans_assign

    print(f"[load] config: {args.config}")
    cfg = LazyConfig.load(args.config)
    print(f"[load] instantiating model")
    model = instantiate(cfg.model)
    model.eval()
    if torch.cuda.is_available():
        model.cuda()
    print(f"[load] checkpoint: {args.ckpt}")
    DetectionCheckpointer(model).load(args.ckpt)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    capture = {}
    install_capture_hook(model, capture)

    dataset_dicts = get_detection_dataset_dicts(
        names="lvis_v1_val", filter_empty=False
    )
    end = args.skip_images + args.num_images
    dataset_dicts = dataset_dicts[args.skip_images:end]
    loader = build_detection_test_loader(
        dataset=dataset_dicts,
        mapper=instantiate(cfg.dataloader.test.mapper),
        num_workers=0,
    )

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    print(f"[run] inference + cluster vis on {len(dataset_dicts)} images...")
    with torch.no_grad():
        for i, batched_inputs in enumerate(loader):
            _ = model(batched_inputs)

            output_memory = capture.get("output_memory")        # [1,N,D]
            spatial_shapes = capture.get("spatial_shapes")      # [L,2]
            if output_memory is None:
                print(f"[!] image {i}: no capture; skipping")
                continue

            r, _ = soft_kmeans_assign(
                output_memory, K=args.K,
                sigma=args.sigma, iters=args.em_iters,
            )
            r_cpu = r.detach().cpu()  # [1, N, K]

            level_tensors = slice_levels(r_cpu, spatial_shapes)  # list of [1,H,W,K]
            level_grids = [t[0].argmax(dim=-1).numpy() for t in level_tensors]

            img_rgb = reconstruct_image_from_input(batched_inputs[0])
            file_label = Path(batched_inputs[0].get("file_name", f"img_{i}")).stem

            if args.feature_level >= 0:
                lvl = min(args.feature_level, len(level_grids) - 1)
                level_grids = [level_grids[lvl]]

            outpath = out_dir / f"rpsa_overlay_{i:02d}_{file_label}.pdf"
            fig_for_image(
                img_rgb, level_grids, args.K, args.alpha, file_label, outpath
            )
            unique_per_level = [int(np.unique(g).size) for g in level_grids]
            print(f"  [{i+1}/{len(dataset_dicts)}] {file_label}: "
                  f"levels={[g.shape for g in level_grids]} "
                  f"unique_clusters={unique_per_level}  -> {outpath.name}")

    print(f"\n[done] {len(dataset_dicts)} overlays in {out_dir}")


if __name__ == "__main__":
    main()
