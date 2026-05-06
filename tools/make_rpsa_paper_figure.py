"""
Compose the final paper Figure 4: RPSA region-cluster overlays.

Runs inference on a fixed list of LVIS val image-ids, captures the
post-encoder features via the same hook as plot_rpsa_clusters.py,
runs soft k-means to get cluster assignments, and composes a single
2-row x N-col PDF figure (top row: input; bottom row: overlay) ready
for direct \\includegraphics inclusion in the LaTeX source.

Default selection (NeurIPS Figure 4 candidates):
  - 154  zebras           : multi-instance discrimination at scales
  - 597  elephants        : multi-instance class-level grouping
  - 595  TV in river      : small foreground isolation
  - 785  skier            : within-object semantic parts

Run on the training server (1 GPU, ~30s):
    python tools/make_rpsa_paper_figure.py \\
        --ckpt /root/autodl-tmp/model_final.pth \\
        --image-ids 154 597 595 785 \\
        --captions \\
            "(a) Multi-instance, scale" \\
            "(b) Class-level grouping" \\
            "(c) Small-foreground" \\
            "(d) Within-object parts" \\
        --output /root/autodl-tmp/figs/rpsa_clusters.pdf
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch


def install_capture_hook(model, capture):
    transformer = model.transformer
    original = transformer.gen_encoder_output_proposals

    def patched(memory, mask, spatial_shapes):
        output_memory, output_proposals = original(memory, mask, spatial_shapes)
        capture["output_memory"] = output_memory.detach()
        capture["spatial_shapes"] = spatial_shapes.detach().cpu()
        return output_memory, output_proposals

    transformer.gen_encoder_output_proposals = patched


def slice_levels(tokens, spatial_shapes):
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


def overlay_clusters(img_rgb, cluster_grid, K, alpha, palette):
    from PIL import Image

    H, W = img_rgb.shape[:2]
    cluster_rgb = palette[cluster_grid]  # H_l x W_l x 3
    cluster_rgb_resized = np.array(
        Image.fromarray(cluster_rgb).resize((W, H), Image.NEAREST)
    )
    out = (
        (1.0 - alpha) * img_rgb.astype(np.float32)
        + alpha * cluster_rgb_resized.astype(np.float32)
    )
    return out.clip(0, 255).astype(np.uint8)


def letterbox(img_rgb, target_aspect, pad_value=255):
    """Pad img_rgb to target_aspect (W/H) with `pad_value`.
    Preserves all content; never crops. Used to make panels uniform size
    in the figure while keeping every pixel of the original image."""
    H, W = img_rgb.shape[:2]
    cur = W / H
    if abs(cur - target_aspect) < 1e-3:
        return img_rgb
    if cur > target_aspect:
        new_h = int(round(W / target_aspect))
        pad_total = new_h - H
        top = pad_total // 2
        bot = pad_total - top
        out = np.full((new_h, W, 3), pad_value, dtype=img_rgb.dtype)
        out[top:top + H] = img_rgb
        return out
    else:
        new_w = int(round(H * target_aspect))
        pad_total = new_w - W
        left = pad_total // 2
        right = pad_total - left
        out = np.full((H, new_w, 3), pad_value, dtype=img_rgb.dtype)
        out[:, left:left + W] = img_rgb
        return out


def clamp_aspect(img_rgb, min_aspect, max_aspect, pad_value=255):
    """Letterbox `img_rgb` only if its W/H aspect falls outside
    [min_aspect, max_aspect]; otherwise return it unchanged. This brings
    extreme panels (e.g. very wide panoramas or very tall portraits)
    closer to the others while leaving moderate panels untouched."""
    H, W = img_rgb.shape[:2]
    cur = W / H
    if cur < min_aspect:
        return letterbox(img_rgb, min_aspect, pad_value)
    if cur > max_aspect:
        return letterbox(img_rgb, max_aspect, pad_value)
    return img_rgb


def center_crop_to_aspect(img_rgb, target_aspect):
    """Center-crop img_rgb to target W/H aspect (loses content but produces
    uniform panel shapes with no padding). Use when uniform layout
    matters more than content preservation."""
    H, W = img_rgb.shape[:2]
    cur = W / H
    if abs(cur - target_aspect) < 1e-3:
        return img_rgb
    if cur > target_aspect:
        new_w = int(round(H * target_aspect))
        x0 = (W - new_w) // 2
        return img_rgb[:, x0:x0 + new_w]
    new_h = int(round(W / target_aspect))
    y0 = (H - new_h) // 2
    return img_rgb[y0:y0 + new_h]


def center_crop_grid(grid, target_aspect):
    """Same center-crop as center_crop_to_aspect but for the cluster grid."""
    H, W = grid.shape[:2]
    cur = W / H
    if abs(cur - target_aspect) < 1e-3:
        return grid
    if cur > target_aspect:
        new_w = max(1, int(round(H * target_aspect)))
        x0 = (W - new_w) // 2
        return grid[:, x0:x0 + new_w]
    new_h = max(1, int(round(W / target_aspect)))
    y0 = (H - new_h) // 2
    return grid[y0:y0 + new_h]


def reconstruct_image(batched_input):
    img = batched_input["image"]
    if isinstance(img, torch.Tensor):
        arr = img.permute(1, 2, 0).cpu().numpy()
        if arr.dtype != np.uint8:
            arr = arr.clip(0, 255).astype(np.uint8)
        return arr
    return np.asarray(img)


def compose_figure(panels, captions, K, alpha, out_pdf,
                   target_aspect, panel_height_in, hspace, vspace,
                   max_fig_width_in=None,
                   min_aspect=None, max_aspect=None,
                   crop_to_aspect=None):
    """Compose 2 x N grid (input row, overlay row) and save as vector PDF.

    Each panel uses its native aspect ratio (no cropping, no padding); all
    panels share the same height so the two rows align vertically. If
    target_aspect is provided (>0), images that differ from it are
    letterboxed instead, producing perfectly uniform panel widths at the
    cost of some white space.

    If `max_fig_width_in` is given, panel_height_in is reduced so the total
    figure width fits within that limit (useful for NeurIPS single-column
    where text is ~5.5 inches wide).
    """
    import matplotlib.pyplot as plt

    n = len(panels)
    cmap = plt.cm.get_cmap("tab10", K)
    palette = (np.array([cmap(i)[:3] for i in range(K)]) * 255).astype(np.uint8)

    # Compute (potentially padded / cropped) image arrays + each panel's W/H.
    rendered = []
    for img_rgb, cluster_grid in panels:
        if crop_to_aspect is not None and crop_to_aspect > 0:
            # Crop image AND cluster grid to the same target aspect, then
            # compute the overlay so cluster colors align with cropped pixels.
            img_rgb = center_crop_to_aspect(img_rgb, crop_to_aspect)
            cluster_grid = center_crop_grid(cluster_grid, crop_to_aspect)
            ov = overlay_clusters(img_rgb, cluster_grid, K, alpha, palette)
        else:
            ov = overlay_clusters(img_rgb, cluster_grid, K, alpha, palette)
            if target_aspect and target_aspect > 0:
                img_rgb = letterbox(img_rgb, target_aspect)
                ov = letterbox(ov, target_aspect)
            elif min_aspect is not None or max_aspect is not None:
                mn = min_aspect if min_aspect is not None else -float("inf")
                mx = max_aspect if max_aspect is not None else float("inf")
                img_rgb = clamp_aspect(img_rgb, mn, mx)
                ov = clamp_aspect(ov, mn, mx)
        H, W = img_rgb.shape[:2]
        rendered.append((img_rgb, ov, W / H))

    aspects = [a for _, _, a in rendered]
    panel_widths_in = [panel_height_in * a for a in aspects]
    fig_w = sum(panel_widths_in) + (n - 1) * hspace
    fig_h = 2 * panel_height_in + vspace + 0.25

    if max_fig_width_in is not None and fig_w > max_fig_width_in:
        scale = max_fig_width_in / fig_w
        panel_height_in *= scale
        panel_widths_in = [w * scale for w in panel_widths_in]
        fig_w = sum(panel_widths_in) + (n - 1) * hspace
        fig_h = 2 * panel_height_in + vspace + 0.25

    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = fig.add_gridspec(
        2, n,
        width_ratios=panel_widths_in,
        wspace=hspace / max(np.mean(panel_widths_in), 1e-6),
        hspace=vspace / max(panel_height_in, 1e-6),
    )

    for col, (img_rgb, ov, _) in enumerate(rendered):
        ax_top = fig.add_subplot(gs[0, col])
        ax_top.imshow(img_rgb)
        ax_top.axis("off")

        ax_bot = fig.add_subplot(gs[1, col])
        ax_bot.imshow(ov)
        ax_bot.axis("off")

        cap = captions[col] if col < len(captions) else ""
        ax_bot.text(
            0.5, -0.04, cap, transform=ax_bot.transAxes,
            ha="center", va="top", fontsize=7,
            linespacing=1.05,
        )

    fig.savefig(out_pdf, bbox_inches="tight", dpi=300)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config",
                   default="lami_dino/configs/dino_convnext_large_4scale_12ep_lvis.py")
    p.add_argument("--ckpt", default="/root/autodl-tmp/model_final.pth")
    p.add_argument("--image-ids", type=int, nargs="+",
                   default=[154, 597, 595, 785])
    p.add_argument("--captions", type=str, nargs="+", default=[
        "(a) multi-instance, scale",
        "(b) class-level grouping",
        "(c) small foreground",
        "(d) within-object parts",
    ])
    p.add_argument("--feature-level", type=int, default=0)
    p.add_argument("--K", type=int, default=8)
    p.add_argument("--sigma", type=float, default=1.0)
    p.add_argument("--em-iters", type=int, default=5)
    p.add_argument("--alpha", type=float, default=0.55)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--target-aspect", type=float, default=0.0,
                   help="W/H aspect ratio to letterbox each panel to. "
                        "0 (default) = use native aspect per panel (no padding).")
    p.add_argument("--min-aspect", type=float, default=1.0,
                   help="if --target-aspect=0, panels narrower than this "
                        "(e.g. portrait images) are letterboxed to this aspect. "
                        "Default 1.0 brings tall portraits to square.")
    p.add_argument("--max-aspect", type=float, default=2.0,
                   help="if --target-aspect=0, panels wider than this (e.g. "
                        "panoramas) are letterboxed to this aspect. "
                        "Default 2.0 caps very wide images.")
    p.add_argument("--crop-to-aspect", type=float, default=0.0,
                   help="if >0, center-CROP every panel to this W/H aspect "
                        "(loses content at the edges, but produces uniform "
                        "panel widths with no padding). Overrides "
                        "--target-aspect / --min-aspect / --max-aspect.")
    p.add_argument("--panel-height-in", type=float, default=1.30)
    p.add_argument("--max-fig-width-in", type=float, default=5.5,
                   help="cap total figure width (NeurIPS single-column ~5.5in)")
    p.add_argument("--hspace", type=float, default=0.06)
    p.add_argument("--vspace", type=float, default=0.05)
    p.add_argument("--output", default="figs/rpsa_clusters.pdf")
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

    dataset_dicts = get_detection_dataset_dicts(
        names="lvis_v1_val", filter_empty=False
    )
    wanted = set(args.image_ids)
    def _img_id(d):
        try:
            return int(Path(d["file_name"]).stem)
        except ValueError:
            return None
    selected = [d for d in dataset_dicts if _img_id(d) in wanted]
    order = {iid: i for i, iid in enumerate(args.image_ids)}
    selected.sort(key=lambda d: order.get(_img_id(d), 1e9))
    if len(selected) != len(wanted):
        missing = wanted - {_img_id(d) for d in selected}
        sys.exit(f"[!] missing image_ids in lvis_v1_val: {sorted(missing)}")

    loader = build_detection_test_loader(
        dataset=selected,
        mapper=instantiate(cfg.dataloader.test.mapper),
        num_workers=0,
    )

    capture = {}
    install_capture_hook(model, capture)

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    print(f"[run] inference on {len(selected)} images "
          f"(level={args.feature_level}, K={args.K}, em_iters={args.em_iters})")
    panels = []
    with torch.no_grad():
        for i, batched_inputs in enumerate(loader):
            _ = model(batched_inputs)
            output_memory = capture.get("output_memory")
            spatial_shapes = capture.get("spatial_shapes")
            if output_memory is None:
                sys.exit(f"[!] image {i}: capture missing")

            r, _ = soft_kmeans_assign(
                output_memory, K=args.K,
                sigma=args.sigma, iters=args.em_iters,
            )
            r_cpu = r.detach().cpu()
            level_tensors = slice_levels(r_cpu, spatial_shapes)
            lvl = min(args.feature_level, len(level_tensors) - 1)
            cluster_grid = level_tensors[lvl][0].argmax(dim=-1).numpy()

            img_rgb = reconstruct_image(batched_inputs[0])
            panels.append((img_rgb, cluster_grid))
            file_label = Path(batched_inputs[0]["file_name"]).stem
            print(f"  {i+1}/{len(selected)}  id={file_label}  "
                  f"img={img_rgb.shape[:2]}  cluster={cluster_grid.shape}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    compose_figure(
        panels, args.captions, args.K, args.alpha, out_path,
        target_aspect=args.target_aspect,
        panel_height_in=args.panel_height_in,
        hspace=args.hspace, vspace=args.vspace,
        max_fig_width_in=args.max_fig_width_in,
        min_aspect=args.min_aspect, max_aspect=args.max_aspect,
        crop_to_aspect=args.crop_to_aspect if args.crop_to_aspect > 0 else None,
    )
    print(f"[save] {out_path}")


if __name__ == "__main__":
    main()
