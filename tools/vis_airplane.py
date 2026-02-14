#!/usr/bin/env python3
import argparse
import os
from glob import glob

import torch

from detectron2.config import LazyConfig, instantiate
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import MetadataCatalog
from detectron2.data import transforms as T
from detectron2.data.detection_utils import read_image
from detectron2.utils.visualizer import ColorMode, Visualizer


def collect_images(path):
    if os.path.isdir(path):
        exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
        files = []
        for e in exts:
            files.extend(glob(os.path.join(path, e)))
        return sorted(files)
    return [path]


def main():
    ap = argparse.ArgumentParser(description="Visualize airplane images with a DINO model.")
    ap.add_argument("--config", required=True, help="LazyConfig file.")
    ap.add_argument("--weights", required=True, help="Model .pth path.")
    ap.add_argument("--input", required=True, help="Image file or directory.")
    ap.add_argument("--output", required=True, help="Output directory.")
    ap.add_argument("--score-thresh", type=float, default=None, help="Override score threshold.")
    ap.add_argument("--class-name", default=None, help="Only visualize a specific class name.")
    ap.add_argument("--topk", type=int, default=0, help="Keep top-K boxes by score (0 = no limit).")
    ap.add_argument("--debug", action="store_true", help="Print debug stats per image.")
    ap.add_argument("--device", default="cuda", help="cuda or cpu.")
    ap.add_argument("--limit", type=int, default=0, help="Limit number of images (0 = all).")
    args = ap.parse_args()

    cfg = LazyConfig.load(args.config)
    cfg.train.init_checkpoint = args.weights
    model = instantiate(cfg.model)
    model.to(args.device)
    model.eval()
    DetectionCheckpointer(model).load(args.weights)

    dataset_name = cfg.dataloader.test.dataset.names
    if isinstance(dataset_name, (list, tuple)):
        dataset_name = dataset_name[0]
    metadata = MetadataCatalog.get(dataset_name) if dataset_name else None

    os.makedirs(args.output, exist_ok=True)
    files = collect_images(args.input)
    if args.limit > 0:
        files = files[: args.limit]

    score_thresh = args.score_thresh
    if score_thresh is None:
        score_thresh = float(getattr(cfg.model, "test_score_thresh", 0.0))

    class_id = None
    if args.class_name and metadata and hasattr(metadata, "thing_classes"):
        if args.class_name in metadata.thing_classes:
            class_id = metadata.thing_classes.index(args.class_name)
        else:
            raise ValueError(f"class-name '{args.class_name}' not in metadata.thing_classes")

    # Create the same augmentation pipeline as evaluation
    # This ensures consistency with the evaluation results
    aug = T.ResizeShortestEdge(short_edge_length=800, max_size=1333)
    
    for path in files:
        # Read image in RGB format (same as evaluation pipeline)
        img_original = read_image(path, format="RGB")
        original_height, original_width = img_original.shape[:2]
        
        # Apply the same resize transformation as evaluation
        aug_input = T.AugInput(img_original)
        transforms = aug(aug_input)
        img_resized = aug_input.image
        
        # Convert to tensor for model input
        image = torch.as_tensor(img_resized.astype("float32")).permute(2, 0, 1).to(args.device)
        inputs = [{"image": image, "height": original_height, "width": original_width}]
        
        with torch.no_grad():
            outputs = model(inputs)[0]["instances"].to("cpu")
        if args.debug:
            num_all = len(outputs)
            topk_scores = outputs.scores.sort(descending=True).values[:10].tolist() if num_all else []
            topk_classes = outputs.pred_classes[:10].tolist() if num_all else []
            print(f"[debug] {os.path.basename(path)}: num={num_all} top_scores={topk_scores} top_classes={topk_classes}")
        if score_thresh > 0:
            outputs = outputs[outputs.scores > score_thresh]
        if class_id is not None:
            outputs = outputs[outputs.pred_classes == class_id]
        if args.topk and len(outputs) > args.topk:
            idx = outputs.scores.sort(descending=True).indices[: args.topk]
            outputs = outputs[idx]

        # Use original image for visualization (RGB to BGR for cv2)
        vis = Visualizer(
            img_original[:, :, ::-1],
            metadata=metadata,
            scale=1.0,
            instance_mode=ColorMode.IMAGE_BW,
        )
        vis = vis.draw_instance_predictions(outputs)
        out_path = os.path.join(args.output, os.path.basename(path))
        vis.save(out_path)
        print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
