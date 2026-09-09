#!/usr/bin/env python
"""Cache a compact, exact candidate pool for offline detector/CLIP fusion.

The dense tensor has roughly 900 x 1203 pairs per LVIS image. Saving detector
logits and ROI features densely would consume tens of GB. Instead this script
stores the union of the top-300 query/class pairs for a declared set of fusion
profiles. Offline evaluation is exact for every profile listed in manifest.json
and normally occupies only a few hundred MB for the full validation set.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from lami_dino.diagnostic_ops import (  # noqa: E402
    prototype_variant_logits,
    sparse_fusion_candidate_pairs,
)


def build_profiles(model):
    novel_scale = float(model.novel_scale)
    profiles = [
        {
            "name": "current_power",
            "detector_source": "logmeanexp",
            "fusion": "power",
            "base_weight": float(model.alpha),
            "novel_weight": float(model.beta),
            "novel_scale": novel_scale,
        },
        {
            "name": "prototype_mean_power",
            "detector_source": "prototype_mean",
            "fusion": "power",
            "base_weight": float(model.alpha),
            "novel_weight": float(model.beta),
            "novel_scale": novel_scale,
        },
        {
            "name": "prompt_mean_power",
            "detector_source": "prompt_mean",
            "fusion": "power",
            "base_weight": float(model.alpha),
            "novel_weight": float(model.beta),
            "novel_scale": novel_scale,
        },
        {
            "name": "detector_only",
            "detector_source": "logmeanexp",
            "fusion": "power",
            "base_weight": 0.0,
            "novel_weight": 0.0,
            "novel_scale": 1.0,
        },
        {
            "name": "detector_scaled",
            "detector_source": "logmeanexp",
            "fusion": "power",
            "base_weight": 0.0,
            "novel_weight": 0.0,
            "novel_scale": novel_scale,
        },
        {
            "name": "vlm_only",
            "detector_source": "logmeanexp",
            "fusion": "power",
            "base_weight": 1.0,
            "novel_weight": 1.0,
            "novel_scale": 1.0,
        },
        {
            "name": "vlm_scaled",
            "detector_source": "logmeanexp",
            "fusion": "power",
            "base_weight": 1.0,
            "novel_weight": 1.0,
            "novel_scale": novel_scale,
        },
    ]
    for weight in (0.1, 0.3, 0.5, 1.0):
        profiles.append(
            {
                "name": f"logprob_novel_{weight:g}",
                "detector_source": "logmeanexp",
                "fusion": "logprob_add",
                "base_weight": 0.0,
                "novel_weight": weight,
                "novel_scale": novel_scale,
            }
        )
    return profiles


def novel_only_component_pairs(detector_logits, vlm_logits, novel_mask, topk):
    """Top novel pairs for each isolated branch used by rescue gates.

    Full-profile top-k pools may be occupied by base classes.  Explicitly
    caching the novel-only detector and CLIP pools makes later gates that leave
    base scores unchanged exact rather than approximate.
    """
    if detector_logits.shape != vlm_logits.shape or detector_logits.ndim != 2:
        raise ValueError("detector and VLM logits must share shape [Q,C]")
    if novel_mask.ndim != 1 or novel_mask.numel() != detector_logits.shape[-1]:
        raise ValueError("novel_mask must match the class dimension")
    novel_count = int(novel_mask.sum())
    if novel_count == 0:
        raise ValueError("novel-only candidate pools require at least one novel class")
    count = min(int(topk), detector_logits.shape[0] * novel_count)
    if count < 1:
        raise ValueError("topk must be positive")

    base_mask = ~novel_mask[None, :]
    branch_scores = {
        "detector_scaled_novel_only": F.logsigmoid(detector_logits).masked_fill(
            base_mask, -torch.inf
        ),
        "vlm_scaled_novel_only": F.log_softmax(vlm_logits, dim=-1).masked_fill(
            base_mask, -torch.inf
        ),
    }
    num_classes = detector_logits.shape[-1]
    result = {}
    for name, scores in branch_scores.items():
        flat_ids = scores.reshape(-1).topk(count).indices
        result[name] = (
            torch.div(flat_ids, num_classes, rounding_mode="floor"),
            flat_ids % num_classes,
        )
    return result


def install_capture_hooks(model, capture):
    final_index = model.transformer.decoder.num_layers - 1
    classifier = model.class_embed[final_index]
    original_logits = classifier._compute_tpa_logits

    def capture_logits(x, *, content_inds, additional_class):
        result = original_logits(
            x,
            content_inds=content_inds,
            additional_class=additional_class,
        )
        capture["detector_logits"] = result.detach()
        prototypes = classifier._external_prototypes
        if prototypes is None:
            prototypes = classifier._cached_eval
        if prototypes is None:
            raise RuntimeError("TPA prototypes were unavailable in final classifier")
        capture["projected_features"] = x.detach()
        capture["prototypes"] = prototypes.detach()
        capture["prompt_features"] = classifier.eval_text_feats.detach()
        return result

    classifier._compute_tpa_logits = capture_logits

    original_extract = model.extract_region_feature

    def capture_region_feature(features, bbox, layer_name):
        result = original_extract(features, bbox, layer_name)
        if layer_name == "p3":
            capture["roi_features"] = result.detach()
        return result

    model.extract_region_feature = capture_region_feature

    original_inference = model.inference

    def capture_inference(box_cls, box_pred, image_sizes, wo_sigmoid=False):
        capture["query_boxes"] = box_pred.detach()
        return original_inference(
            box_cls,
            box_pred,
            image_sizes,
            wo_sigmoid=wo_sigmoid,
        )

    model.inference = capture_inference
    return classifier


def select_dataset_records(records, num_images, seed):
    if num_images <= 0 or num_images >= len(records):
        return records
    generator = np.random.default_rng(seed)
    indices = np.sort(generator.choice(len(records), size=num_images, replace=False))
    return [records[int(index)] for index in indices]


def dump_worker(args):
    from detectron2.checkpoint import DetectionCheckpointer
    from detectron2.config import LazyConfig, instantiate
    from detectron2.data import MetadataCatalog, build_detection_test_loader
    from detectron2.data import get_detection_dataset_dicts
    from detectron2.utils import comm

    cfg = LazyConfig.load(args.config_file)
    cfg = LazyConfig.apply_overrides(cfg, args.opts)
    dataset_name = cfg.dataloader.test.dataset.names
    if not isinstance(dataset_name, str):
        if len(dataset_name) != 1:
            raise ValueError("raw-score dump requires exactly one test dataset")
        dataset_name = dataset_name[0]

    device = torch.device("cuda", comm.get_local_rank())
    torch.cuda.set_device(device)
    model = instantiate(cfg.model).to(device).eval()
    if not model.score_ensemble:
        raise ValueError("raw score fusion requires model.score_ensemble=True")
    DetectionCheckpointer(model).load(args.checkpoint)
    profiles = build_profiles(model)
    novel_mask = model.novel_idx.to(device=device)

    records = get_detection_dataset_dicts(names=dataset_name, filter_empty=False)
    records = select_dataset_records(records, args.num_images, args.seed)
    loader = build_detection_test_loader(
        dataset=records,
        mapper=instantiate(cfg.dataloader.test.mapper),
        num_workers=args.num_workers,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if not args.resume:
        existing = next(output_dir.glob("raw_*.pth"), None)
        if existing is not None:
            raise FileExistsError(
                f"{output_dir} already contains raw dumps (for example {existing.name}); "
                "use --resume or choose a new directory"
            )
    comm.synchronize()

    capture = {}
    classifier = install_capture_hooks(model, capture)
    written = 0
    skipped = 0
    with torch.no_grad():
        for batch_index, batched_inputs in enumerate(loader):
            output_paths = [
                output_dir / f"raw_{int(item['image_id'])}.pth"
                for item in batched_inputs
            ]
            if args.resume and all(path.exists() for path in output_paths):
                skipped += len(output_paths)
                continue

            capture.clear()
            _ = model(batched_inputs)
            missing = {
                "detector_logits",
                "projected_features",
                "prototypes",
                "prompt_features",
                "roi_features",
                "query_boxes",
            } - set(capture)
            if missing:
                raise RuntimeError(f"capture hooks missed: {sorted(missing)}")
            detector_logits = capture["detector_logits"].float()
            feature_shape = capture["projected_features"].shape
            detector_variants = prototype_variant_logits(
                capture["projected_features"].reshape(-1, feature_shape[-1]),
                capture["prototypes"],
                capture["prompt_features"],
                temperature=float(classifier.tpa_cls_tau),
                logit_scale=float(classifier.norm_temperature),
            )
            detector_variants = {
                name: logits.reshape(*feature_shape[:-1], logits.shape[-1])
                for name, logits in detector_variants.items()
            }
            if classifier.use_bias:
                detector_variants = {
                    name: logits + classifier.cls_bias
                    for name, logits in detector_variants.items()
                }
            recompute_error = (
                detector_variants["logmeanexp"] - detector_logits
            ).abs().max()
            if float(recompute_error) > 5e-4:
                raise RuntimeError(
                    "recomputed final TPA logits do not match the classifier: "
                    f"max_abs_error={float(recompute_error)}"
                )
            detector_variants["logmeanexp"] = detector_logits
            roi_features = capture["roi_features"].float()
            query_boxes = capture["query_boxes"].float()
            vlm_logits = (
                roi_features @ model.vlm_content_query_embedding.t()
            ) * float(model.vlm_temperature)

            for local_index, (model_input, output_path) in enumerate(
                zip(batched_inputs, output_paths)
            ):
                if args.resume and output_path.exists():
                    skipped += 1
                    continue
                candidate_flat_ids = []
                num_classes = detector_logits.shape[-1]
                for profile in profiles:
                    source = profile["detector_source"]
                    profile_query_ids, profile_class_ids = (
                        sparse_fusion_candidate_pairs(
                            detector_variants[source][local_index],
                            vlm_logits[local_index],
                            novel_mask,
                            topk=args.topk,
                            profiles=[profile],
                        )
                    )
                    candidate_flat_ids.append(
                        profile_query_ids * num_classes + profile_class_ids
                    )
                novel_component_pairs = novel_only_component_pairs(
                    detector_variants["logmeanexp"][local_index],
                    vlm_logits[local_index],
                    novel_mask,
                    args.topk,
                )
                for profile_query_ids, profile_class_ids in (
                    novel_component_pairs.values()
                ):
                    candidate_flat_ids.append(
                        profile_query_ids * num_classes + profile_class_ids
                    )
                flat_ids = torch.unique(torch.cat(candidate_flat_ids), sorted=True)
                query_ids = torch.div(flat_ids, num_classes, rounding_mode="floor")
                class_ids = flat_ids % num_classes
                detector_top_probabilities, detector_top_classes = (
                    detector_variants["logmeanexp"][local_index]
                    .sigmoid()
                    .topk(2, dim=-1)
                )
                vlm_log_normalizer = torch.logsumexp(
                    vlm_logits[local_index], dim=-1
                )
                vlm_top_logits, vlm_top_classes = vlm_logits[local_index].topk(
                    2, dim=-1
                )
                vlm_top_probabilities = (
                    vlm_top_logits - vlm_log_normalizer[:, None]
                ).exp()
                payload = {
                    "schema_version": 1,
                    "image_id": int(model_input["image_id"]),
                    "height": int(model_input.get("height", model_input["image"].shape[-2])),
                    "width": int(model_input.get("width", model_input["image"].shape[-1])),
                    "query_boxes": query_boxes[local_index].cpu(),
                    "candidate_query_ids": query_ids.to(torch.int16).cpu(),
                    "candidate_class_ids": class_ids.to(torch.int16).cpu(),
                    "detector_logits_by_source": {
                        name: logits[local_index, query_ids, class_ids].cpu()
                        for name, logits in detector_variants.items()
                    },
                    "vlm_logits": vlm_logits[local_index, query_ids, class_ids].cpu(),
                    "vlm_log_normalizer": vlm_log_normalizer.cpu(),
                    "component_query_summary": {
                        "detector_top_probabilities": detector_top_probabilities.cpu(),
                        "detector_top_classes": detector_top_classes.to(torch.int16).cpu(),
                        "vlm_top_probabilities": vlm_top_probabilities.cpu(),
                        "vlm_top_classes": vlm_top_classes.to(torch.int16).cpu(),
                    },
                }
                temporary = output_path.with_suffix(".tmp")
                torch.save(payload, temporary)
                os.replace(temporary, output_path)
                written += 1
            if (batch_index + 1) % 100 == 0:
                print(
                    f"[rank {comm.get_rank()}] batches={batch_index + 1} "
                    f"written={written} skipped={skipped}",
                    flush=True,
                )

    counts = comm.gather({"written": written, "skipped": skipped}, dst=0)
    comm.synchronize()
    if comm.is_main_process():
        metadata = MetadataCatalog.get(dataset_name)
        manifest = {
            "schema_version": 1,
            "config_file": args.config_file,
            "checkpoint": args.checkpoint,
            "dataset_name": dataset_name,
            "lvis_json": os.path.abspath(metadata.json_file),
            "num_dataset_images": len(records),
            "topk_per_profile": args.topk,
            "profiles": profiles,
            "candidate_pool_extensions": [
                "detector_scaled_novel_only",
                "vlm_scaled_novel_only",
            ],
            "novel_class_ids": torch.nonzero(model.novel_idx, as_tuple=False)
            .flatten()
            .tolist(),
            "workers": counts,
        }
        manifest_path = output_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        total_written = sum(item["written"] for item in counts)
        total_skipped = sum(item["skipped"] for item in counts)
        print(f"[done] written={total_written}, skipped={total_skipped}")
        print(f"[save] {manifest_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-file",
        default="lami_dino/configs/dino_convnext_large_4scale_12ep_lvis.py",
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--num-images", type=int, default=0, help="0 means all images")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--topk", type=int, default=300)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("opts", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    from detectron2.engine import launch

    launch(
        dump_worker,
        args.num_gpus,
        num_machines=1,
        machine_rank=0,
        dist_url="auto",
        args=(args,),
    )


if __name__ == "__main__":
    main()
