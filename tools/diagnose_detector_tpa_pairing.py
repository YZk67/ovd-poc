#!/usr/bin/env python
"""Paired final-classifier replay; never trains or changes model parameters.

Each checkpoint runs its own native query fusion and box prediction. Only the
last prototype bank is exchanged offline. Cross-checkpoint query indices are
never paired. The panel includes every rare-GT image plus seeded, verified
rare-negative controls. Panel PR/coverage is NOT full-validation LVIS AP.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, median

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from lami_dino.checkpoint_init import load_trusted_torch_file
from lami_dino.diagnostic_ops import fuse_detector_vlm_scores
from lami_dino.pairing_diagnostic_ops import analyze_image, replay_classifier
from tools.diagnose_tpa_usage import box_cxcywh_to_xyxy, install_capture_hooks
from tools.pairing_lvis_support import evaluate_panel_predictions, select_panel


VARIANTS = {"old": ("old_old", "old_new"), "new": ("new_new", "new_old", "new_mean")}
SCHEMA_VERSION = 1


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_digest(tensor):
    value = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256(str((tuple(value.shape), str(value.dtype))).encode())
    digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def save_tensor_file(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(data, temporary)
    temporary.replace(path)


def locked_manifest(path, inputs):
    fingerprint = hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest()
    expected = {"fingerprint": fingerprint, "inputs": inputs}
    path = Path(path)
    if path.exists():
        found = json.loads(path.read_text())
        if found != expected:
            raise ValueError("Cache manifest differs from this run. Use a new --output directory; do not mix checkpoints/protocols.")
    else:
        write_json(path, expected)
    return fingerprint


def classifier_options(args, label):
    # Buffers in new checkpoints restore their learned/recorded strengths.
    # Missing legacy buffers must stay at zero, not inherit the new structure.
    return [
        f"model.alpha={args.alpha}", f"model.beta={args.beta}",
        f"model.novel_scale={args.novel_scale}",
        f"model.classifier.tpa_tau={args.tpa_tau}",
        f"model.classifier.tpa_cls_tau={args.cls_tau}",
        "model.classifier.tpa_eval_legacy_logsumexp=False",
        "model.classifier.tpa_eval_logit_bias=0.0",
        "model.tpa_eval_mode_scale=1.0", "model.soft_category_topk=3",
        "model.inference_query_class_topk=0",
        f"model.select_box_nums_for_evaluation={args.max_dets}",
        f"model.classifier.tpa_slot_prior_strength={0.0 if label == 'old' else 0.2}",
        f"model.classifier.tpa_prototype_mode_strength={0.0 if label == 'old' else 1.5}",
        f"train.device={args.device}", f"model.device={args.device}",
    ]


def input_asset_hashes(args, cache_dir):
    """External prompt banks are not checkpoint buffers: lock their bytes too."""
    if args.phase == "analyze":
        previous = json.loads((cache_dir / "manifest.json").read_text())
        paths = list(previous["inputs"]["asset_sha256"])
    else:
        from detectron2.config import LazyConfig
        cfg = LazyConfig.load(args.config_file)
        paths = []
        for name in ("query_path", "eval_query_path", "vlm_query_path", "seen_classes", "all_classes", "clip_head_path"):
            value = cfg.model.get(name)
            if value:
                paths.append(str(Path(value).resolve()))
        classifier = cfg.model.classifier
        train_text = classifier.get("text_embed_path") or classifier.get("zs_weight_path")
        eval_text = classifier.get("eval_text_embed_path") or classifier.get("eval_zs_weight_path") or train_text
        for value in (train_text, eval_text):
            if value and value != "rand":
                paths.append(str(Path(value).resolve()))
    return {path: sha256_file(path) for path in sorted(set(paths))}


def validate_parameter_state(model, state):
    parameters = dict(model.named_parameters())
    missing = sorted(set(parameters) - state.keys())
    if missing:
        raise ValueError(f"Checkpoint misses model parameters (refusing random weights): {missing[:20]}")
    wrong = [key for key, expected in model.state_dict().items() if key in state
             and (not torch.is_tensor(state[key]) or tuple(state[key].shape) != tuple(expected.shape))]
    if wrong:
        raise ValueError(f"Checkpoint tensor shapes/types disagree (loader must not skip weights): {wrong[:20]}")


def validate_existing_bank(path, bank):
    if not Path(path).exists():
        return
    previous = load_trusted_torch_file(path)
    if set(previous) != set(bank):
        raise ValueError("Partial-cache bank schema changed; use a new output directory")
    for key, value in bank.items():
        equal = torch.equal(previous[key], value) if torch.is_tensor(value) else previous[key] == value
        if not equal:
            raise ValueError(f"Partial-cache bank changed: {key}; refusing to mix existing image features")


def check_replay(native_logits, native_scores, replay_logits, replay_scores, *, max_dets, logit_atol=1e-4, score_atol=5e-6):
    for tensor in (native_logits, native_scores, replay_logits, replay_scores):
        if not torch.isfinite(tensor).all():
            raise ValueError("Non-finite native/replayed scores")
    if native_logits.shape != replay_logits.shape or native_scores.shape != replay_scores.shape:
        raise ValueError("Native/replayed shapes differ")
    logit_error = float((native_logits - replay_logits).abs().max())
    score_error = float((native_scores - replay_scores).abs().max())
    if logit_error > logit_atol or score_error > score_atol:
        raise ValueError(f"Native replay failed: logit error={logit_error:g}, score error={score_error:g}")
    k = min(max_dets, native_scores.numel())
    native_ids = native_scores.flatten().topk(k).indices
    replay_ids = replay_scores.flatten().topk(k).indices
    differing = set(native_ids.tolist()) ^ set(replay_ids.tolist())
    threshold = float(native_scores.flatten()[native_ids[-1]]) if k else None
    if differing:
        boundary_error = max(abs(float(native_scores.flatten()[index]) - threshold) for index in differing)
        if boundary_error > 2 * score_atol:
            raise ValueError("Native top-k membership differs beyond floating-point boundary ties")
    return {"logit_max_abs_error": logit_error, "score_max_abs_error": score_error,
            "topk_symmetric_difference": len(differing), "boundary_tie_tolerance": 2 * score_atol}


def to_predictions(top_predictions, image_id, category_ids):
    result = []
    for item in top_predictions:
        x0, y0, x1, y1 = item["box_xyxy"]
        # Same ordering as detector_postprocess: top-k first, then remove empty
        # clipped boxes. Never refill top-k or filter rare classes beforehand.
        if x1 <= x0 or y1 <= y0:
            continue
        result.append({"image_id": int(image_id), "category_id": category_ids[item["class_index"]],
                       "bbox": [x0, y0, x1 - x0, y1 - y0], "score": item["score"]})
    return result


def summarize_pair_rows(rows):
    """Equal-class paired summaries; do not call these AP or training causes."""
    groups = defaultdict(dict)
    for row in rows:
        key = (row["image_id"], row["gt_id"], row["iou_threshold"])
        if row["variant"] in groups[key]:
            raise ValueError("Duplicate paired GT/variant")
        groups[key][row["variant"]] = row
    summary = {}
    for threshold in sorted({key[2] for key in groups}):
        selected = [group for key, group in groups.items() if key[2] == threshold]
        if any(set(group) != set(sum((list(v) for v in VARIANTS.values()), [])) for group in selected):
            raise ValueError("Old/new panel GT IDs do not pair across all variants")
        intersection = [group for group in selected if group["old_old"]["eligible"] and group["new_new"]["eligible"]]
        counts = {"all_gt": len(selected), "both_eligible": len(intersection),
                  "old_only_eligible": sum(g["old_old"]["eligible"] and not g["new_new"]["eligible"] for g in selected),
                  "new_only_eligible": sum(g["new_new"]["eligible"] and not g["old_old"]["eligible"] for g in selected),
                  "neither_eligible": sum(not g["new_new"]["eligible"] and not g["old_old"]["eligible"] for g in selected)}
        variants = {}
        for variant in sum((list(v) for v in VARIANTS.values()), []):
            by_class = defaultdict(list)
            for group in intersection:
                by_class[group[variant]["class_index"]].append(group[variant])
            per_class = []
            for class_index, items in sorted(by_class.items()):
                per_class.append({
                    "class_index": class_index, "num_gt": len(items),
                    "native_anchor_top1": mean(float(r["detector_rank"] == 1) for r in items),
                    "native_anchor_top5": mean(float(r["detector_rank"] <= 5) for r in items),
                    "native_anchor_margin_mean": mean(r["margin"] for r in items),
                    "native_anchor_margin_median": median(r["margin"] for r in items),
                    "geometry_anchor_top1": mean(float(r["geometry_anchor"]["detector_rank"] == 1) for r in items),
                    "geometry_anchor_top5": mean(float(r["geometry_anchor"]["detector_rank"] <= 5) for r in items),
                    "geometry_anchor_margin_mean": mean(r["geometry_anchor"]["margin"] for r in items),
                    "pair_topk_coverage": mean(float(r["pair_topk"]) for r in items),
                    "native_anchor_threshold_ratio": mean(r["threshold_ratio"] for r in items if r["threshold_ratio"] is not None)
                    if any(r["threshold_ratio"] is not None for r in items) else None,
                    "threshold_ratio_valid_gt": sum(r["threshold_ratio"] is not None for r in items),
                })
            macro_keys = ("native_anchor_top1", "native_anchor_top5", "native_anchor_margin_mean",
                          "geometry_anchor_top1", "geometry_anchor_top5", "geometry_anchor_margin_mean", "pair_topk_coverage")
            variants[variant] = {"paired_gt": len(intersection), "paired_classes": len(per_class),
                                 "macro": {key: mean(r[key] for r in per_class) if per_class else None for key in macro_keys},
                                 "per_class": per_class}
        class_localization = {}
        for class_index in sorted({g["old_old"]["class_index"] for g in selected}):
            class_groups = [g for g in selected if g["old_old"]["class_index"] == class_index]
            class_localization[str(class_index)] = {
                "gt": len(class_groups),
                "both_eligible": sum(g["old_old"]["eligible"] and g["new_new"]["eligible"] for g in class_groups),
                "old_only_eligible": sum(g["old_old"]["eligible"] and not g["new_new"]["eligible"] for g in class_groups),
                "new_only_eligible": sum(g["new_new"]["eligible"] and not g["old_old"]["eligible"] for g in class_groups),
                "neither_eligible": sum(not g["old_old"]["eligible"] and not g["new_new"]["eligible"] for g in class_groups),
            }
        comparisons = {}
        for changed, native in (("new_old", "new_new"), ("old_new", "old_old"), ("new_mean", "new_new")):
            comparisons[f"{changed}_minus_{native}"] = {
                key: variants[changed]["macro"][key] - variants[native]["macro"][key]
                if variants[changed]["macro"][key] is not None else None
                for key in variants[changed]["macro"]
            }
        threshold_key = f"{threshold:.2f}" if threshold == round(threshold, 2) else f"{threshold:.12g}"
        summary[threshold_key] = {"localization_partition": counts, "variants": variants,
                                  "all_gt_classes": len(class_localization), "per_class_localization": class_localization,
                                  "terminal_swap_macro_deltas": comparisons}
    return summary


def dump_checkpoint(args, label, panel, dataset, fingerprint, cache_dir):
    from detectron2.checkpoint import DetectionCheckpointer
    from detectron2.config import LazyConfig, instantiate
    from detectron2.data import MetadataCatalog, build_detection_test_loader, get_detection_dataset_dicts

    branch_dir = cache_dir / label
    branch_dir.mkdir(parents=True, exist_ok=True)
    bank_path = branch_dir / "bank.pt"
    pending = [image_id for image_id in panel["image_ids"] if not (branch_dir / f"{image_id}.pt").exists()]
    if not bank_path.exists():
        pending = panel["image_ids"]
    if bank_path.exists() and not pending:
        print(f"[cache] {label}: all {len(panel['image_ids'])} images present; no forward", flush=True)
        return
    cfg = LazyConfig.load(args.config_file)
    cfg = LazyConfig.apply_overrides(cfg, classifier_options(args, label))
    augmentations = cfg.dataloader.test.mapper.augmentation
    if not (len(augmentations) == 1 and "ResizeShortestEdge" in str(augmentations[0]._target_)
            and cfg.dataloader.test.mapper.augmentation_with_crop is None):
        raise ValueError("Pairing requires the native resize-only validation mapper")
    short_edges = augmentations[0].short_edge_length
    if isinstance(short_edges, int):
        short_edges = [short_edges]
    if len(set(short_edges)) != 1 or cfg.dataloader.test.mapper.is_train:
        raise ValueError("Pairing requires a deterministic single-size eval mapper")
    dataset_name = cfg.dataloader.test.dataset.names
    if not isinstance(dataset_name, str):
        if len(dataset_name) != 1:
            raise ValueError("Exactly one validation dataset required")
        dataset_name = dataset_name[0]
    records = get_detection_dataset_dicts(names=dataset_name, filter_empty=False)
    metadata = MetadataCatalog.get(dataset_name)
    if Path(metadata.json_file).resolve() != Path(args.annotations).resolve():
        raise ValueError("Config validation annotations differ from --annotations")
    category_ids = sorted(c["id"] for c in dataset["categories"])
    expected_mapping = {cat_id: i for i, cat_id in enumerate(category_ids)}
    if getattr(metadata, "thing_dataset_id_to_contiguous_id", expected_mapping) != expected_mapping:
        raise ValueError("Unexpected LVIS contiguous category mapping")
    records_by_id = {r["image_id"]: r for r in records}
    if not set(panel["image_ids"]).issubset(records_by_id):
        raise ValueError("Some panel images are absent from configured validation dataset")
    print(f"[load] {label}: {getattr(args, label + '_checkpoint')}", flush=True)
    model = instantiate(cfg.model).to(args.device).eval()
    checkpoint_path = getattr(args, label + "_checkpoint")
    checkpoint = load_trusted_torch_file(checkpoint_path)
    state = checkpoint.get("model", checkpoint)
    state = {key[7:] if key.startswith("module.") else key: value for key, value in state.items()}
    validate_parameter_state(model, state)
    iteration = checkpoint.get("iteration")
    del checkpoint, state
    DetectionCheckpointer(model).load(checkpoint_path)
    capture = {}
    classifier = install_capture_hooks(model, capture)
    if not model.score_ensemble or not classifier.use_tpa or not classifier.norm_weight:
        raise ValueError("Requires normalized TPA classifier and CLIP score ensemble")
    if model.num_classes != len(category_ids):
        raise ValueError("Model and annotation vocabulary sizes differ")
    bank = None
    loader = build_detection_test_loader(dataset=[records_by_id[i] for i in pending],
                                        mapper=instantiate(cfg.dataloader.test.mapper), num_workers=0)
    print(f"[dump] {label}: {len(pending)} pending images; caches are float32", flush=True)
    with torch.no_grad():
        for index, inputs in enumerate(loader):
            if len(inputs) != 1:
                raise ValueError("Use one image per native forward for controlled preprocessing")
            capture.clear()
            model(inputs)
            required = {"projected_features", "prototypes", "roi_features", "query_boxes", "detector_logits", "final_scores"}
            if required - capture.keys():
                raise RuntimeError(f"Capture missed {required - capture.keys()}")
            image_id = int(inputs[0]["image_id"])
            if bank is None:
                bank = {"fingerprint": fingerprint, "label": label, "iteration": iteration,
                        "prototypes": capture["prototypes"].float().cpu(),
                        "vlm_text": model.vlm_content_query_embedding.float().cpu(),
                        "prompt_sha256": tensor_digest(classifier.eval_text_feats),
                        "novel_mask": model.novel_idx.bool().cpu(), "category_ids": category_ids,
                        "temperature": float(classifier.tpa_cls_tau), "logit_scale": float(classifier.norm_temperature),
                        "cls_bias": float(classifier.cls_bias) if classifier.use_bias else 0.0,
                        "vlm_temperature": float(model.vlm_temperature),
                        "slot_prior_strength": float(classifier.tpa.slot_prior_strength),
                        "prototype_mode_strength": float(classifier.tpa.prototype_mode_strength),
                        "tpa_tau": float(classifier.tpa.tau)}
                expected_rare_mask = torch.tensor([next(c for c in dataset["categories"] if c["id"] == cat_id)["frequency"] == "r" for cat_id in category_ids])
                if not torch.equal(bank["novel_mask"], expected_rare_mask):
                    raise ValueError("Model novel mask and LVIS rare classes disagree")
                validate_existing_bank(bank_path, bank)
                save_tensor_file(bank_path, bank)
                print(f"[bank] {label}: iteration={iteration}, slots={tuple(bank['prototypes'].shape)}, mode_strength={bank['prototype_mode_strength']}", flush=True)
            features = capture["projected_features"][0].float()
            roi_features = capture["roi_features"][0].float()
            prototypes = bank["prototypes"].to(features.device)
            replay = replay_classifier(features, prototypes, temperature=bank["temperature"],
                                       logit_scale=bank["logit_scale"], cls_bias=bank["cls_bias"], query_chunk_size=args.query_chunk_size)
            vlm_logits = roi_features @ model.vlm_content_query_embedding.float().t() * bank["vlm_temperature"]
            fused = fuse_detector_vlm_scores(replay, vlm_logits, model.novel_idx.to(features.device),
                                             base_weight=args.alpha, novel_weight=args.beta, novel_scale=args.novel_scale).exp()
            errors = check_replay(capture["detector_logits"][0].float(), capture["final_scores"][0].float(), replay, fused, max_dets=args.max_dets)
            width, height = int(inputs[0]["width"]), int(inputs[0]["height"])
            boxes = box_cxcywh_to_xyxy(capture["query_boxes"][0].float()).clamp(0, 1)
            boxes = boxes * boxes.new_tensor([width, height, width, height])
            if index == 0:
                estimate = (features.numel() + roi_features.numel()) * 4 * len(pending)
                free = shutil.disk_usage(branch_dir).free
                print(f"[disk] {label} pending cache ~{estimate / 2**30:.2f} GiB; free={free / 2**30:.2f} GiB", flush=True)
                if free < estimate + 512 * 1024**2:
                    raise OSError("Insufficient free disk for float32 cache; no existing files removed")
            save_tensor_file(branch_dir / f"{image_id}.pt", {
                "fingerprint": fingerprint, "label": label, "image_id": image_id,
                "width": width, "height": height, "features": features.cpu(),
                "roi_features": roi_features.cpu(), "query_boxes": boxes.cpu(),
                "native_replay_check": errors,
            })
            if index == 0 or (index + 1) % args.log_interval == 0 or index + 1 == len(pending):
                print(f"[dump {label}] {index + 1}/{len(pending)} image={image_id} native_errors={errors}", flush=True)
    capture.clear()
    del model, classifier, loader
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def analyze_cache(args, panel, dataset, fingerprint, cache_dir):
    banks = {label: load_trusted_torch_file(cache_dir / label / "bank.pt") for label in VARIANTS}
    for label, bank in banks.items():
        if bank["fingerprint"] != fingerprint or bank["label"] != label:
            raise ValueError("Bank/cache fingerprint mismatch")
    for key in ("category_ids", "temperature", "logit_scale", "vlm_temperature", "prompt_sha256"):
        if banks["old"][key] != banks["new"][key]:
            raise ValueError(f"Cannot cross-score: old/new {key} differ")
    for key in ("vlm_text", "novel_mask"):
        if not torch.equal(banks["old"][key], banks["new"][key]):
            raise ValueError(f"Cannot cross-score: old/new {key} differ")
    if banks["old"]["prototypes"].shape != banks["new"]["prototypes"].shape:
        raise ValueError("Expected same [classes, K, dimension] in old/new banks")
    category_ids = banks["old"]["category_ids"]
    category_index = {cat_id: i for i, cat_id in enumerate(category_ids)}
    rare_ids = set(panel["rare_category_ids"])
    annotations_by_image = defaultdict(list)
    for annotation in dataset["annotations"]:
        if annotation["category_id"] in rare_ids and not annotation.get("ignore", False):
            annotations_by_image[annotation["image_id"]].append(annotation)
    predictions = {variant: [] for names in VARIANTS.values() for variant in names}
    rows, native_checks = [], []
    device = torch.device(args.device)
    for label, variants in VARIANTS.items():
        bank = banks[label]
        text = bank["vlm_text"].to(device)
        novel_mask = bank["novel_mask"].to(device)
        native_variant = f"{label}_{label}"
        for index, image_id in enumerate(panel["image_ids"]):
            sample = load_trusted_torch_file(cache_dir / label / f"{image_id}.pt")
            if sample["fingerprint"] != fingerprint or sample["label"] != label or sample["image_id"] != image_id:
                raise ValueError("Image cache identity mismatch")
            native_checks.append(sample["native_replay_check"])
            features = sample["features"].to(device)
            vlm_logits = sample["roi_features"].to(device) @ text.t() * bank["vlm_temperature"]
            logits = {}
            for variant in variants:
                destination = variant.split("_")[1]
                prototypes = banks["new" if destination == "mean" else destination]["prototypes"]
                if destination == "mean":
                    prototypes = prototypes.mean(dim=1, keepdim=True)
                logits[variant] = replay_classifier(features, prototypes.to(device), temperature=bank["temperature"],
                                                    logit_scale=bank["logit_scale"], cls_bias=bank["cls_bias"], query_chunk_size=args.query_chunk_size)
            annotations = sorted(annotations_by_image[image_id], key=lambda ann: ann["id"])
            xyxy = [[a["bbox"][0], a["bbox"][1], a["bbox"][0] + a["bbox"][2], a["bbox"][1] + a["bbox"][3]] for a in annotations]
            result = analyze_image(torch.tensor(xyxy, dtype=torch.float32, device=device).reshape(-1, 4),
                                   torch.tensor([category_index[a["category_id"]] for a in annotations], dtype=torch.long, device=device),
                                   [a["id"] for a in annotations], sample["query_boxes"].to(device), logits, vlm_logits, novel_mask,
                                   native_variant=native_variant, alpha=args.alpha, beta=args.beta, novel_scale=args.novel_scale,
                                   max_dets=args.max_dets, iou_thresholds=args.iou_thresholds)
            rows.extend({**row, "image_id": image_id} for row in result["rows"])
            for variant, top in result["top_predictions"].items():
                predictions[variant].extend(to_predictions(top, image_id, category_ids))
            if index == 0 or (index + 1) % args.log_interval == 0 or index + 1 == len(panel["image_ids"]):
                print(f"[replay {label}] {index + 1}/{len(panel['image_ids'])}", flush=True)
    summary = summarize_pair_rows(rows)
    for threshold, result in summary.items():
        print(f"\n=== Paired classification @IoU={threshold} ===", flush=True)
        print(f"localization: {result['localization_partition']}", flush=True)
        print("variant   GT/classes  anchor-top5%  geometric-top5%  geometric-margin  pair-topk%", flush=True)
        for variant, data in result["variants"].items():
            macro = data["macro"]
            number = lambda key, scale=1: f"{scale * macro[key]:.4f}" if macro[key] is not None else "NA"
            print(f"{variant:9} {data['paired_gt']:4}/{data['paired_classes']:<3} {number('native_anchor_top5', 100):>12} "
                  f"{number('geometry_anchor_top5', 100):>15} {number('geometry_anchor_margin_mean'):>17} {number('pair_topk_coverage', 100):>11}", flush=True)
    report = {"scope": "Final-classifier compatibility on a diagnostic panel, NOT full LVIS AP or causal attribution to training changes.",
              "fingerprint": fingerprint, "panel": panel, "paired": summary, "rows": rows,
              "banks": {label: {key: bank[key] for key in ("iteration", "temperature", "logit_scale", "cls_bias", "slot_prior_strength", "prototype_mode_strength", "tpa_tau")} for label, bank in banks.items()},
              "native_replay": {"max_logit_error": max(r["logit_max_abs_error"] for r in native_checks),
                                "max_score_error": max(r["score_max_abs_error"] for r in native_checks),
                                "topk_symmetric_difference_total": sum(r["topk_symmetric_difference"] for r in native_checks)},
              "interpretation_limits": [
                  "Final classifier only: upstream query fusion, projections, boxes and CLIP scores stay native to each feature source.",
                  "Both cross combinations failing indicates incompatibility/co-adaptation; it does not allocate training blame.",
                  "Features include the trained final linear projection, not raw CLIP ROI features.",
                  "Paired class means condition on boxes eligible in both checkpoints; omitted GT are separately counted.",
                  "Pair-topk measures coverage, not one-to-one true positives; use the LVIS panel matching below for TP/FP.",
                  "Verified-negative controls are sampled, so panel PR cannot substitute for full-validation APr.",
              ], "panel_pr": {}}
    write_json(args.output, report)
    for variant, variant_predictions in predictions.items():
        prediction_file = Path(args.output).parent / f"{variant}_predictions.json"
        write_json(prediction_file, variant_predictions)
        print(f"[LVIS panel] {variant}: {len(variant_predictions)} predictions", flush=True)
        report["panel_pr"][variant] = evaluate_panel_predictions(dataset, panel["image_ids"], variant_predictions,
                                                               iou_thresholds=args.iou_thresholds, max_dets=args.max_dets)
        print(f"[LVIS panel] {variant}: {report['panel_pr'][variant]['summary_by_iou']}", flush=True)
        write_json(args.output, report)
    report["complete"] = True
    report["panel_pr_deltas"] = {}
    print("\n=== Final-only swap deltas (panel, NOT AP) ===", flush=True)
    for changed, native in (("new_old", "new_new"), ("old_new", "old_old"), ("new_mean", "new_new")):
        key = f"{changed}_minus_{native}"
        report["panel_pr_deltas"][key] = {}
        for iou, changed_pr in report["panel_pr"][changed]["summary_by_iou"].items():
            native_pr = report["panel_pr"][native]["summary_by_iou"][iou]
            delta = {metric: changed_pr[metric] - native_pr[metric]
                     for metric in ("true_positives", "false_positives")}
            report["panel_pr_deltas"][key][iou] = delta
            print(f"{key} IoU={iou}: {delta}", flush=True)
    write_json(args.output, report)
    print(f"[save] {args.output}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-checkpoint", required=True)
    parser.add_argument("--new-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config-file", default="lami_dino/configs/dino_convnext_large_4scale_12ep_lvis.py")
    parser.add_argument("--annotations", default="dataset/lvis/lvis_v1_val.json")
    parser.add_argument("--phase", choices=("all", "dump", "analyze"), default="all")
    parser.add_argument("--device", default="cuda:0", help="analyze can use cpu; dump needs working Detectron2/Detrex")
    parser.add_argument("--negative-images", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--iou-thresholds", type=float, nargs="+", default=[0.5, 0.75])
    parser.add_argument("--max-dets", type=int, default=300)
    parser.add_argument("--alpha", type=float, default=0.0)
    parser.add_argument("--beta", type=float, default=0.3)
    parser.add_argument("--novel-scale", type=float, default=3.0)
    parser.add_argument("--tpa-tau", type=float, default=0.004375)
    parser.add_argument("--cls-tau", type=float, default=0.07)
    parser.add_argument("--query-chunk-size", type=int, default=128)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--log-interval", type=int, default=25)
    args = parser.parse_args()
    if args.negative_images < 0 or min(args.max_dets, args.query_chunk_size, args.cpu_threads, args.log_interval) < 1:
        parser.error("Invalid negative count, top-k, chunk size, threads or log interval")
    if (not args.iou_thresholds or len(set(args.iou_thresholds)) != len(args.iou_thresholds)
            or any(not 0 < t <= 1 for t in args.iou_thresholds)):
        parser.error("IoU thresholds must be distinct and in (0,1]")
    if not (0 <= args.alpha <= 1 and 0 <= args.beta <= 1 and args.novel_scale > 0 and args.tpa_tau > 0 and args.cls_tau > 0):
        parser.error("Invalid classification/fusion protocol")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        parser.error("CUDA unavailable in this Python environment; use the lami interpreter, or --phase analyze --device cpu with an existing cache")
    torch.set_num_threads(args.cpu_threads)
    torch.manual_seed(args.seed)
    # Fail dependency checks before a costly dump. Analyze remains independent
    # of Detectron2/Detrex, but LVIS matching is required for both workflows.
    try:
        import lvis  # noqa: F401
    except ImportError as error:
        raise ImportError("Install/use LVIS in the lami environment before running the panel") from error
    output_dir = Path(args.output).resolve().parent
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = output_dir / "pairing_cache"
    dataset = json.loads(Path(args.annotations).read_text())
    panel = select_panel(dataset, negative_images=args.negative_images, seed=args.seed)
    if not panel["rare_image_ids"]:
        raise ValueError("No rare-GT images in validation annotations")
    print(f"[panel] {panel['counts']} policy={panel['control_policy']}", flush=True)
    print("[hash] checkpoint and code identities; existing cache is reused only on an exact match", flush=True)
    sources = sorted(set(REPO_ROOT.glob("lami_dino/**/*.py")) | set(REPO_ROOT.glob("detrex/config/**/*.py"))
                     | set(REPO_ROOT.glob("configs/common/**/*.py")) | {
        Path(__file__).resolve(), REPO_ROOT / "tools/pairing_lvis_support.py",
        REPO_ROOT / "tools/diagnose_tpa_usage.py", REPO_ROOT / args.config_file,
        REPO_ROOT / "detrex/modeling/backbone/convnext.py",
    })
    code_hash = hashlib.sha256("".join(f"{path}:{sha256_file(path)}\n" for path in sources).encode()).hexdigest()
    inputs = {"schema_version": SCHEMA_VERSION, "annotations_sha256": sha256_file(args.annotations),
              "asset_sha256": input_asset_hashes(args, cache_dir),
              "code_sha256": code_hash, "image_ids": panel["image_ids"], "negative_images": args.negative_images, "seed": args.seed,
              "old_checkpoint": str(Path(args.old_checkpoint).resolve()), "old_sha256": sha256_file(args.old_checkpoint),
              "new_checkpoint": str(Path(args.new_checkpoint).resolve()), "new_sha256": sha256_file(args.new_checkpoint),
              "alpha": args.alpha, "beta": args.beta, "novel_scale": args.novel_scale, "tpa_tau": args.tpa_tau, "cls_tau": args.cls_tau,
              "max_dets": args.max_dets, "iou_thresholds": args.iou_thresholds}
    fingerprint = locked_manifest(cache_dir / "manifest.json", inputs)
    write_json(output_dir / "panel.json", panel)
    if args.phase in ("all", "dump"):
        for label in VARIANTS:
            dump_checkpoint(args, label, panel, dataset, fingerprint, cache_dir)
    if args.phase in ("all", "analyze"):
        with torch.no_grad():
            analyze_cache(args, panel, dataset, fingerprint, cache_dir)


if __name__ == "__main__":
    main()
