#!/usr/bin/env python3
"""Bounded bidirectional 8ep/10ep module swaps on an existing rare-stage panel.

No training, gradients, optimizer, full-validation evaluation, or checkpoint
writes. Actual endpoint weights, not a hypothetical SGD direction. These are
module interventions, NOT attribution to a loss or replay of historical AdamW.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import gc
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
import torch.nn.functional as F

from lami_dino.checkpoint_init import load_trusted_torch_file
from tools.compare_rare_pr_reports import file_identity, load_json, save_json
from tools.query_path_update_ops import (
    GROUPS, canonical_name, clear_eval_caches, inventory, region_effects, swap_group,
)
from tools.rare_stage_update_ops import grouped_margins, selected_logits


def prepare(args):
    from tools.diagnose_rare_fp_regions import fingerprint, model_code_hash, read_manifest, validate_sample
    from tools.tpa_geometry_audit_ops import extract_shared_tpa

    stage = Path(args.stage_dir).resolve()
    output = Path(args.output_dir).resolve()
    if output == stage or stage in output.parents or output in stage.parents:
        raise ValueError("Use a separate sibling output directory; never overwrite the stage cache")
    if not 1 <= args.max_images <= 8 or not 1 <= args.max_forwards <= 192 or args.cpu_threads < 1:
        raise ValueError("Hard budgets: 1..8 images, 1..192 forwards, positive CPU threads")
    if not 0 < args.match_iou <= 1:
        raise ValueError("Invalid match IoU")
    report = load_json(stage / "report.json")
    manifest = read_manifest(stage / "manifest.json")
    parent = read_manifest(stage / "pairing_cache/manifest.json")
    if not report.get("complete") or report["sources"] != manifest["inputs"]["sources"]:
        raise ValueError("Need a complete stage report matching its locked manifest")
    sources = report["sources"]
    protected = [Path(v["path"]).resolve() for v in sources.values()]
    if any(output == p or output in p.parents for p in protected):
        raise ValueError("Output directory contains protected source inputs")
    for name in ("old_checkpoint", "new_checkpoint", "annotations", "config_file", "prompt_bank"):
        print(f"[identity] {name}", flush=True)
        if file_identity(sources[name]["path"])["sha256"] != sources[name]["sha256"]:
            raise ValueError(f"Stage input changed: {name}")
    if model_code_hash(sources["config_file"]["path"]) != parent["inputs"]["code_sha256"]:
        raise ValueError("Model/config code changed since native cache; refusing mixed evidence")
    for path, digest in parent["inputs"]["asset_sha256"].items():
        if file_identity(path)["sha256"] != digest:
            raise ValueError(f"Model asset changed: {path}")
    protocol = parent["inputs"]
    for k, value in {"alpha": 0., "beta": .3, "novel_scale": 3., "tpa_tau": .004375,
                     "cls_tau": .07, "max_dets": 300}.items():
        if protocol[k] != value:
            raise ValueError(f"Unexpected stage protocol: {k}")
    states, banks, samples = {}, {}, {}
    for side, iteration in (("old", 56799), ("new", 70999)):
        ckpt = load_trusted_torch_file(sources[side + "_checkpoint"]["path"])
        tpa, info = extract_shared_tpa(ckpt)
        if (info["iteration"] != iteration or float(tpa["prototype_mode_strength"]) != 0.
                or abs(float(tpa["slot_prior_strength"]) - .2) > 1e-6
                or tpa["prototype_queries"].shape != (5, 256)):
            raise ValueError("Expected no-radius K5 slot-prior=.2 8ep/10ep endpoints")
        states[side] = {k[7:] if k.startswith("module.") else k: v for k, v in ckpt["model"].items()}
        del ckpt
        if protocol[side + "_sha256"] != sources[side + "_checkpoint"]["sha256"]:
            raise ValueError("Stage checkpoint and native cache differ")
        banks[side] = load_trusted_torch_file(stage / "pairing_cache" / side / "bank.pt")
    for key in ("category_ids", "temperature", "logit_scale", "tpa_tau", "prompt_sha256", "vlm_temperature"):
        if banks["old"][key] != banks["new"][key]:
            raise ValueError(f"Readout protocol changed: {key}")
    for key in ("vlm_text", "novel_mask"):
        if not torch.equal(banks["old"][key], banks["new"][key]):
            raise ValueError(f"CLIP bank changed: {key}")
    records = report["paired_regions"]
    if not records["old"] or len(records["old"]) != len(records["new"]):
        raise ValueError("Need nonempty paired regions")
    for old, new in zip(records["old"], records["new"]):
        if any(old[k] != new[k] for k in ("category", "kind", "image_id")):
            raise ValueError("Stage region labels/order are not paired")
    image_ids = sorted({r["image_id"] for r in records["new"]})
    if len(image_ids) > args.max_images:
        raise ValueError(f"Panel has {len(image_ids)} images; budget={args.max_images}")
    dataset = load_json(sources["annotations"]["path"])
    id_to_index = {c: i for i, c in enumerate(banks["new"]["category_ids"])}
    name_to_index = {c["name"]: id_to_index[c["id"]] for c in dataset["categories"]}
    for side in ("old", "new"):
        for image_id in image_ids:
            sample = load_trusted_torch_file(stage / "pairing_cache" / side / f"{image_id}.pt")
            validate_sample(sample, banks[side], parent["fingerprint"], side, image_id)
            samples[side, image_id] = sample
        for row in records[side]:
            sample = samples[side, row["image_id"]]
            q, index = row["query_id"], name_to_index[row["category"]]
            if row["kind"] not in ("tp", "fp") or not 0 <= q < len(sample["features"]):
                raise ValueError("Invalid stage query/label")
            if not banks[side]["novel_mask"][index]:
                raise ValueError("Expected rare region; fixed beta=.3 is rare-only")
            if not torch.allclose(sample["query_boxes"][q].double(), torch.tensor(row["box_xyxy"]).double(), atol=1e-4, rtol=0):
                raise ValueError("Stage anchor box differs from native cache")
            z = selected_logits(sample["features"][q:q+1], banks[side]["prototypes"], torch.tensor([index]), banks[side])
            if abs(float(z) - row["native_cache_logit"]) > 2e-3:
                raise ValueError("Stage anchor logit differs from native cache")
    groups = inventory(states["old"], states["new"])
    active = [g for g in GROUPS if groups.get(g, {}).get("changed_keys")]
    variants = ["native"] + active + (["all_query"] if active else [])
    forwards = 2 * len(image_ids) * len(variants)
    if forwards > args.max_forwards:
        raise ValueError(f"Needs {forwards} forwards, budget={args.max_forwards}; no GPU started")
    identity = {"schema_version": 1, "stage_report": file_identity(stage / "report.json"),
                "parent_fingerprint": parent["fingerprint"], "sources": sources,
                "model_code_sha256": protocol["code_sha256"],
                "audit_code": {str(p.relative_to(ROOT)): file_identity(p)["sha256"] for p in (
                    Path(__file__).resolve(), ROOT / "tools/query_path_update_ops.py")},
                "image_ids": image_ids, "variants": variants, "match_iou": args.match_iou,
                "device": args.device, "torch_version": str(torch.__version__)}
    print(f"[budget] {len(image_ids)} images x 2 endpoints x {len(variants)} variants = "
          f"{forwards} forwards maximum; ZERO training/gradient probes", flush=True)
    return SimpleNamespace(stage=stage, output=output, source=report, identity=identity,
                           fingerprint=fingerprint(identity), states=states, banks=banks, samples=samples,
                           records=records, name_to_index=name_to_index, inventory=groups,
                           image_ids=image_ids, variants=variants, max_forwards=forwards)


def cache_path(ctx, side, variant, image_id):
    return ctx.output / "forward_cache" / side / variant / f"{image_id}.pt"


def read_forward(ctx, side, variant, image_id):
    value = load_trusted_torch_file(cache_path(ctx, side, variant, image_id))
    if (value.get("fingerprint") != ctx.fingerprint or value.get("side") != side
            or value.get("variant") != variant or value.get("image_id") != image_id):
        raise ValueError("Forward cache identity mismatch")
    expected = ctx.samples[side, image_id]
    for key in ("features", "query_boxes"):
        if value[key].shape != expected[key].shape or not torch.isfinite(value[key]).all():
            raise ValueError(f"Invalid cached {key}")
    if variant == "native":
        if ((value["features"] - expected["features"]).abs().max() > 2e-3
                or (value["query_boxes"] - expected["query_boxes"]).abs().max() > .05):
            raise ValueError("Cached native forward differs from stage cache")
    if not value.get("mapped_input_sha256"):
        raise ValueError("Missing mapped input identity")
    return value


def capture_forwards(args, ctx):
    from detectron2.config import LazyConfig, instantiate
    from detectron2.data import get_detection_dataset_dicts, MetadataCatalog
    from tools.diagnose_detector_tpa_pairing import classifier_options, save_tensor_file, validate_parameter_state
    from tools.diagnose_tpa_usage import install_capture_hooks, box_cxcywh_to_xyxy
    from tools.tpa_geometry_audit_ops import tensor_digest

    options = SimpleNamespace(alpha=0., beta=.3, novel_scale=3., tpa_tau=.004375,
                              cls_tau=.07, max_dets=300, device=args.device)
    cfg = LazyConfig.load(ctx.source["sources"]["config_file"]["path"])
    cfg = LazyConfig.apply_overrides(cfg, classifier_options(options, "new") + [
        "model.classifier.tpa_slot_prior_strength=0.2", "model.classifier.tpa_prototype_mode_strength=0.0"])
    aug = cfg.dataloader.test.mapper.augmentation
    if (len(aug) != 1 or "ResizeShortestEdge" not in str(aug[0]._target_)
            or cfg.dataloader.test.mapper.augmentation_with_crop is not None
            or cfg.dataloader.test.mapper.is_train):
        raise ValueError("Requires native deterministic resize-only eval mapper")
    edges = aug[0].short_edge_length
    if not isinstance(edges, int) and len(set(edges)) != 1:
        raise ValueError("Random eval scale is unsupported")
    dataset_name = cfg.dataloader.test.dataset.names
    if not isinstance(dataset_name, str):
        if len(dataset_name) != 1:
            raise ValueError("Expected one validation dataset")
        dataset_name = dataset_name[0]
    records = get_detection_dataset_dicts(names=dataset_name, filter_empty=False)
    meta = MetadataCatalog.get(dataset_name)
    if Path(meta.json_file).resolve() != Path(ctx.source["sources"]["annotations"]["path"]).resolve():
        raise ValueError("Mapper annotations differ from source")
    mapper = instantiate(cfg.dataloader.test.mapper)
    records_by_id = {r["image_id"]: r for r in records}
    mapped = {i: mapper(records_by_id[i]) for i in ctx.image_ids}
    del records, records_by_id, mapper
    for side in ("old", "new"):
        for image_id in ctx.image_ids:
            if cache_path(ctx, side, "native", image_id).exists():
                saved = read_forward(ctx, side, "native", image_id)
                if saved["mapped_input_sha256"] != tensor_digest(mapped[image_id]["image"]):
                    raise ValueError("Image pixels changed since partial forward cache; no mixed rerun")
    model = instantiate(cfg.model).to(args.device).eval()
    model.requires_grad_(False)
    if model.transformer.decoder.num_layers != 6:
        raise ValueError("Module partition assumes six decoder layers")
    # Check the model really shares exactly the aliases used by our partition.
    aliases = {}
    for key, tensor in model.state_dict().items():
        if not tensor.numel():
            continue
        storage = (tensor.data_ptr(), tuple(tensor.shape), tuple(tensor.stride()))
        if storage in aliases and canonical_name(key) != aliases[storage]:
            raise ValueError(f"Unexpected runtime parameter alias: {key}")
        aliases[storage] = canonical_name(key)
    capture = {}
    classifier = install_capture_hooks(model, capture)
    if not model.score_ensemble or not classifier.use_tpa or not classifier.norm_weight:
        raise ValueError("Requires normalized TPA classifier and CLIP ensemble")
    with torch.no_grad():
        for side in ("old", "new"):
            validate_parameter_state(model, ctx.states[side])
            model.load_state_dict(ctx.states[side], strict=True)
            clear_eval_caches(model)
            donor = ctx.states["new" if side == "old" else "old"]
            for variant in ctx.variants:
                pending = [i for i in ctx.image_ids if not cache_path(ctx, side, variant, i).exists()]
                for i in set(ctx.image_ids) - set(pending):
                    read_forward(ctx, side, variant, i)
                if not pending:
                    print(f"[reuse] {side}/{variant}: all images cached", flush=True)
                    continue
                context = nullcontext() if variant == "native" else swap_group(model, donor, variant)
                with context:
                    for image_id in pending:
                        capture.clear()
                        inputs = mapped[image_id]
                        model([inputs])
                        features = capture["projected_features"][0].float().cpu()
                        boxes = box_cxcywh_to_xyxy(capture["query_boxes"][0].float()).clamp(0, 1).cpu()
                        boxes *= boxes.new_tensor([inputs["width"], inputs["height"], inputs["width"], inputs["height"]])
                        if not torch.isfinite(features).all() or not torch.isfinite(boxes).all():
                            raise ValueError("Nonfinite module-intervention forward")
                        errors = None
                        if variant == "native":
                            cached = ctx.samples[side, image_id]
                            errors = {"feature": float((features - cached["features"]).abs().max()),
                                      "box": float((boxes - cached["query_boxes"]).abs().max()),
                                      "prototype": float((capture["prototypes"].float().cpu() - ctx.banks[side]["prototypes"]).abs().max())}
                            if errors["feature"] > 2e-3 or errors["box"] > .05 or errors["prototype"] > 1e-5:
                                raise ValueError(f"Native endpoint no longer reproduces stage cache: {errors}")
                        save_tensor_file(cache_path(ctx, side, variant, image_id), {
                            "fingerprint": ctx.fingerprint, "side": side, "variant": variant,
                            "image_id": image_id, "features": features, "query_boxes": boxes,
                            "mapped_input_sha256": tensor_digest(inputs["image"]),
                            "native_replay_error": errors})
                        print(f"[forward] {side}/{variant} image={image_id} replay={errors}", flush=True)
            for key, value in model.state_dict().items():
                if not torch.equal(value.cpu(), ctx.states[side][key]):
                    raise ValueError(f"Endpoint not restored after interventions: {key}")
    capture.clear()
    del model, classifier, mapped
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def analyze(ctx, threshold):
    result = {}
    for side in ("old", "new"):
        result[side] = {}
        bank = ctx.banks[side]
        for variant in ctx.variants:
            rows = []
            for image_id in ctx.image_ids:
                records = [r for r in ctx.records[side] if r["image_id"] == image_id]
                native = read_forward(ctx, side, "native", image_id)
                hybrid = read_forward(ctx, side, variant, image_id)
                if native["mapped_input_sha256"] != hybrid["mapped_input_sha256"]:
                    raise ValueError("Mapped pixels changed between interventions")
                ids = torch.tensor([ctx.name_to_index[r["category"]] for r in records], dtype=torch.long)
                # The native control uses EXACT source IDs, not re-matching box ties.
                if variant == "native":
                    rows.extend({**r, "match": {"query_id": r["query_id"], "iou": 1.},
                                 "feature_cosine": 1., "delta_fused_log_score": 0.} for r in records)
                else:
                    rows.extend(region_effects(records, native, hybrid, bank, ids, threshold=threshold)["regions"])
            result[side][variant] = summarize(rows)
        # Isolate final scalar bias analytically; it is deliberately fixed above.
        other = ctx.banks["new" if side == "old" else "old"]
        shifted = dict(bank, cls_bias=other["cls_bias"])
        rows = []
        for r in ctx.records[side]:
            sample = ctx.samples[side, r["image_id"]]
            x = sample["features"][r["query_id"]:r["query_id"]+1]
            ids = torch.tensor([ctx.name_to_index[r["category"]]])
            z0 = selected_logits(x, bank["prototypes"], ids, bank)
            z1 = selected_logits(x, bank["prototypes"], ids, shifted)
            rows.append({**r, "match": {"query_id": r["query_id"], "iou": 1.},
                         "feature_cosine": 1., "delta_fused_log_score": float(.7 * (F.logsigmoid(z1) - F.logsigmoid(z0)))})
        result[side]["final_bias_only"] = summarize(rows)
    return result


def summarize(rows):
    keep = [r for r in rows if r["match"] is not None]
    margins = grouped_margins(torch.tensor([r["delta_fused_log_score"] for r in keep], dtype=torch.float64), keep)
    for name in sorted({r["category"] for r in rows}):
        full = [r for r in rows if r["category"] == name]
        matched = [r for r in full if r["match"] is not None]
        row = margins.setdefault(name, {"tp_regions": 0, "fp_regions": 0, "pairs": 0, "mean_tp_minus_fp": None})
        row["matched_regions"], row["total_regions"] = len(matched), len(full)
        row["full_panel_margin_change"] = row["mean_tp_minus_fp"] if len(matched) == len(full) else None
        row["min_match_iou"] = min((r["match"]["iou"] for r in matched), default=None)
        row["reused_query_indices"] = sum(r["match"]["query_id"] == r["query_id"] for r in matched)
    return {"by_class": margins, "regions": rows}


def display(report):
    print("\n=== Actual endpoint module swaps; frozen terminal bank/bias/CLIP ===", flush=True)
    print("Module effects are NOT additive and NOT AP. Incomplete panels print NA.", flush=True)
    for side, variants in report["interventions"].items():
        print(f"\n{side}: {'8ep + 10ep module' if side == 'old' else '10ep + 8ep module'}", flush=True)
        print("module                  class           dMargin      kept   minIoU", flush=True)
        for variant, result in variants.items():
            for name, row in result["by_class"].items():
                value = row["full_panel_margin_change"]
                number = f"{value:+.6f}" if value is not None else "NA"
                print(f"{variant:23} {name:14} {number:>10} "
                      f"{row['matched_regions']:2}/{row['total_regions']:<2} {row['min_match_iou']}", flush=True)
    print("Positive opens the selected TP-FP log-score gap. Negative closes it. "
          "Geometric matching is score-blind; loss of counterparts is NOT counted as improvement. "
          "No optimizer history, training-loss cause, or full-validation performance claim.", flush=True)
    print("\n=== Bidirectional check (negative injection + positive rollback is consistent local harm) ===", flush=True)
    for row in report["bidirectional_check"]:
        print(f"{row['module']:23} {row['category']:14} "
              f"8ep+10ep={row['inject_10ep_into_8ep']} 10ep+8ep={row['rollback_10ep_to_8ep']}", flush=True)


def run(args):
    from tools.diagnose_detector_tpa_pairing import locked_manifest

    torch.set_num_threads(args.cpu_threads)
    ctx = prepare(args)
    locked_manifest(ctx.output / "manifest.json", ctx.identity)
    if args.prepare_only:
        save_json(ctx.output / "preflight.json", {"inventory": ctx.inventory, "max_forwards": ctx.max_forwards,
                                                "image_ids": ctx.image_ids, "variants": ctx.variants})
        print("[prepare only] no GPU forwards; remove --prepare-only to run", flush=True)
        return
    missing = [(s, v, i) for s in ("old", "new") for v in ctx.variants for i in ctx.image_ids
               if not cache_path(ctx, s, v, i).is_file()]
    if missing and args.analyze_only:
        raise FileNotFoundError(f"Missing {len(missing)} forwards; --analyze-only NEVER runs GPU: {missing[:3]}")
    report = {"complete": False, "inputs": ctx.identity, "inventory": ctx.inventory,
              "max_native_forwards": ctx.max_forwards, "training_image_exposures": 0,
              "optimizer_created": False, "checkpoint_files_written": False,
              "excluded_stage_regions": ctx.source["excluded_regions"],
              "scope": "Bidirectional actual endpoint-weight module interventions on selected regions; "
                       "NOT historical update replay, additive attribution, loss attribution, or LVIS AP.",
              "matching": "Within each image: maximum-cardinality then maximum-IoU one-to-one; "
                          "no class scores/GT used; ambiguous geometric ties excluded. Native query IDs are not assumed invariant.",
              "fixed_readout": "Anchor endpoint terminal prototypes, cls_bias, and CLIP score. "
                               "upstream_tpa swaps affect query construction only in the measured readout.",
              "interventions": {}}
    save_json(ctx.output / "report.json", report)
    if missing:
        if args.device.startswith("cuda") and not torch.cuda.is_available():
            raise ValueError("CUDA unavailable; use the lami Python with compiled detrex")
        capture_forwards(args, ctx)
    report["interventions"] = analyze(ctx, args.match_iou)
    report["bidirectional_check"] = [
        {"module": variant, "category": name,
         "inject_10ep_into_8ep": old["full_panel_margin_change"],
         "rollback_10ep_to_8ep": report["interventions"]["new"][variant]["by_class"][name]["full_panel_margin_change"]}
        for variant, data in report["interventions"]["old"].items() if variant != "native"
        for name, old in data["by_class"].items()]
    report["complete"] = True
    save_json(ctx.output / "report.json", report)
    display(report)
    print(f"[save] {ctx.output / 'report.json'}", flush=True)
    return report


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-dir", required=True, help="Completed audit_rare_stage_updates output, including caches")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--max-images", type=int, default=8)
    parser.add_argument("--max-forwards", type=int, default=192)
    parser.add_argument("--match-iou", type=float, default=.5)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--prepare-only", action="store_true")
    mode.add_argument("--analyze-only", action="store_true", help="Read completed caches only; never fall back to GPU")
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
