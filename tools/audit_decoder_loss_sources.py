#!/usr/bin/env python3
"""Bounded first-order decoder loss-source audit; no optimizer, updates or AP run.

Four paired 32-image probe windows per endpoint by default (256 exposures in
total). One GPU, native train_norare losses; NOT a distributed optimizer replay.
Validation gradients are read-only observables, never training supervision.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import contextmanager
from copy import deepcopy
import gc
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
import torch.nn.functional as F

from lami_dino.checkpoint_init import load_trusted_torch_file
from tools.audit_tpa_gradients import save_gradients
from tools.capture_tpa_gradients import state_versions
from tools.compare_rare_pr_reports import file_identity, load_json, save_json
from tools.decoder_loss_audit_ops import (
    GROUPS, average_gradients, classification_reconstruction, component_gradients,
    directional_effects, gradient_summary, isolated_rng, official_control_regions,
    select_controls, summarize_probes,
)
from tools.diagnose_rare_fp_regions import fingerprint
from tools.evaluate_decoder_rollback import endpoint_state, validate_audit
from tools.query_path_update_ops import canonical_name, clear_eval_caches, key_group, match_regions
from tools.tpa_geometry_audit_ops import tensor_digest


def check_budget(args):
    if (not 1 <= args.windows <= 4 or not 1 <= args.microbatches <= 16
            or not 1 <= args.batch_size <= 4 or not 1 <= args.controls_per_split <= 4
            or args.cpu_threads < 1 or not 0 <= args.seed < 2**31):
        raise ValueError("Budgets: windows<=4, microbatches<=16, batch_size<=4, controls/split<=4")
    total = 2 * args.windows * args.microbatches * args.batch_size
    if total > 256:
        raise ValueError("Hard budget: <=256 training-image exposures across BOTH endpoints")
    return total


def prepare(args):
    budget = check_budget(args)
    report, _ = validate_audit(args.query_audit)
    sources = report["inputs"]["sources"]
    stage = Path(report["inputs"]["stage_report"]["path"]).parent
    output = Path(args.output_dir).resolve()
    protected = [Path(args.query_audit).resolve(), stage, *[Path(s["path"]).resolve() for s in sources.values()]]
    if any(output == p or output in p.parents or p in output.parents for p in protected):
        raise ValueError("Output must be separate from source checkpoints, predictions and audit caches")
    records = {s: [{**r, "panel": "focus", "frequency": "r"}
                   for r in report["interventions"][s]["native"]["regions"]] for s in ("old", "new")}
    image_ids = sorted({r["image_id"] for r in records["new"]})
    if not 1 <= len(image_ids) <= 8:
        raise ValueError("Expected <=8 existing focus images")
    dataset = load_json(sources["annotations"]["path"])
    controls = select_controls(dataset, {r["category"] for r in records["new"]}, image_ids,
                               per_split=args.controls_per_split, seed=args.seed + 701)
    # Lock training code too: the old endpoint audit's hash covered the model/eval path.
    code = (set(ROOT.glob("lami_dino/**/*.py")) | set(ROOT.glob("configs/common/**/*.py"))
            | set(ROOT.glob("detrex/modeling/**/*.py")) | set(ROOT.glob("detrex/data/**/*.py"))
            | {ROOT / "tools/train_net.py", Path(__file__).resolve(), ROOT / "tools/decoder_loss_audit_ops.py",
               ROOT / "tools/capture_tpa_gradients.py", ROOT / "tools/query_path_update_ops.py",
               ROOT / "tools/pairing_lvis_support.py"})
    identity = {"query_audit": file_identity(args.query_audit), "sources": sources,
                "windows": args.windows, "microbatches": args.microbatches, "batch_size": args.batch_size,
                "seed": args.seed, "controls": controls, "focus_images": image_ids,
                "training_image_exposures": budget, "device": args.device,
                "code": {str(p.relative_to(ROOT)): file_identity(p)["sha256"] for p in sorted(code)},
                "torch_version": str(torch.__version__)}
    signature = fingerprint(identity)
    manifest = output / "manifest.json"
    if manifest.exists():
        previous = load_json(manifest)
        if previous != {"inputs": identity, "fingerprint": signature}:
            raise ValueError("Audit inputs/code/budget changed; use a new output directory")
    else:
        if args.analyze_only:
            raise ValueError("--analyze-only requires an existing locked audit")
        if output.exists() and any(output.iterdir()):
            raise ValueError("Refusing a nonempty output directory without this audit's manifest")
        save_json(manifest, {"inputs": identity, "fingerprint": signature})
    print(f"[budget] {budget} train-image exposures maximum, "
          f"{2 * (len(image_ids) + len(controls))} validation forwards maximum; no updates", flush=True)
    return SimpleNamespace(report=report, sources=sources, stage=stage, output=output, records=records,
                           dataset=dataset, controls=controls, signature=signature, identity=identity)


def read_locked(path, signature, *, tensor=False):
    value = load_trusted_torch_file(path) if tensor else load_json(path)
    if value.get("fingerprint") != signature:
        raise ValueError(f"Stale probe cache: {path}")
    return value


def unchanged(model, versions):
    if state_versions(model) != versions or any(p.grad is not None for p in model.parameters()):
        raise ValueError("Parameters/persistent buffers changed or .grad populated during read-only audit")


def training_windows(model, parameters, cfg, args, ctx, side, iteration, delta):
    from detectron2.config import instantiate
    from detectron2.data import MetadataCatalog
    from detectron2.utils.events import EventStorage

    if cfg.dataloader.train.dataset.names != "lvis_v1_train_norare":
        raise ValueError("Never use validation/rare GT for training gradients")
    cfg.dataloader.train.total_batch_size = args.batch_size
    cfg.dataloader.train.num_workers = 0
    device_index = torch.device(args.device).index or 0
    windows = []
    reference_path = ctx.output / "new_capture.json"
    reference = (read_locked(reference_path, ctx.signature)["windows"]
                 if side == "old" and reference_path.exists() else None)
    reference = getattr(ctx, "training_reference", reference)
    model.train()
    model.tpa_advance_step = False
    if any(isinstance(m, torch.nn.modules.batchnorm._BatchNorm) and m.training for m in model.modules()):
        raise ValueError("Training BatchNorm requires distributed-statistics analysis; stopping")
    versions = state_versions(model)
    original_filter, sampled = model.filter_content_info, {}

    def record_filter(data):
        indices, targets = original_filter(data)
        sampled["fedloss_category_indices"] = indices.detach().cpu().tolist()
        return indices, targets

    model.filter_content_info = record_filter
    try:
        with isolated_rng(args.seed, device_index), EventStorage(start_iter=iteration):
            loader = iter(instantiate(cfg.dataloader.train))
            annotation = file_identity(MetadataCatalog.get(cfg.dataloader.train.dataset.names).json_file)
            if reference and annotation != reference[0]["train_annotations"]:
                raise ValueError("Endpoint training annotations differ; no gradient probes started")
            for w in range(args.windows):
                path = ctx.output / side / f"training_window_{w}.pt"
                saved = read_locked(path, ctx.signature, tensor=True) if path.exists() else None
                if saved and (saved["side"] != side or saved["window"] != w):
                    raise ValueError("Cached gradient endpoint/window differs")
                if saved and saved["train_annotations"] != annotation:
                    raise ValueError("Training annotations changed since capture")
                micros, gradients = [], []
                for micro in range(args.microbatches):
                    data = next(loader)
                    if len(data) != args.batch_size:
                        raise ValueError("Unexpected microbatch size")
                    for item in data:
                        if any(c < 0 or c >= len(model.novel_idx) or bool(model.novel_idx[c])
                               for c in item["instances"].gt_classes.tolist()):
                            raise ValueError("Rare/invalid GT entered train_norare probe")
                    mapped = [{"image_id": int(d["image_id"]), "image": tensor_digest(d["image"]),
                               "boxes": tensor_digest(d["instances"].gt_boxes.tensor),
                               "classes": tensor_digest(d["instances"].gt_classes)} for d in data]
                    if reference and mapped != reference[w]["microbatches"][micro]["mapped_inputs"]:
                        raise ValueError("Endpoint training images/augmentation differ; stopping before backward")
                    if saved:
                        if mapped != saved["microbatches"][micro]["mapped_inputs"]:
                            raise ValueError("Training loader/augmentation changed since cached window")
                        continue
                    sampled.clear()
                    with isolated_rng(args.seed + 10000 + w * args.microbatches + micro, device_index):
                        losses = model(data)
                        options = {"splitter": ctx.loss_splitter} if hasattr(ctx, "loss_splitter") else {}
                        g, connected, loss_keys, loss_values = component_gradients(losses, parameters, **options)
                    if "fedloss_category_indices" not in sampled:
                        raise ValueError("Missing native FedLoss capture")
                    if reference and sampled["fedloss_category_indices"] != reference[w]["microbatches"][micro]["fedloss_category_indices"]:
                        raise ValueError("Endpoint FedLoss category sampling differs")
                    gradients.append(g)
                    record = {"mapped_inputs": mapped, **sampled, "connections": connected,
                              "loss_keys": loss_keys, "weighted_losses": loss_values}
                    if hasattr(ctx, "verify_micro"):
                        ctx.verify_micro(record, reference[w]["microbatches"][micro])
                    micros.append(record)
                    del losses, data
                    print(f"[train partial gradients] {side} window={w+1}/{args.windows} "
                          f"micro={micro+1}/{args.microbatches}", flush=True)
                unchanged(model, versions)
                if saved is None:
                    averaged = average_gradients(gradients)
                    saved = {"fingerprint": ctx.signature, "side": side, "window": w,
                             "train_annotations": annotation, "microbatches": micros,
                             "gradients": averaged, "summary": gradient_summary(averaged, delta)}
                    if "class_final" in averaged:
                        saved["classification_reconstruction"] = classification_reconstruction(averaged)
                    if hasattr(ctx, "verify_window"):
                        saved["parent_gradient_replay"] = ctx.verify_window(saved, w)
                    save_gradients(path, saved)
                print(f"[window] {side}/{w} " + str(saved["summary"]), flush=True)
                windows.append(saved)
            del loader
    finally:
        model.filter_content_info = original_filter
    return windows


@contextmanager
def capture_eval_graph(model):
    """Retain first-order decoder logits; CLIP and geometric bookkeeping detached."""
    classifier = model.class_embed[model.transformer.decoder.num_layers - 1]
    original_logits, original_inference = classifier._compute_tpa_logits, model.inference
    capture = {}

    def logits(x, *, content_inds, additional_class):
        value = original_logits(x, content_inds=content_inds, additional_class=additional_class)
        capture["logits"] = value
        capture["features"] = x.detach()
        return value

    def inference(box_cls, box_pred, image_sizes, wo_sigmoid=False):
        capture["boxes"] = box_pred.detach()
        capture["scores"] = box_cls.detach() if wo_sigmoid else box_cls.detach().sigmoid()
        return original_inference(box_cls, box_pred, image_sizes, wo_sigmoid=wo_sigmoid)

    classifier._compute_tpa_logits, model.inference = logits, inference
    try:
        yield capture
    finally:
        classifier._compute_tpa_logits, model.inference = original_logits, original_inference
        capture.clear()


def validation_probes(model, parameters, cfg, args, ctx, side, windows, bank):
    from detectron2.config import instantiate
    from detectron2.data import get_detection_dataset_dicts, MetadataCatalog
    from tools.diagnose_tpa_usage import box_cxcywh_to_xyxy

    if (cfg.dataloader.test.dataset.names != "lvis_v1_val" or cfg.dataloader.test.mapper.is_train
            or cfg.dataloader.test.mapper.augmentation_with_crop is not None):
        raise ValueError("Expected native deterministic LVIS validation mapper")
    augmentation = cfg.dataloader.test.mapper.augmentation
    if len(augmentation) != 1 or "ResizeShortestEdge" not in str(augmentation[0]._target_):
        raise ValueError("Expected resize-only validation augmentation")
    edges = augmentation[0].short_edge_length
    if not isinstance(edges, int) and len(set(edges)) != 1:
        raise ValueError("Random validation resize is unsupported")
    raw = get_detection_dataset_dicts(names="lvis_v1_val", filter_empty=False)
    if file_identity(MetadataCatalog.get("lvis_v1_val").json_file)["sha256"] != ctx.sources["annotations"]["sha256"]:
        raise ValueError("Validation annotations differ from audit")
    mapper = instantiate(cfg.dataloader.test.mapper)
    image_ids = sorted(set(ctx.identity["focus_images"]) | {r["image_id"] for r in ctx.controls})
    raw = {r["image_id"]: r for r in raw if r["image_id"] in image_ids}
    category_indices = {c: i for i, c in enumerate(bank["category_ids"])}
    name_indices = {c["name"]: category_indices[c["id"]] for c in ctx.dataset["categories"]}
    controls_by_image = {r["image_id"]: r for r in ctx.controls}
    model.eval()
    clear_eval_caches(model)
    versions = state_versions(model)
    probes, control_coverage = [], []
    with capture_eval_graph(model) as captured:
        for image_id in image_ids:
            path = ctx.output / side / f"validation_{image_id}.json"
            mapped = mapper(deepcopy(raw[image_id]))
            pixels = tensor_digest(mapped["image"])
            if path.exists():
                cached = read_locked(path, ctx.signature)
                if cached["pixels"] != pixels or cached["side"] != side or cached["image_id"] != image_id:
                    raise ValueError("Mapped validation image changed")
                probes.extend(cached["probes"])
                control_coverage.extend(cached["control_coverage"])
                continue
            captured.clear()
            with torch.enable_grad():
                result = model([mapped])
            boxes = box_cxcywh_to_xyxy(captured["boxes"][0].float()).clamp(0, 1).cpu()
            boxes *= boxes.new_tensor([mapped["width"], mapped["height"], mapped["width"], mapped["height"]])
            scores = captured["scores"][0].float().cpu()
            logits = captured["logits"][0]
            if not logits.requires_grad or not torch.isfinite(logits).all():
                raise ValueError("Native evaluation logits must have a finite decoder gradient path")
            local_coverage = []
            if image_id in controls_by_image:
                if side == "new":
                    rows = official_control_regions(ctx.dataset, controls_by_image[image_id], boxes, scores, bank["category_ids"])
                else:
                    reference_dir = getattr(ctx, "control_reference_dir", ctx.output)
                    reference_signature = getattr(ctx, "control_reference_signature", ctx.signature)
                    source = read_locked(reference_dir / "new" / f"validation_{image_id}.json", reference_signature)
                    rows = source["anchor_regions"]
                matches = ([{"query_id": r["query_id"], "iou": 1.} for r in rows] if side == "new"
                           else match_regions(rows, boxes, .5))
                local_coverage = [{**controls_by_image[image_id], "side": side,
                                   "anchor_counts": dict(Counter(r["kind"] for r in rows)),
                                   "matched_counts": dict(Counter(r["kind"] for r, m in zip(rows, matches) if m))}]
            else:
                rows = [r for r in ctx.records[side] if r["image_id"] == image_id]
                matches = [{"query_id": r["query_id"], "iou": 1.} for r in rows]
                sample = load_trusted_torch_file(ctx.stage / "pairing_cache" / side / f"{image_id}.pt")
                if sample["fingerprint"] != ctx.report["inputs"]["parent_fingerprint"]:
                    raise ValueError("Focus cache fingerprint changed")
                if ((captured["features"][0].detach().float().cpu() - sample["features"]).abs().max() > 2e-3
                        or (boxes - sample["query_boxes"]).abs().max() > .05):
                    raise ValueError("Gradient-enabled native forward does not reproduce original focus cache")
            grouped = defaultdict(list)
            for row, match in zip(rows, matches):
                grouped[row["panel"], row["category"], row["kind"]].append((row, match))
            local_probes = []
            for number, ((panel, category, kind), items) in enumerate(grouped.items()):
                keep = [(r, m) for r, m in items if m is not None]
                effects = []
                if keep:
                    index = name_indices[category]
                    qs = [m["query_id"] for _, m in keep]
                    coefficient = .7 if bool(bank["novel_mask"][index]) else 1.
                    observable = coefficient * F.logsigmoid(logits[qs, index]).sum()
                    values = torch.autograd.grad(observable, parameters, allow_unused=True,
                                                 retain_graph=number < len(grouped) - 1)
                    h = torch.cat([(v.detach() if v is not None else torch.zeros_like(p)).flatten()
                                   for p, v in zip(parameters, values)]).cpu()
                    effects = [directional_effects(h, window["gradients"]) for window in windows]
                    del h, values, observable
                local_probes.append({"panel": panel, "category": category, "kind": kind,
                                     "image_id": image_id, "expected_count": len(items), "count": len(keep),
                                     "matches": [m for _, m in items], "effects": effects})
            unchanged(model, versions)
            cached = {"fingerprint": ctx.signature, "side": side, "image_id": image_id, "pixels": pixels,
                      "anchor_regions": rows, "control_coverage": local_coverage, "probes": local_probes}
            save_json(path, cached)
            probes.extend(local_probes)
            control_coverage.extend(local_coverage)
            for coverage in local_coverage:
                print(f"[control coverage] {coverage}", flush=True)
            captured.clear()
            del result, logits, mapped, boxes, scores
            print(f"[validation first-order] {side} image={image_id} groups={len(local_probes)}", flush=True)
    return probes, control_coverage


def capture_side(args, ctx, side, delta):
    from detectron2.config import LazyConfig, instantiate
    from tools.diagnose_detector_tpa_pairing import classifier_options

    if not torch.cuda.is_available() or not args.device.startswith("cuda") or torch.distributed.is_initialized():
        raise ValueError("Capture requires one CUDA process, not DDP; use lami Python")
    torch.backends.cudnn.benchmark = False
    options = SimpleNamespace(alpha=0., beta=.3, novel_scale=3., tpa_tau=.004375,
                              cls_tau=.07, max_dets=300, device=args.device)
    cfg = LazyConfig.load(ctx.sources["config_file"]["path"])
    cfg = LazyConfig.apply_overrides(cfg, classifier_options(options, "new") + [
        "model.classifier.tpa_prototype_mode_strength=0.0"])
    if cfg.model.tpa_stabilization_steps != 0 or cfg.model.tpa_task_gradient_scale != 1.:
        raise ValueError("Requires native full task gradients")
    if cfg.model.get("teacher_rpsa", False) or not cfg.model.transformer.use_rpsa or not cfg.model.use_fed_loss:
        raise ValueError("Expected ordinary encoder RPSA and FedLoss")
    model = instantiate(cfg.model)
    iteration = 70999 if side == "new" else 56799
    checkpoint = load_trusted_torch_file(ctx.sources[side + "_checkpoint"]["path"])
    state = endpoint_state(checkpoint, iteration)
    model.load_state_dict(state, strict=True)
    del checkpoint, state
    model.to(args.device)
    named = [(k, p) for k, p in model.named_parameters() if key_group(canonical_name(k)) == "decoder_core"]
    named.sort(key=lambda kv: canonical_name(kv[0]))
    if [canonical_name(k) for k, _ in named] != sorted(ctx.report["inventory"]["decoder_core"]["keys"]):
        raise ValueError("Live decoder parameter layout differs from endpoint audit")
    ids = {id(p) for _, p in named}
    for p in model.parameters():
        p.requires_grad_(id(p) in ids)
    parameters = tuple(p for _, p in named)
    bank = load_trusted_torch_file(ctx.stage / "pairing_cache" / side / "bank.pt")
    if bank["fingerprint"] != ctx.report["inputs"]["parent_fingerprint"]:
        raise ValueError("Classifier bank fingerprint changed")
    windows = training_windows(model, parameters, cfg, args, ctx, side, iteration, delta)
    probes, controls = validation_probes(model, parameters, cfg, args, ctx, side, windows, bank)
    result = {"fingerprint": ctx.signature, "side": side, "iteration": iteration,
              "parameters": [k for k, _ in named],
              "windows": [{k: v for k, v in w.items() if k != "gradients"} for w in windows],
              "probes": probes, "control_coverage": controls, "weights_unchanged": True,
              "optimizer_created": False, "training_updates": 0}
    save_json(ctx.output / f"{side}_capture.json", result)
    del model, parameters, named, windows, bank
    gc.collect()
    torch.cuda.empty_cache()
    return result


def analyze(ctx, captures):
    old, new = captures["old"], captures["new"]
    for side, capture in captures.items():
        if (capture["side"] != side or not capture["weights_unchanged"]
                or capture["optimizer_created"] or capture["training_updates"]):
            raise ValueError("Capture does not certify read-only endpoint identity")
    if len(old["windows"]) != len(new["windows"]):
        raise ValueError("Endpoint window counts differ")
    for a, b in zip(old["windows"], new["windows"]):
        if a["train_annotations"] != b["train_annotations"]:
            raise ValueError("Endpoint training annotations differ")
        if len(a["microbatches"]) != len(b["microbatches"]):
            raise ValueError("Endpoint microbatch counts differ")
        for x, y in zip(a["microbatches"], b["microbatches"]):
            if any(x[k] != y[k] for k in ("mapped_inputs", "fedloss_category_indices", "loss_keys")):
                raise ValueError("Endpoints did not use paired mapped inputs/FedLoss/loss grouping")
    rows = {s: summarize_probes(captures[s]["probes"], len(captures[s]["windows"])) for s in captures}
    paired_controls = {}
    for side in captures:
        for split in ("control_rare", "control_base"):
            selected = [c for c in captures[side]["control_coverage"] if c["panel"] == split]
            paired_controls[f"{side}/{split}"] = sum(
                all(c["anchor_counts"].get(k, 0) > 0
                    and c["anchor_counts"][k] == c["matched_counts"].get(k, 0) for k in ("tp", "fp"))
                for c in selected)
    report = {"complete": True, "fingerprint": ctx.signature, "inputs": ctx.identity,
              "captures": {s: file_identity(ctx.output / f"{s}_capture.json") for s in captures},
              "gradient_summaries": {s: [w["summary"] for w in captures[s]["windows"]] for s in captures},
              "local_margin_derivatives": rows,
              "control_coverage": {s: captures[s]["control_coverage"] for s in captures},
              "fully_paired_control_classes": paired_controls,
              "paired_training_inputs_verified": True, "training_updates": 0, "optimizer_created": False,
              "scope": ["First-order partial derivatives with respect to decoder_core only, not AdamW updates or AP.",
                        "Native detach boundaries (including per-layer reference_points.detach) and discrete query identities are retained: autograd-path conditional sensitivity, not full finite-step inference sensitivity.",
                        "Negative unit-descent TP-FP derivative is locally adverse; disconnected is not evidence of no indirect effect.",
                        "No routing/clipping/optimizer simulation. Encoder-RPSA can affect non-TPA global clipping; APR/TPA can change future inputs.",
                        "Single-GPU 32-image mean gradient uses serial microbatches/FedLoss, not the original DDP effective batch.",
                        "Same checkpoint/iteration in every window; validation gradients never train the detector.",
                        "Controls are annotation/seed-selected, with fixed 10ep official TP/FP labels. Missing pairs are not zero effects.",
                        "Probe signs across four windows are screening evidence, not historical loss attribution or statistical significance."]}
    save_json(ctx.output / "report.json", report)
    print("\n=== Decoder loss-source local audit (unit negative-gradient direction) ===")
    print(f"[fully paired controls] {paired_controls}")
    if any(n == 0 for n in paired_controls.values()):
        print("[scope warning] At least one endpoint/control split has no complete TP/FP pair; "
              "cannot claim cross-control consistency or absence of harm.")
    for side, entries in rows.items():
        groups = defaultdict(list)
        for row in entries:
            groups[row["panel"], row["category"], row["loss_group"]].append(row["unit_descent_derivative"]["tp_minus_fp"])
        for (panel, category, source), values in groups.items():
            print(f"{side:3} {panel:12} {category:18} {source:15} " +
                  " ".join("NA" if v is None else f"{v:+.6g}" for v in values), flush=True)
    print(f"[save] {ctx.output / 'report.json'}; no training/optimizer steps", flush=True)
    return report


def run(args):
    torch.set_num_threads(args.cpu_threads)
    ctx = prepare(args)
    if args.prepare_only:
        return ctx.identity
    captures = {}
    pending = [s for s in ("new", "old") if not (ctx.output / f"{s}_capture.json").exists()]
    if pending and args.analyze_only:
        raise ValueError(f"Missing completed captures {pending}; --analyze-only never falls back to GPU")
    delta = None
    if pending:
        vectors = []
        for side in ("old", "new"):
            checkpoint = load_trusted_torch_file(ctx.sources[side + "_checkpoint"]["path"])
            state = endpoint_state(checkpoint, 56799 if side == "old" else 70999)
            vectors.append(torch.cat([state[k].flatten() for k in sorted(ctx.report["inventory"]["decoder_core"]["keys"])]))
            del state, checkpoint
        delta = vectors[1] - vectors[0]
        del vectors
    for side in ("new", "old"):  # Fix native10 control labels before measuring the 8ep counterpart.
        path = ctx.output / f"{side}_capture.json"
        captures[side] = read_locked(path, ctx.signature) if path.exists() else capture_side(args, ctx, side, delta)
    return analyze(ctx, captures)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query-audit", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--windows", type=int, default=4)
    parser.add_argument("--microbatches", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--controls-per-split", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--cpu-threads", type=int, default=2)
    parser.add_argument("--device", default="cuda:0")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--prepare-only", action="store_true")
    mode.add_argument("--analyze-only", action="store_true")
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
