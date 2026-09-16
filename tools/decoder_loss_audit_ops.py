"""First-order decoder-only loss probes. No updates or historical attribution."""

from __future__ import annotations

from collections import Counter, defaultdict
from contextlib import contextmanager
from copy import deepcopy
import math
import random
import re

import numpy as np
import torch

from tools.pairing_lvis_support import _lvis_from_dataset


GROUPS = ("classification", "bbox_l1", "bbox_giou", "rpsa", "apr")
CLASSIFICATION_PARTS = ("class_final", "class_aux", "class_dn", "class_encoder")
DETAIL_GROUPS = ("classification", *CLASSIFICATION_PARTS, *GROUPS[1:])


def split_losses(losses):
    groups = {g: [] for g in GROUPS}
    for key, value in losses.items():
        if not key.startswith("loss"):
            continue
        if not torch.is_tensor(value) or value.numel() != 1 or not torch.isfinite(value).all():
            raise ValueError(f"Nonfinite/non-scalar loss: {key}")
        if key == "loss_apr":
            group = "apr"
        elif key == "loss_rpsa":
            group = "rpsa"
        else:
            group = next((g for prefix, g in (("loss_class", "classification"),
                          ("loss_bbox", "bbox_l1"), ("loss_giou", "bbox_giou"))
                          if key == prefix or key.startswith(prefix + "_")), None)
        if group is None:
            raise ValueError(f"Unclassified training loss: {key}")
        groups[group].append(key)
    if any(not keys for keys in groups.values()):
        raise ValueError("Expected classification/L1/GIoU/RPSA/APR native weighted losses")
    return groups


def split_classification_losses(losses):
    """Keep the independently differentiated total as a reconstruction check.

    Encoder classification is not a decoder auxiliary loss. Record it explicitly
    and require it to be disconnected from decoder_core in this experiment.
    """
    base = split_losses(losses)
    parts = {k: [] for k in CLASSIFICATION_PARTS}
    for key in base["classification"]:
        if key == "loss_class":
            group = "class_final"
        elif re.fullmatch(r"loss_class_\d+", key):
            group = "class_aux"
        elif re.fullmatch(r"loss_class_dn(?:_\d+)?", key):
            group = "class_dn"
        elif key == "loss_class_enc":
            group = "class_encoder"
        else:
            raise ValueError(f"Unclassified classification branch: {key}")
        parts[group].append(key)
    if any(not keys for keys in parts.values()):
        raise ValueError("Expected final/aux/DN/encoder classification branches")
    return {g: (parts[g] if g in parts else base[g]) for g in DETAIL_GROUPS}


def classification_reconstruction(gradients):
    total = gradients["classification"]
    if gradients["class_encoder"] is not None:
        raise ValueError("Encoder classification unexpectedly connects to decoder_core")
    if total is None:
        raise ValueError("Total classification has no decoder gradient")
    parts = [gradients[k] for k in CLASSIFICATION_PARTS if gradients[k] is not None]
    summed = sum(parts, torch.zeros_like(total))
    error = (summed.double() - total.double()).norm().item()
    relative = error / max(total.double().norm().item(), 1e-12)
    if not torch.allclose(summed, total, rtol=5e-4, atol=2e-6) or relative > 5e-4:
        raise ValueError("Classification branches do not reconstruct total gradient")
    return {"relative_l2_error": relative, "max_abs_error": (summed-total).abs().max().item()}


def component_gradients(losses, parameters, *, splitter=split_losses):
    """Same forward/targets/dropout/matching for every weighted loss component.

    None means no autograd path to ANY decoder-core parameter, not a small norm.
    Partially unused parameters are explicit. Frozen upstreams keep their forward
    values; only partial derivatives with respect to decoder_core are requested.
    """
    groups = splitter(losses)
    totals = {g: sum(losses[k] for k in keys) for g, keys in groups.items()}
    active = [g for g, value in totals.items() if value.requires_grad]
    gradients, connections = {}, {}
    for group, total in totals.items():
        values = (torch.autograd.grad(total, parameters, allow_unused=True,
                  retain_graph=group != active[-1]) if total.requires_grad else (None,) * len(parameters))
        connected = [i for i, value in enumerate(values) if value is not None]
        connections[group] = connected
        gradients[group] = (torch.cat([(value.detach() if value is not None else torch.zeros_like(p)).flatten()
                             for p, value in zip(parameters, values)]).cpu() if connected else None)
        if gradients[group] is not None and not torch.isfinite(gradients[group]).all():
            raise ValueError(f"Nonfinite decoder gradient: {group}")
    if "class_final" in groups:
        classification_reconstruction(gradients)
    return gradients, connections, groups, {k: float(v.detach()) for k, v in losses.items() if k.startswith("loss")}


def average_gradients(rows):
    result = {}
    if not rows or any(row.keys() != rows[0].keys() for row in rows):
        raise ValueError("Microbatch gradient group sets differ or are empty")
    for group in rows[0]:
        present = [row[group] for row in rows if row[group] is not None]
        result[group] = sum(present) / len(rows) if present else None
    return result


def gradient_summary(gradients, historical_delta):
    result = {}
    dn = float(historical_delta.double().norm()) if historical_delta is not None else 0.
    for group, gradient in gradients.items():
        norm = float(gradient.double().norm()) if gradient is not None else 0.
        result[group] = {"connected": gradient is not None, "norm": norm,
                         "descent_cosine_with_8_to_10ep_delta":
                         float(torch.dot(-gradient.double(), historical_delta.double()) / (norm * dn))
                         if norm > 1e-12 and dn > 1e-12 else None}
    return result


def directional_effects(observable_gradient, gradients):
    """dM/dε for θ_decoder - ε g; unit descent separates direction from magnitude."""
    h = observable_gradient.detach().double().cpu()
    if not torch.isfinite(h).all():
        raise ValueError("Nonfinite validation-observable gradient")
    result = {}
    for group, g in gradients.items():
        norm = float(g.double().norm()) if g is not None else 0.
        raw = float(torch.dot(h, -g.double())) if g is not None else 0.
        if not math.isfinite(raw):
            raise ValueError("Nonfinite local derivative")
        result[group] = {"raw_descent_derivative": raw,
                         "unit_descent_derivative": raw / norm if norm > 1e-12 else None}
    return result


@contextmanager
def isolated_rng(seed, cuda_device=None):
    """Pair stochastic forwards without allowing them to alter the loader RNG."""
    python_state, numpy_state = random.getstate(), np.random.get_state()
    devices = [] if cuda_device is None else [cuda_device]
    try:
        with torch.random.fork_rng(devices=devices):
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if cuda_device is not None:
                with torch.cuda.device(cuda_device):
                    torch.cuda.manual_seed(seed)
            yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)


def select_controls(dataset, excluded_names, excluded_images, *, per_split=4, seed=1729):
    """Select by annotations and seed ONLY, before reading model predictions/AP.

    Equal-class sampling, one image/class. Exclude non-exhaustive/ignored/crowd
    target annotations. No replacement if a selected class yields no TP/FP pair.
    """
    rng = random.Random(seed)
    images = {i["id"]: i for i in dataset["images"]}
    candidates = defaultdict(set)
    for ann in dataset["annotations"]:
        image = images[ann["image_id"]]
        if (ann["image_id"] not in excluded_images and not ann.get("ignore", 0)
                and not ann.get("iscrowd", 0) and ann.get("area", 1) > 0
                and ann["category_id"] not in image.get("not_exhaustive_category_ids", [])):
            candidates[ann["category_id"]].add(ann["image_id"])
    selected, used = [], set(excluded_images)
    for split in ("rare", "base"):
        categories = sorted([c for c in dataset["categories"] if c["name"] not in excluded_names
                             and (c["frequency"] == "r") == (split == "rare")
                             and candidates[c["id"]]], key=lambda c: c["id"])
        rng.shuffle(categories)
        for category in categories:
            options = sorted(candidates[category["id"]] - used)
            if not options:
                continue
            image_id = rng.choice(options)
            selected.append({"category_id": category["id"], "category": category["name"],
                             "frequency": category["frequency"], "panel": "control_" + split,
                             "image_id": image_id})
            used.add(image_id)
            if sum(r["panel"] == "control_" + split for r in selected) == per_split:
                break
        if sum(r["panel"] == "control_" + split for r in selected) != per_split:
            raise ValueError("Insufficient independent annotation-selected control classes/images")
    return selected


def official_control_regions(dataset, control, boxes, scores, category_ids, *, limit=3):
    """Label native 10ep top-300 detections using official LVIS ignores/matching.

    All classes compete for 300 slots first. Highest-score TP/FP regions of the
    preselected class are kept, at most limit each. No search for another class.
    """
    from lvis import LVIS, LVISEval, LVISResults

    image_id, category_id = control["image_id"], control["category_id"]
    ids = torch.topk(scores.flatten(), min(300, scores.numel())).indices.tolist()
    candidates, predictions = [], []
    for index in ids:
        q, c = divmod(index, scores.shape[1])
        x0, y0, x1, y1 = boxes[q].tolist()
        if x1 <= x0 or y1 <= y0:
            continue
        candidates.append({**control, "query_id": q, "class_index": c,
                           "box_xyxy": [x0, y0, x1, y1]})
        predictions.append({"image_id": image_id, "category_id": category_ids[c],
                            "score": float(scores[q, c]), "bbox": [x0, y0, x1-x0, y1-y0]})
    if not predictions:
        return []
    subset = deepcopy({**dataset, "images": [i for i in dataset["images"] if i["id"] == image_id],
                       "annotations": [a for a in dataset["annotations"] if a["image_id"] == image_id]})
    gt = _lvis_from_dataset(LVIS, subset)
    dt = LVISResults(gt, deepcopy(predictions), max_dets=300)
    ev = LVISEval(gt, dt, "bbox")
    ev.params.img_ids, ev.params.cat_ids = [image_id], [category_id]
    ev.params.iou_thrs = np.asarray([.5])
    ev.params.area_rng = [ev.params.area_rng[ev.params.area_rng_lbl.index("all")]]
    ev.params.area_rng_lbl = ["all"]
    ev.evaluate()
    result, counts = [], Counter()
    for item in ev.eval_imgs:
        if item is None:
            continue
        for index, dt_id in enumerate(item["dt_ids"]):
            if item["dt_ignore"][0, index]:
                continue
            record = dt.anns[int(dt_id)]
            position = int(dt_id) - 1
            if not 0 <= position < len(predictions) or any(record[k] != predictions[position][k]
                                                         for k in ("image_id", "category_id", "bbox", "score")):
                raise ValueError("Official LVIS detection IDs differ from query mapping")
            kind = "tp" if item["dt_matches"][0, index] else "fp"
            if counts[kind] < limit:
                result.append({**candidates[position], "kind": kind})
                counts[kind] += 1
    return result


def summarize_probes(probes, windows, *, groups=GROUPS):
    """Aggregate directional TP-FP differences, requiring full paired coverage."""
    buckets = defaultdict(list)
    for probe in probes:
        buckets[probe["panel"], probe["category"]].append(probe)
    output = []
    for (panel, category), rows in sorted(buckets.items()):
        expected = {kind: sum(r["expected_count"] for r in rows if r["kind"] == kind) for kind in ("tp", "fp")}
        matched = {kind: sum(r["count"] for r in rows if r["kind"] == kind) for kind in ("tp", "fp")}
        valid = all(expected[k] == matched[k] and matched[k] > 0 for k in ("tp", "fp"))
        for window in range(windows):
            for source in groups:
                values = {}
                for metric in ("raw_descent_derivative", "unit_descent_derivative"):
                    means = {}
                    for kind in ("tp", "fp"):
                        selected = [r["effects"][window][source][metric] for r in rows
                                    if r["kind"] == kind and r["count"]]
                        means[kind] = sum(selected) / matched[kind] if selected and all(v is not None for v in selected) else None
                    values[metric] = {"tp_mean": means["tp"], "fp_mean": means["fp"],
                                      "tp_minus_fp": means["tp"] - means["fp"]
                                      if valid and all(v is not None for v in means.values()) else None}
                output.append({"panel": panel, "category": category, "window": window,
                               "loss_group": source, "full_pair_coverage": valid,
                               "expected": expected, "matched": matched, **values})
    return output
