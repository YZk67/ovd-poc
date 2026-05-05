"""
TPA prototype-usage diagnostic.

Runs inference on N LVIS val images, hooks the final decoder's class_embed
to capture (a) the final-layer query features [B,Q,D] and (b) the TPA
prototypes [C,K,D]. For every (query, class) pair whose sigmoid score
exceeds a threshold, computes argmax over K (which prototype "won").
Aggregates per class to get a distribution over K and its entropy.

Decision rule (avg normalized entropy across classes with >=5 detections):
  > 0.6   --> B-route viable: detection-space t-SNE colored by argmax-K
              should reveal sub-clusters within each class.
  0.4-0.6 --> grey zone: inspect per-class detail.
  < 0.4   --> B-route fails: prototypes get used in collapsed mode at
              detection time. Stick with A-route (prototype-space t-SNE).

Outputs to --save-dir:
  features.pt       (last image's query features [Q,D])
  prototypes.pt     (TPA prototypes [C,K,D])
  per_class_argmax.pt  (dict: class_id -> [argmax-K, ...])

Run:
  python tools/diagnose_tpa_usage.py \
    --config lami_dino/configs/dino_convnext_large_4scale_12ep_lvis.py \
    --ckpt /root/autodl-tmp/model_final.pth \
    --num-images 100 \
    --save-dir /root/autodl-tmp/tpa_diagnostic
"""
import argparse
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def install_capture_hook(model, capture):
    """Monkey-patch the last class_embed's _compute_tpa_logits so each call
    saves the same `features` and `prototypes` tensors used in its einsum."""
    last = model.transformer.decoder.class_embed[-1]
    original = last._compute_tpa_logits

    def patched(x, *, content_inds, additional_class):
        result = original(x, content_inds=content_inds, additional_class=additional_class)

        if last._external_prototypes is not None:
            prototypes = last._external_prototypes
        elif last._cached_eval is not None:
            prototypes = last._cached_eval
        else:
            return result

        if last.norm_weight:
            prototypes = F.normalize(prototypes, p=2, dim=-1)

        features = last._normalize_features(x)
        capture["features"] = features.detach().cpu()
        capture["prototypes"] = prototypes.detach().cpu()
        return result

    last._compute_tpa_logits = patched


def diagnose(per_class_argmax, K, min_det):
    log_K = math.log(K)
    detail = []
    for cls, ams in per_class_argmax.items():
        if len(ams) < min_det:
            continue
        cnt = Counter(ams)
        probs = np.array([cnt[k] / len(ams) for k in range(K)])
        entropy = -np.sum(probs * np.log(probs + 1e-12))
        detail.append((cls, len(ams), entropy, probs))

    if not detail:
        print(f"[!] no class has >= {min_det} detections; lower --score-thresh or "
              f"--min-det-per-class.")
        return None

    detail.sort(key=lambda x: -x[2])
    norm_ents = np.array([d[2] for d in detail]) / log_K

    print(f"\n=== Diagnostic Report (K={K}) ===")
    print(f"Classes with >= {min_det} detections: {len(detail)}")
    print(f"Normalized entropy (entropy / log(K)):")
    print(f"  mean   = {norm_ents.mean():.3f}")
    print(f"  median = {np.median(norm_ents):.3f}")
    print(f"  min    = {norm_ents.min():.3f}")
    print(f"  max    = {norm_ents.max():.3f}")

    avg = float(norm_ents.mean())
    print(f"\n=== Decision ===")
    if avg > 0.6:
        verdict = "B-VIABLE"
        msg = "B-route VIABLE: detection-space sub-cluster viz should work"
    elif avg > 0.4:
        verdict = "GREY"
        msg = "GREY ZONE: inspect per-class detail before deciding"
    else:
        verdict = "A-ONLY"
        msg = "B-route FAILS at detection time. Stick with A-route (prototype t-SNE)"
    print(f"  norm_entropy = {avg:.3f}  ->  {verdict}")
    print(f"  {msg}")

    print(f"\nTop 8 most-diverse classes (highest entropy):")
    for cls, n, ent, probs in detail[:8]:
        ps = " ".join(f"{p:.2f}" for p in probs)
        print(f"  cls={cls:4d}  N={n:5d}  ent={ent:.3f}  norm={ent/log_K:.3f}  "
              f"probs=[{ps}]")

    print(f"\nTop 8 most-collapsed classes (lowest entropy):")
    for cls, n, ent, probs in detail[-8:]:
        ps = " ".join(f"{p:.2f}" for p in probs)
        print(f"  cls={cls:4d}  N={n:5d}  ent={ent:.3f}  norm={ent/log_K:.3f}  "
              f"probs=[{ps}]")
    return verdict


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config",
                   default="lami_dino/configs/dino_convnext_large_4scale_12ep_lvis.py")
    p.add_argument("--ckpt", default="/root/autodl-tmp/model_final.pth")
    p.add_argument("--num-images", type=int, default=100)
    p.add_argument("--score-thresh", type=float, default=0.1)
    p.add_argument("--min-det-per-class", type=int, default=5)
    p.add_argument("--save-dir", default="/root/autodl-tmp/tpa_diagnostic",
                   help="dir to save features + prototypes + per_class_argmax")
    args = p.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(repo_root))

    from detectron2.checkpoint import DetectionCheckpointer
    from detectron2.config import LazyConfig, instantiate
    from detectron2.data import (
        build_detection_test_loader,
        get_detection_dataset_dicts,
    )

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
    dataset_dicts = dataset_dicts[: args.num_images]
    loader = build_detection_test_loader(
        dataset=dataset_dicts,
        mapper=instantiate(cfg.dataloader.test.mapper),
        num_workers=2,
    )

    capture = {}
    install_capture_hook(model, capture)

    per_class_argmax = defaultdict(list)
    K = None
    last_features = None
    last_prototypes = None

    print(f"[run] inference on {len(dataset_dicts)} images...")
    with torch.no_grad():
        for i, batched_inputs in enumerate(loader):
            _ = model(batched_inputs)

            features = capture.get("features")        # [1, Q, D]
            prototypes = capture.get("prototypes")    # [C, K, D]
            if features is None or prototypes is None:
                print(f"[!] image {i}: hook didn't capture; skipping")
                continue

            B, Q, D = features.shape
            C, Kp, _ = prototypes.shape
            K = Kp
            last_features = features.clone()
            last_prototypes = prototypes.clone()

            logits_4d = torch.einsum("bqd,ckd->bqck", features, prototypes)
            logits_3d = torch.logsumexp(logits_4d, dim=-1)
            scores = logits_3d.sigmoid()

            mask = scores[0] > args.score_thresh
            qs, cs = mask.nonzero(as_tuple=True)
            for q, c in zip(qs.tolist(), cs.tolist()):
                argmax_k = logits_4d[0, q, c].argmax().item()
                per_class_argmax[c].append(argmax_k)

            if (i + 1) % 10 == 0:
                total = sum(len(v) for v in per_class_argmax.values())
                print(f"  [{i+1}/{len(dataset_dicts)}] cumulative dets={total}")

    total = sum(len(v) for v in per_class_argmax.values())
    print(f"\n[done] processed {len(dataset_dicts)} images, {total} detections "
          f"(score>{args.score_thresh}), {len(per_class_argmax)} active classes")

    if K is None:
        print("[!] no captures; aborting")
        return

    verdict = diagnose(per_class_argmax, K, args.min_det_per_class)

    if args.save_dir:
        out = Path(args.save_dir)
        out.mkdir(parents=True, exist_ok=True)
        torch.save(last_features, out / "features_last.pt")
        torch.save(last_prototypes, out / "prototypes.pt")
        torch.save(dict(per_class_argmax), out / "per_class_argmax.pt")
        print(f"\n[save] features_last.pt, prototypes.pt, per_class_argmax.pt -> {out}")
        print(f"       (use these for plotting: prototype t-SNE [A] or detection viz [B])")


if __name__ == "__main__":
    main()
