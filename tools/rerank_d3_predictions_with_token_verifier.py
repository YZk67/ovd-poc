#!/usr/bin/env python3
"""
Rerank D3 OWLv2-style predictions with a token-level crop/phrase verifier.

This is the proposal-source counterpart of
tools/rerank_d3_topk_candidates_with_token_verifier.py. It consumes COCO-format
predictions, selects class-agnostic proposal boxes, optionally uses the expanded
crop-verifier score cache to preselect box/phrase pairs, then runs the
token-level cross-attention verifier on only those candidate pairs.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))

from rerank_d3_predictions_with_clip_crops import (  # noqa: E402
    _evaluate_coco,
    _expanded_base_scores_from_category_scores,
    _expanded_cache_to_arrays,
    _expanded_category_scores,
    _expanded_score_cache_path,
    _fuse_verifier_logit_matrix,
    _group_predictions,
    _load_categories,
    _load_expanded_score_cache,
    _load_image_ids_from_jsonl,
    _load_json,
    _resolve_image_path,
    _save_json,
    _select_class_agnostic_proposals,
    _top_matrix_indices,
)
from train_d3_token_cross_verifier import (  # noqa: E402
    ClipTokenEncoder,
    TokenCrossVerifier,
    _expanded_xyxy,
    _jsonable_args,
    _torch_load,
)


def _sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def _logit(score: float) -> float:
    score = min(max(score, 1e-6), 1.0 - 1e-6)
    return math.log(score / (1.0 - score))


def _fuse_score(detector_score: float, verifier_logit: float, *, mode: str, fusion_weight: float) -> float:
    verifier_prob = _sigmoid(verifier_logit)
    if mode == "logit_add":
        return _sigmoid(_logit(detector_score) + fusion_weight * verifier_logit)
    if mode == "linear":
        return (1.0 - fusion_weight) * detector_score + fusion_weight * verifier_prob
    if mode == "replace":
        return verifier_prob
    raise ValueError(f"Unsupported token fusion mode: {mode}")


def _load_token_verifier(path: Path, device: str) -> Tuple[TokenCrossVerifier, Mapping[str, Any]]:
    checkpoint = _torch_load(path, map_location=device)
    verifier = TokenCrossVerifier(
        image_dim=int(checkpoint["image_dim"]),
        text_dim=int(checkpoint["text_dim"]),
        hidden_dim=int(checkpoint["hidden_dim"]),
        num_heads=int(checkpoint["num_heads"]),
        dropout=float(checkpoint.get("dropout", 0.0)),
        use_detector_score=bool(checkpoint.get("use_detector_score", True)),
    )
    verifier.load_state_dict(checkpoint["model_state"])
    verifier.to(device)
    verifier.eval()
    print(
        f"loaded token verifier: {path} "
        f"(epoch={checkpoint.get('epoch')}, score={checkpoint.get('score')}, "
        f"backend={checkpoint.get('backend_name')})"
    )
    return verifier, checkpoint


@torch.no_grad()
def _encode_text_bank(
    *,
    clip_encoder: ClipTokenEncoder,
    prompts: Sequence[str],
    batch_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    token_tensors = []
    mask_tensors = []
    for start in tqdm(range(0, len(prompts), batch_size), desc="encoding phrase tokens"):
        token_ids = clip_encoder.tokenizer(list(prompts[start : start + batch_size]))
        text_tokens, text_mask = clip_encoder.encode_text_tokens(token_ids)
        token_tensors.append(text_tokens.cpu().to(dtype=torch.float16))
        mask_tensors.append(text_mask.cpu())
    return torch.cat(token_tensors, dim=0), torch.cat(mask_tensors, dim=0)


def _load_candidate_cache(
    *,
    cache_dir: Path,
    image_id: int,
) -> Optional[Mapping[str, Any]]:
    path = _expanded_score_cache_path(cache_dir, image_id)
    if not path.exists():
        return None
    try:
        cache = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        cache = torch.load(path, map_location="cpu")
    if not isinstance(cache, Mapping):
        return None
    return cache


def _candidate_mlp_scores(
    *,
    base_scores: np.ndarray,
    pair_signal: np.ndarray,
    args: argparse.Namespace,
) -> np.ndarray:
    return _fuse_verifier_logit_matrix(
        base_scores,
        pair_signal,
        mode=args.candidate_verifier_fusion,
        fusion_weight=args.candidate_verifier_fusion_weight,
    )


@torch.no_grad()
def _score_token_pairs(
    *,
    verifier: TokenCrossVerifier,
    image_tokens_by_proposal: Mapping[int, torch.Tensor],
    text_tokens: torch.Tensor,
    text_masks: torch.Tensor,
    pairs: Sequence[Mapping[str, Any]],
    device: str,
    batch_size: int,
) -> List[float]:
    logits: List[float] = []
    for start in range(0, len(pairs), batch_size):
        chunk = list(pairs[start : start + batch_size])
        image_batch = torch.stack(
            [image_tokens_by_proposal[int(item["proposal_idx"])] for item in chunk],
            dim=0,
        ).to(device)
        category_idx = torch.tensor([int(item["category_idx"]) for item in chunk], dtype=torch.long)
        text_batch = text_tokens[category_idx].to(device)
        mask_batch = text_masks[category_idx].to(device)
        detector_scores = torch.tensor(
            [float(item["token_detector_score"]) for item in chunk],
            dtype=torch.float32,
            device=device,
        )
        chunk_logits = verifier(image_batch, text_batch, mask_batch, detector_scores)
        logits.extend(float(value) for value in chunk_logits.detach().cpu().tolist())
    return logits


def _select_candidate_pairs(
    *,
    proposals: Sequence[Mapping[str, Any]],
    base_scores: np.ndarray,
    candidate_scores: np.ndarray,
    category_ids: Sequence[int],
    args: argparse.Namespace,
) -> List[Dict[str, Any]]:
    pairs: List[Dict[str, Any]] = []
    for proposal_idx, category_idx in _top_matrix_indices(candidate_scores, args.candidate_topk_per_image):
        base_score = float(base_scores[proposal_idx, category_idx])
        candidate_score = float(candidate_scores[proposal_idx, category_idx])
        if args.token_base_score == "category_score":
            token_base_score = base_score
        else:
            token_base_score = candidate_score
        if args.token_detector_score == "category_score":
            token_detector_score = base_score
        else:
            token_detector_score = candidate_score
        pairs.append(
            {
                "proposal_idx": int(proposal_idx),
                "category_idx": int(category_idx),
                "category_id": int(category_ids[category_idx]),
                "base_score": base_score,
                "candidate_score": candidate_score,
                "token_base_score": float(token_base_score),
                "token_detector_score": float(token_detector_score),
            }
        )
    return pairs


def _build_uncached_inputs(
    *,
    preds: Sequence[Mapping[str, Any]],
    category_to_row: Mapping[int, int],
    category_ids: Sequence[int],
    args: argparse.Namespace,
) -> Tuple[List[Dict[str, Any]], np.ndarray, np.ndarray]:
    proposals = _select_class_agnostic_proposals(
        preds,
        topk=args.proposal_topk_per_image,
        nms_thresh=args.proposal_nms_thresh,
    )
    category_scores = _expanded_category_scores(
        proposals,
        preds,
        category_to_row=category_to_row,
        num_categories=len(category_ids),
        match_iou=args.category_score_match_iou,
    )
    base_scores = _expanded_base_scores_from_category_scores(
        proposals,
        category_scores,
        num_categories=len(category_ids),
        mode=args.expanded_base_score,
        missing_category_score_scale=args.missing_category_score_scale,
    )
    return proposals, category_scores, base_scores


def rerank(args: argparse.Namespace) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    verifier, checkpoint = _load_token_verifier(args.verifier_checkpoint, args.device)
    crop_margin = args.crop_margin if args.crop_margin is not None else float(checkpoint.get("crop_margin", 0.25))
    prompt_template = args.prompt_template or str(checkpoint.get("prompt_template", "the described target is {phrase}"))

    clip_encoder = ClipTokenEncoder(
        backend=args.clip_backend or str(checkpoint.get("clip_backend", "open_clip")),
        model_name=args.clip_model or str(checkpoint.get("clip_model", "ViT-L-14")),
        pretrained=args.clip_pretrained or str(checkpoint.get("clip_pretrained", "openai")),
        openai_clip_model=args.openai_clip_model or str(checkpoint.get("openai_clip_model", "ViT-L/14")),
        device=args.device,
    )
    print(f"loaded token backend: {clip_encoder.backend_name}")

    annotation = _load_json(args.annotation)
    predictions = _load_json(args.predictions)
    if not isinstance(predictions, list):
        raise ValueError(f"Expected prediction JSON list, got {type(predictions).__name__}.")

    image_infos = {int(item["id"]): item for item in annotation["images"]}
    categories_by_id = _load_categories(annotation, args.phrases_json)
    category_ids = sorted(categories_by_id)
    category_to_row = {category_id: idx for idx, category_id in enumerate(category_ids)}
    prompts = [prompt_template.format(phrase=categories_by_id[category_id]) for category_id in category_ids]
    text_tokens, text_masks = _encode_text_bank(
        clip_encoder=clip_encoder,
        prompts=prompts,
        batch_size=args.text_batch_size,
    )

    grouped = _group_predictions(predictions)
    image_ids = sorted(grouped)
    if args.image_id_jsonl is not None:
        selected_ids = set(_load_image_ids_from_jsonl(args.image_id_jsonl))
        image_ids = [image_id for image_id in image_ids if image_id in selected_ids]
        print(f"using image ids from {args.image_id_jsonl}: {len(image_ids)}")
    if args.max_images is not None:
        image_ids = image_ids[: args.max_images]

    results: List[Dict[str, Any]] = []
    processed_image_ids: List[int] = []
    missing_images = 0
    missing_candidate_caches = 0
    invalid_crops = 0
    processed_crops = 0
    scored_token_pairs = 0

    for image_id in tqdm(image_ids, desc="token reranking predictions"):
        preds = grouped[image_id]
        image_info = image_infos.get(image_id)
        if image_info is None:
            continue
        processed_image_ids.append(image_id)

        cache = None
        if args.candidate_source == "cache_signal":
            if args.candidate_score_cache_dir is None:
                raise ValueError("--candidate-source cache_signal requires --candidate-score-cache-dir.")
            cache = _load_candidate_cache(cache_dir=args.candidate_score_cache_dir, image_id=image_id)
            if cache is None:
                missing_candidate_caches += 1
                continue
            proposals, category_scores, pair_signal, _ = _expanded_cache_to_arrays(cache)
            if pair_signal is None:
                raise ValueError(f"Candidate cache for image {image_id} has no pair_signal.")
            base_scores = _expanded_base_scores_from_category_scores(
                proposals,
                category_scores,
                num_categories=len(category_ids),
                mode=args.expanded_base_score,
                missing_category_score_scale=args.missing_category_score_scale,
            )
            candidate_scores = _candidate_mlp_scores(base_scores=base_scores, pair_signal=pair_signal, args=args)
        else:
            proposals, _, base_scores = _build_uncached_inputs(
                preds=preds,
                category_to_row=category_to_row,
                category_ids=category_ids,
                args=args,
            )
            candidate_scores = base_scores

        if not proposals:
            continue

        candidate_pairs = _select_candidate_pairs(
            proposals=proposals,
            base_scores=base_scores,
            candidate_scores=candidate_scores,
            category_ids=category_ids,
            args=args,
        )
        if not candidate_pairs:
            continue

        image_path = _resolve_image_path(args.image_root, str(image_info["file_name"]))
        if not image_path.exists():
            missing_images += 1
            continue
        image = Image.open(image_path).convert("RGB")
        width, height = image.size

        needed_proposal_indices = sorted({int(pair["proposal_idx"]) for pair in candidate_pairs})
        crop_tensors = []
        crop_proposal_indices = []
        for proposal_idx in needed_proposal_indices:
            proposal = proposals[proposal_idx]
            xyxy = _expanded_xyxy(proposal["bbox"], width=width, height=height, margin=crop_margin)
            if xyxy is None:
                invalid_crops += 1
                continue
            crop_tensors.append(clip_encoder.preprocess(image.crop(xyxy)))
            crop_proposal_indices.append(proposal_idx)
        image.close()

        if not crop_tensors:
            continue

        image_tokens_by_proposal: Dict[int, torch.Tensor] = {}
        for start in range(0, len(crop_tensors), args.image_batch_size):
            batch = torch.stack(crop_tensors[start : start + args.image_batch_size], dim=0)
            with torch.no_grad():
                image_tokens = clip_encoder.encode_image_tokens(batch).cpu().to(dtype=torch.float16)
            for offset, proposal_idx in enumerate(crop_proposal_indices[start : start + args.image_batch_size]):
                image_tokens_by_proposal[int(proposal_idx)] = image_tokens[offset]
        processed_crops += len(image_tokens_by_proposal)

        valid_pairs = [pair for pair in candidate_pairs if int(pair["proposal_idx"]) in image_tokens_by_proposal]
        token_logits = _score_token_pairs(
            verifier=verifier,
            image_tokens_by_proposal=image_tokens_by_proposal,
            text_tokens=text_tokens,
            text_masks=text_masks,
            pairs=valid_pairs,
            device=args.device,
            batch_size=args.verifier_batch_size,
        )
        scored_token_pairs += len(token_logits)

        output_candidates = []
        for pair, token_logit in zip(valid_pairs, token_logits):
            token_base_score = float(pair["token_base_score"])
            fused_score = _fuse_score(
                token_base_score,
                token_logit,
                mode=args.token_fusion,
                fusion_weight=args.token_fusion_weight,
            )
            proposal = proposals[int(pair["proposal_idx"])]
            item = {
                "image_id": int(image_id),
                "category_id": int(pair["category_id"]),
                "bbox": [float(value) for value in proposal["bbox"]],
                "score": float(fused_score),
            }
            if args.include_debug_fields:
                item.update(
                    {
                        "proposal_score": float(proposal["score"]),
                        "source_category_id": int(proposal["source_category_id"]),
                        "base_score": float(pair["base_score"]),
                        "candidate_score": float(pair["candidate_score"]),
                        "token_base_score": token_base_score,
                        "token_detector_score": float(pair["token_detector_score"]),
                        "token_logit": float(token_logit),
                        "token_score": float(_sigmoid(token_logit)),
                        "proposal_idx": int(pair["proposal_idx"]),
                        "category_idx": int(pair["category_idx"]),
                    }
                )
            output_candidates.append(item)

        output_candidates.sort(key=lambda item: float(item["score"]), reverse=True)
        if args.keep_topk_per_image > 0:
            output_candidates = output_candidates[: args.keep_topk_per_image]
        results.extend(output_candidates)

    summary = {
        "args": _jsonable_args(args),
        "num_requested_images": len(image_ids),
        "num_processed_images": len(processed_image_ids),
        "missing_images": missing_images,
        "missing_candidate_caches": missing_candidate_caches,
        "invalid_crops": invalid_crops,
        "processed_crops": processed_crops,
        "scored_token_pairs": scored_token_pairs,
        "output_predictions": len(results),
    }
    return results, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--annotation", type=Path, default=Path("dataset/d3/annotations/d3_intra_full.json"))
    parser.add_argument("--image-root", type=Path, default=Path("dataset/d3/images"))
    parser.add_argument("--phrases-json", type=Path, default=Path("dataset/metadata/d3_phrases.json"))
    parser.add_argument("--verifier-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image-id-jsonl", type=Path, default=None)
    parser.add_argument("--prompt-template", default=None)
    parser.add_argument("--clip-backend", choices=("auto", "open_clip", "openai_clip"), default=None)
    parser.add_argument("--clip-model", default=None)
    parser.add_argument("--clip-pretrained", default=None)
    parser.add_argument("--openai-clip-model", default=None)
    parser.add_argument("--crop-margin", type=float, default=None)
    parser.add_argument("--proposal-topk-per-image", type=int, default=100)
    parser.add_argument("--proposal-nms-thresh", type=float, default=0.9)
    parser.add_argument("--keep-topk-per-image", type=int, default=100)
    parser.add_argument("--expanded-base-score", choices=("objectness", "category_score"), default="category_score")
    parser.add_argument("--category-score-match-iou", type=float, default=0.9)
    parser.add_argument("--missing-category-score-scale", type=float, default=0.3)
    parser.add_argument(
        "--candidate-source",
        choices=("category_score", "cache_signal"),
        default="cache_signal",
        help="Use category-score base only or a previous expanded crop-verifier cache for pair preselection.",
    )
    parser.add_argument(
        "--candidate-score-cache-dir",
        type=Path,
        default=None,
        help="Per-image expanded score cache from rerank_d3_predictions_with_clip_crops.py.",
    )
    parser.add_argument("--candidate-topk-per-image", type=int, default=1000)
    parser.add_argument("--candidate-verifier-fusion", choices=("logit_add", "linear", "replace"), default="logit_add")
    parser.add_argument(
        "--candidate-verifier-fusion-weight",
        type=float,
        default=0.45,
        help="Crop-verifier fusion weight used to reconstruct candidate scores from cache_signal.",
    )
    parser.add_argument(
        "--token-base-score",
        choices=("candidate_score", "category_score"),
        default="candidate_score",
        help="Base score for final token-verifier fusion.",
    )
    parser.add_argument(
        "--token-detector-score",
        choices=("candidate_score", "category_score"),
        default="category_score",
        help="Score feature passed into the token verifier. Keep category_score to match training.",
    )
    parser.add_argument("--token-fusion", choices=("logit_add", "linear", "replace"), default="logit_add")
    parser.add_argument("--token-fusion-weight", type=float, default=0.5)
    parser.add_argument("--image-batch-size", type=int, default=32)
    parser.add_argument("--text-batch-size", type=int, default=128)
    parser.add_argument("--verifier-batch-size", type=int, default=128)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--include-debug-fields", action="store_true")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--eval-output-images-only", action="store_true")
    parser.add_argument("--summary-output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results, summary = rerank(args)
    _save_json(args.output, results)
    summary_output = args.summary_output or args.output.with_suffix(args.output.suffix + ".summary.json")
    _save_json(summary_output, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"saved token-verifier predictions to {args.output}")
    print(f"saved summary to {summary_output}")
    if args.eval:
        eval_image_ids = None
        output_image_ids = sorted({int(result["image_id"]) for result in results})
        if args.eval_output_images_only or args.max_images is not None or args.image_id_jsonl is not None:
            eval_image_ids = output_image_ids
        else:
            annotation = _load_json(args.annotation)
            annotation_image_ids = {int(item["id"]) for item in annotation.get("images", [])}
            if len(output_image_ids) < len(annotation_image_ids):
                eval_image_ids = output_image_ids
                print(
                    "evaluation output covers a subset of annotation images; "
                    f"restricting COCOeval to output image ids: {len(output_image_ids)}"
                )
        _evaluate_coco(args.annotation, results, image_ids=eval_image_ids)


if __name__ == "__main__":
    main()
