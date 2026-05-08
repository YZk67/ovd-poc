#!/usr/bin/env python3
"""
Build a fixed-size LVIS phrase bank and run a text-only retrieval pilot.

The goal is to validate whether richer phrase prototypes are a better
representation target than a single class-name prototype before wiring the bank
into detector training.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HUB_OFFLINE", "1")
LVIS_CATEGORIES_PATH = ROOT / "detectron2" / "detectron2" / "data" / "datasets" / "lvis_v1_categories.py"
_spec = importlib.util.spec_from_file_location("lvis_v1_categories_local", LVIS_CATEGORIES_PATH)
_module = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_module)
LVIS_CATEGORIES = _module.LVIS_CATEGORIES


TRAIN_TEMPLATES = (
    "a photo of a {}",
    "a photo of the {}",
)

HELDOUT_TEMPLATES = (
    "the {} in this image",
    "this is a {}",
    "one {}",
    "a small {}",
    "a large {}",
)


def sorted_lvis_categories() -> List[dict]:
    return sorted(LVIS_CATEGORIES, key=lambda x: x["id"])


def primary_name(cat: dict) -> str:
    return cat["synonyms"][0]


def alternate_synonyms(cat: dict) -> List[str]:
    return [syn for syn in cat["synonyms"][1:] if syn != cat["synonyms"][0]]


def definition_phrase(cat: dict) -> str:
    return f"an object that is {cat['def']}".strip()


def build_fixed_phrase_bank(categories: Sequence[dict]) -> Tuple[List[str], List[List[str]]]:
    class_names = []
    phrase_bank = []
    for cat in categories:
        name = primary_name(cat)
        alts = alternate_synonyms(cat)
        phrases = [
            name,
            TRAIN_TEMPLATES[0].format(name),
            TRAIN_TEMPLATES[1].format(name),
            alts[0] if len(alts) > 0 else name,
            alts[1] if len(alts) > 1 else (alts[0] if len(alts) > 0 else name),
            definition_phrase(cat),
        ]
        class_names.append(name)
        phrase_bank.append(phrases)
    return class_names, phrase_bank


def build_eval_queries(categories: Sequence[dict]) -> Dict[str, Tuple[List[str], List[int]]]:
    splits: Dict[str, Tuple[List[str], List[int]]] = {}

    template_texts, template_labels = [], []
    for label, cat in enumerate(categories):
        name = primary_name(cat)
        for tmpl in HELDOUT_TEMPLATES:
            template_texts.append(tmpl.format(name))
            template_labels.append(label)
    splits["heldout_templates"] = (template_texts, template_labels)

    synonym_texts, synonym_labels = [], []
    for label, cat in enumerate(categories):
        for syn in alternate_synonyms(cat):
            synonym_texts.append(syn)
            synonym_labels.append(label)
    splits["synonyms"] = (synonym_texts, synonym_labels)

    definition_texts, definition_labels = [], []
    for label, cat in enumerate(categories):
        definition_texts.append(definition_phrase(cat))
        definition_labels.append(label)
    splits["definitions"] = (definition_texts, definition_labels)

    return splits


def batched(iterable: Sequence[str], batch_size: int) -> Iterable[Sequence[str]]:
    for start in range(0, len(iterable), batch_size):
        yield iterable[start : start + batch_size]


def encode_texts(
    texts: Sequence[str],
    *,
    model,
    tokenizer,
    batch_size: int,
    device: str,
    desc: str,
) -> np.ndarray:
    all_feats = []
    with torch.no_grad():
        total = len(texts)
        for batch_idx, batch in enumerate(batched(list(texts), batch_size), start=1):
            tokens = tokenizer(list(batch)).to(device)
            feats = model.encode_text(tokens)
            feats = F.normalize(feats, p=2, dim=-1)
            all_feats.append(feats.cpu())
            if batch_idx == 1 or batch_idx % 10 == 0:
                done = min(batch_idx * batch_size, total)
                print(f"[encode] {desc}: {done}/{total}")
    return torch.cat(all_feats, dim=0).numpy().astype("float32", copy=False)


def compute_metrics(scores: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    order = np.argsort(-scores, axis=1)
    top1 = float((order[:, 0] == labels).mean())
    top5 = float(
        np.mean([label in order[idx, :5] for idx, label in enumerate(labels)])
    )
    ranks = np.argmax(order == labels[:, None], axis=1) + 1
    mrr = float(np.mean(1.0 / ranks))
    return {"top1": top1, "top5": top5, "mrr": mrr}


def score_single_bank(query_embeds: np.ndarray, class_bank: np.ndarray) -> np.ndarray:
    return query_embeds @ class_bank.T


def score_multi_bank(query_embeds: np.ndarray, phrase_bank: np.ndarray, agg: str) -> np.ndarray:
    scores = np.einsum("nd,ckd->nck", query_embeds, phrase_bank)
    if agg == "mean":
        return scores.mean(axis=-1)
    if agg == "max":
        return scores.max(axis=-1)
    if agg == "logsumexp":
        max_scores = scores.max(axis=-1, keepdims=True)
        stable = np.exp(scores - max_scores)
        return np.log(stable.sum(axis=-1)) + max_scores.squeeze(-1)
    raise ValueError(f"Unknown aggregation mode: {agg}")


def save_array(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, array.astype("float32", copy=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="ViT-B-32")
    parser.add_argument("--pretrained", default="openai")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/description_conditioned_pilot"),
    )
    parser.add_argument(
        "--metadata-output-dir",
        type=Path,
        default=Path("dataset/metadata/description_conditioned_pilot"),
    )
    parser.add_argument(
        "--max-classes",
        type=int,
        default=None,
        help="Optional cap for quick pilot runs.",
    )
    args = parser.parse_args()

    import open_clip

    model, _, _ = open_clip.create_model_and_transforms(args.model, pretrained=args.pretrained)
    tokenizer = open_clip.get_tokenizer(args.model)
    model = model.to(args.device)
    model.eval()

    categories = sorted_lvis_categories()
    if args.max_classes is not None:
        categories = categories[: args.max_classes]
    class_names, phrase_bank_text = build_fixed_phrase_bank(categories)
    eval_queries = build_eval_queries(categories)

    flat_phrase_text = [phrase for phrases in phrase_bank_text for phrase in phrases]
    class_name_embeds = encode_texts(
        class_names,
        model=model,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        device=args.device,
        desc="class-names",
    )
    phrase_embeds = encode_texts(
        flat_phrase_text,
        model=model,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        device=args.device,
        desc="phrase-bank",
    ).reshape(len(class_names), len(phrase_bank_text[0]), -1)

    save_array(args.metadata_output_dir / "lvis_class_name_bank.npy", class_name_embeds)
    save_array(args.metadata_output_dir / "lvis_phrase_bank.npy", phrase_embeds)
    (args.output_dir).mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "phrase_bank_preview.json").open("w") as f:
        json.dump(
            {
                "model": args.model,
                "pretrained": args.pretrained,
                "num_classes": len(class_names),
                "num_prototypes": len(phrase_bank_text[0]),
                "preview": {
                    class_names[idx]: phrase_bank_text[idx] for idx in range(5)
                },
            },
            f,
            indent=2,
        )

    rows = []
    for split_name, (texts, labels) in eval_queries.items():
        if not texts:
            continue
        query_embeds = encode_texts(
            texts,
            model=model,
            tokenizer=tokenizer,
            batch_size=args.batch_size,
            device=args.device,
            desc=f"queries:{split_name}",
        )
        labels_np = np.asarray(labels, dtype=np.int64)

        single_scores = score_single_bank(query_embeds, class_name_embeds)
        rows.append(
            {
                "split": split_name,
                "bank": "class_name_only",
                **compute_metrics(single_scores, labels_np),
                "num_queries": len(texts),
            }
        )

        for agg in ("mean", "max", "logsumexp"):
            scores = score_multi_bank(query_embeds, phrase_embeds, agg=agg)
            rows.append(
                {
                    "split": split_name,
                    "bank": f"phrase_bank_{agg}",
                    **compute_metrics(scores, labels_np),
                    "num_queries": len(texts),
                }
            )

    csv_path = args.output_dir / "prototype_pilot_results.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["split", "bank", "top1", "top5", "mrr", "num_queries"],
        )
        writer.writeheader()
        writer.writerows(rows)

    md_lines = [
        "# Description-Conditioned Prototype Pilot",
        "",
        f"- model: `{args.model}`",
        f"- pretrained: `{args.pretrained}`",
        f"- num_classes: `{len(class_names)}`",
        f"- num_prototypes_per_class: `{len(phrase_bank_text[0])}`",
        "",
        "| split | bank | top1 | top5 | mrr | num_queries |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        md_lines.append(
            f"| {row['split']} | {row['bank']} | {row['top1']:.4f} | {row['top5']:.4f} | {row['mrr']:.4f} | {row['num_queries']} |"
        )
    (args.output_dir / "prototype_pilot_results.md").write_text("\n".join(md_lines) + "\n")

    print("\n".join(md_lines))
    print(f"\nSaved CSV to {csv_path}")
    print(f"Saved phrase banks to {args.metadata_output_dir}")


if __name__ == "__main__":
    main()
