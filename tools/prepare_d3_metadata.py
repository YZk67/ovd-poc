#!/usr/bin/env python3
"""
Prepare D3/D-Cube sentence metadata for fixed-bank detection.

This script supports the two formats released by D3:

1. COCO-style evaluation JSON from ``d3_json.zip``.
2. Toolkit pickle annotations from ``d3_pkl.zip``.

For Level-0 fixed-bank evaluation, the important output is an ordered phrase
list where row ``i`` corresponds to Detectron2 contiguous class id ``i`` after
sorting D3 ``sent_id`` values. D3 sent ids are expected to be 1-based, so this is
normally row 0 -> sent_id 1, ..., row 421 -> sent_id 422.
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r") as f:
        return json.load(f)


def _load_pickle(path: Path) -> Dict[int, Dict[str, Any]]:
    with path.open("rb") as f:
        return pickle.load(f)


def _to_builtin(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return _to_builtin(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, bytes):
        return value.decode("ascii")
    if isinstance(value, dict):
        return {k: _to_builtin(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_builtin(v) for v in value]
    return value


def _first_scalar(value: Any) -> Any:
    value = _to_builtin(value)
    if isinstance(value, list) and len(value) == 1:
        return value[0]
    return value


def _select_sentence_ids(
    sentences: Mapping[int, Mapping[str, Any]],
    split: str,
) -> List[int]:
    sent_ids = sorted(int(sent_id) for sent_id in sentences.keys())
    if split == "full":
        return sent_ids
    if split == "pres":
        return [
            sent_id
            for sent_id in sent_ids
            if not bool(sentences[sent_id].get("is_negative", False))
        ]
    if split == "abs":
        return [
            sent_id
            for sent_id in sent_ids
            if bool(sentences[sent_id].get("is_negative", False))
        ]
    raise ValueError(f"Unknown split {split!r}; expected full, pres, or abs.")


def _sentence_records_from_coco(coco: Mapping[str, Any]) -> List[Dict[str, Any]]:
    categories = coco.get("categories")
    if not categories:
        raise ValueError("COCO JSON does not contain a non-empty 'categories' list.")

    records = []
    for cat in sorted(categories, key=lambda item: int(item["id"])):
        text = cat.get("raw_sent", cat.get("name", cat.get("phrase")))
        if not text:
            raise ValueError(f"Category {cat!r} does not contain name/raw_sent/phrase text.")
        records.append(
            {
                "id": int(cat["id"]),
                "raw_sent": str(text),
                "is_negative": bool(cat.get("is_negative", False)),
            }
        )
    return records


def _sentence_records_from_pkl(pkl_root: Path, split: str) -> List[Dict[str, Any]]:
    sentences = _load_pickle(pkl_root / "sentences.pkl")
    selected_ids = _select_sentence_ids(sentences, split)
    records = []
    for sent_id in selected_ids:
        sent = sentences[sent_id]
        records.append(
            {
                "id": int(sent["id"]),
                "raw_sent": str(sent["raw_sent"]),
                "raw_sent_zh": str(sent.get("raw_sent_zh", "")),
                "is_negative": bool(sent.get("is_negative", False)),
                "group_id": _to_builtin(sent.get("group_id", [])),
                "anno_id": _to_builtin(sent.get("anno_id", [])),
            }
        )
    return records


def _write_sentence_outputs(
    records: Sequence[Mapping[str, Any]],
    metadata_output: Path,
    phrases_output: Path,
) -> None:
    metadata_output.parent.mkdir(parents=True, exist_ok=True)
    phrases_output.parent.mkdir(parents=True, exist_ok=True)

    with metadata_output.open("w") as f:
        json.dump(list(records), f, indent=2)

    phrases = [str(record["raw_sent"]) for record in records]
    with phrases_output.open("w") as f:
        json.dump(phrases, f, indent=2)


def _normalize_segmentation(segmentation: Any) -> Any:
    seg = _first_scalar(segmentation)
    if isinstance(seg, dict):
        seg = dict(seg)
        counts = seg.get("counts")
        if isinstance(counts, bytes):
            seg["counts"] = counts.decode("ascii")
    return _to_builtin(seg)


def _sentence_ids_for_annotation(
    annotation: Mapping[str, Any],
    groups: Mapping[int, Mapping[str, Any]],
    *,
    setting: str,
) -> Iterable[int]:
    sent_ids = [int(sent_id) for sent_id in _to_builtin(annotation.get("sent_id", []))]
    if setting == "inter":
        return sent_ids
    if setting == "intra":
        group_id = int(_first_scalar(annotation["group_id"]))
        inner_ids = set(int(sent_id) for sent_id in groups[group_id]["inner_sent_id"])
        return [sent_id for sent_id in sent_ids if sent_id in inner_ids]
    raise ValueError(f"Unknown setting {setting!r}; expected inter or intra.")


def _convert_pkl_to_coco(
    pkl_root: Path,
    coco_output: Path,
    *,
    split: str,
    setting: str,
) -> None:
    sentences = _load_pickle(pkl_root / "sentences.pkl")
    annotations = _load_pickle(pkl_root / "annotations.pkl")
    images = _load_pickle(pkl_root / "images.pkl")
    groups = _load_pickle(pkl_root / "groups.pkl")

    selected_sent_ids = set(_select_sentence_ids(sentences, split))
    coco: Dict[str, Any] = {
        "images": [],
        "annotations": [],
        "categories": [],
    }

    for sent_id in sorted(selected_sent_ids):
        sent = sentences[sent_id]
        coco["categories"].append(
            {
                "id": int(sent["id"]),
                "name": str(sent["raw_sent"]),
                "is_negative": bool(sent.get("is_negative", False)),
            }
        )

    for image_id in sorted(int(img_id) for img_id in images.keys()):
        image = images[image_id]
        coco["images"].append(
            {
                "id": int(image["id"]),
                "file_name": str(image["file_name"]),
                "height": int(image["height"]),
                "width": int(image["width"]),
            }
        )

    output_ann_id = 0
    for ann_id in sorted(int(key) for key in annotations.keys()):
        ann = annotations[ann_id]
        for sent_id in _sentence_ids_for_annotation(ann, groups, setting=setting):
            if sent_id not in selected_sent_ids:
                continue
            coco["annotations"].append(
                {
                    "id": output_ann_id,
                    "image_id": int(ann["image_id"]),
                    "category_id": int(sent_id),
                    "segmentation": _normalize_segmentation(ann.get("segmentation", [])),
                    "area": int(_first_scalar(ann["area"])),
                    "bbox": [
                        float(coord)
                        for coord in _to_builtin(_first_scalar(ann["bbox"]))
                    ],
                    "iscrowd": int(_first_scalar(ann.get("iscrowd", 0))),
                }
            )
            output_ann_id += 1

    coco_output.parent.mkdir(parents=True, exist_ok=True)
    with coco_output.open("w") as f:
        json.dump(coco, f)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--coco-json",
        type=Path,
        help="D3 COCO-style annotation JSON. Categories are used as sent_id phrases.",
    )
    source.add_argument(
        "--pkl-root",
        type=Path,
        help="Directory containing D3 sentences.pkl, annotations.pkl, images.pkl, groups.pkl.",
    )
    parser.add_argument(
        "--split",
        choices=("full", "pres", "abs"),
        default="full",
        help="Sentence split to export. pres/abs require pkl metadata with is_negative.",
    )
    parser.add_argument(
        "--setting",
        choices=("inter", "intra"),
        default="inter",
        help="COCO conversion setting when --coco-output is used with --pkl-root.",
    )
    parser.add_argument(
        "--metadata-output",
        type=Path,
        default=Path("dataset/metadata/d3_sentences.json"),
        help="Output JSON with sentence ids and metadata.",
    )
    parser.add_argument(
        "--phrases-output",
        type=Path,
        default=Path("dataset/metadata/d3_phrases.json"),
        help="Output JSON list of ordered English phrases for text embedding generation.",
    )
    parser.add_argument(
        "--coco-output",
        type=Path,
        default=None,
        help="Optional output COCO annotation JSON. Only supported with --pkl-root.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.coco_json:
        coco = _load_json(args.coco_json)
        records = _sentence_records_from_coco(coco)
        if args.split != "full":
            raise ValueError("--split pres/abs requires --pkl-root because COCO categories may not carry is_negative.")
        if args.coco_output:
            raise ValueError("--coco-output is only supported with --pkl-root.")
    else:
        records = _sentence_records_from_pkl(args.pkl_root, args.split)
        if args.coco_output:
            _convert_pkl_to_coco(
                args.pkl_root,
                args.coco_output,
                split=args.split,
                setting=args.setting,
            )
            print(f"Saved COCO annotation JSON to {args.coco_output}")

    _write_sentence_outputs(records, args.metadata_output, args.phrases_output)
    print(f"Saved sentence metadata to {args.metadata_output}")
    print(f"Saved ordered phrase list to {args.phrases_output}")
    print(f"num_sentences: {len(records)}")
    if records:
        print(f"first: sent_id={records[0]['id']} text={records[0]['raw_sent']!r}")
        print(f"last: sent_id={records[-1]['id']} text={records[-1]['raw_sent']!r}")


if __name__ == "__main__":
    main()
