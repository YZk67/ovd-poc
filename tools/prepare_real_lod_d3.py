#!/usr/bin/env python3
"""Prepare Real-LOD/Real-Model D3 eval and adapter configs.

This helper mirrors the GroundingDINO Route-A preparation script, but targets
the official Real-LOD repository. The first milestone is reproducing the
published Real-Model D3 result from the official checkpoint. Supplying
``--train-ann-file`` additionally writes a conservative text-query-adapter
training config initialized from that checkpoint.
"""

from __future__ import annotations

import argparse
import json
import stat
from pathlib import Path
from typing import Any, Dict


REAL_MODEL_B_CKPT = (
    "https://huggingface.co/datasets/fishandwasabi/Real-LOD-Data/resolve/main/"
    "real-model-ckpts/real-model_b-357a96d2.pth"
)

TEXT_ADAPTER_TRAINABLE_PREFIXES = (
    "text_query_adapter",
    "dn_query_generator.label_embedding",
)
DN_ONLY_TRAINABLE_PREFIXES = ("dn_query_generator.label_embedding",)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _quote(value: str | Path) -> str:
    return repr(str(value))


def _quote_tuple(values: tuple[str, ...]) -> str:
    return "(" + ", ".join(_quote(value) for value in values) + ("," if len(values) == 1 else "") + ")"


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _resolved_path(path: Path) -> Path:
    return path.expanduser().resolve(strict=False)


def _validate_train_val_split(args: argparse.Namespace) -> None:
    if args.train_ann_file is None:
        return
    val_ann_file = args.val_ann_file or args.d3_ann_file
    train_path = _resolved_path(args.train_ann_file)
    val_path = _resolved_path(val_ann_file)
    if train_path == val_path and not args.allow_train_val_same:
        raise ValueError(
            "Training annotation and validation annotation resolve to the same file: "
            f"{train_path}. This is train-on-test for D3 adaptation. Pass a disjoint "
            "--train-ann-file/--val-ann-file pair, or add --allow-train-val-same only "
            "for an intentional smoke/debug run."
        )


def _make_test_cfg(args: argparse.Namespace) -> str:
    items = [f"max_per_img={args.max_per_img}"]
    if args.inference_mode == "parallel":
        items.append(f"chunked_size={args.chunked_size}")
    return f"dict({', '.join(items)})"


def _make_model_train_cfg() -> str:
    return """dict(
        _delete_=True,
        assigner=dict(
            type='HungarianAssigner',
            match_costs=[
                dict(type='BinaryFocalLossCost', weight=2.0),
                dict(type='BBoxL1Cost', weight=5.0, box_format='xywh'),
                dict(type='IoUCost', iou_mode='giou', weight=2.0),
            ],
        )
    )"""


def _make_config(args: argparse.Namespace) -> str:
    base_config = Path(args.real_lod_root) / args.base_config
    lang_model_name = args.bert_path or "bert-base-uncased"
    test_cfg = _make_test_cfg(args)
    is_train = args.train_ann_file is not None
    dataset_type = "D3SubsetDODDataset" if is_train or args.use_subset_dataset else "DODDataset"
    val_ann_file = args.val_ann_file or args.d3_ann_file
    model_train_cfg = _make_model_train_cfg() if is_train else "None"
    bbox_head_cfg = (
        "dict(loss_iou=dict(type='GIoULoss', loss_weight=2.0))"
        if is_train
        else "dict()"
    )
    default_trainable_prefixes = (
        TEXT_ADAPTER_TRAINABLE_PREFIXES
        if args.adapter_mode in {"text", "text_negdn"}
        else DN_ONLY_TRAINABLE_PREFIXES
    )
    trainable_prefixes = tuple(args.trainable_prefix or default_trainable_prefixes)
    freeze_selected_params = is_train and args.trainable_scope == "selected"
    training_note = (
        "# Adapter training keeps Real-Model frozen except the selected trainable prefixes."
        if freeze_selected_params
        else "# Full-model fine-tuning leaves all Real-Model parameters trainable."
        if is_train
        else "# Eval config targets the official Real-LOD/Real-Model Swin-B D3 setup."
    )
    custom_hooks_cfg = (
        """[
    dict(
        type='TrainableParamFreezeHook',
        trainable_prefixes=""" + _quote_tuple(trainable_prefixes) + """,
    )
]"""
        if freeze_selected_params
        else "[]"
    )
    custom_imports = (
        "custom_imports = dict(\n"
        "    imports=['tools.mmdet_d3_route_a', 'tools.real_lod_route_a'],\n"
        "    allow_failed_imports=False,\n"
        ")\n\n"
        if is_train or args.use_subset_dataset
        else ""
    )
    if is_train and args.adapter_mode == "text":
        model_type = (
            "    type='RealModelTextQueryAdapter',\n"
            f"    text_query_adapter=dict(bottleneck_dim={args.adapter_bottleneck_dim}, "
            f"dropout={args.adapter_dropout}, init_scale={args.adapter_init_scale}),\n"
        )
    elif is_train and args.adapter_mode == "text_negdn":
        model_type = (
            "    type='RealModelTextQueryAdapterNegDN',\n"
            f"    text_query_adapter=dict(bottleneck_dim={args.adapter_bottleneck_dim}, "
            f"dropout={args.adapter_dropout}, init_scale={args.adapter_init_scale}),\n"
            f"    negative_dn=dict(same_region_box_noise_scale={args.negative_dn_box_noise}),\n"
        )
    elif is_train:
        model_type = "    type='RealModelD3PromptWrapper',\n"
    else:
        model_type = ""
    load_from = f"load_from = {_quote(args.checkpoint)}\n" if is_train else ""
    subset_meta_keys = (
        "            'sent_group_ids',\n"
        if is_train or args.use_subset_dataset
        else ""
    )

    return f"""# Auto-generated by LaMI-DETR tools/prepare_real_lod_d3.py.
{training_note}

_base_ = {_quote(base_config)}

{custom_imports}\
auto_scale_lr = dict(enable=False)
lang_model_name = {_quote(lang_model_name)}
{load_from}

model = dict(
{model_type}\
    language_model=dict(name=lang_model_name),
    bbox_head={bbox_head_cfg},
    train_cfg={model_train_cfg},
    test_cfg={test_cfg},
)

d3_ann_file = {_quote(args.d3_ann_file)}
d3_train_ann_file = {_quote(args.train_ann_file or args.d3_ann_file)}
d3_val_ann_file = {_quote(val_ann_file)}
d3_image_root = {_quote(args.d3_image_root)}
d3_pkl_root = {_quote(args.d3_pkl_root)}
dataset_type = {_quote(dataset_type)}

test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None, imdecode_backend='pillow'),
    dict(type='FixScaleResize', scale=({args.test_short_edge}, {args.test_long_edge}), keep_ratio=True, backend='pillow'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=(
            'img_id',
            'img_path',
            'ori_shape',
            'img_shape',
            'scale_factor',
            'text',
            'custom_entities',
            'tokens_positive',
            'sent_ids',
{subset_meta_keys}\
        ),
    ),
]

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None, imdecode_backend='pillow'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='FixScaleResize', scale=({args.test_short_edge}, {args.test_long_edge}), keep_ratio=True, backend='pillow'),
    dict(
        type='PackDetInputs',
        meta_keys=(
            'img_id',
            'img_path',
            'ori_shape',
            'img_shape',
            'scale_factor',
            'text',
            'custom_entities',
            'tokens_positive',
            'sent_ids',
{subset_meta_keys}\
        ),
    ),
]

test_dataloader = dict(
    _delete_=True,
    batch_size={args.batch_size},
    num_workers={args.num_workers},
    persistent_workers=False,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root='',
        ann_file=d3_val_ann_file if {is_train} else d3_ann_file,
        data_prefix=dict(img=d3_image_root, anno=d3_pkl_root),
        test_mode=True,
        pipeline=test_pipeline,
        return_classes=True,
    ),
)
val_dataloader = test_dataloader

train_dataloader = {'''dict(
    _delete_=True,
    batch_size=''' + str(args.batch_size) + ''',
    num_workers=''' + str(args.num_workers) + ''',
    persistent_workers=False,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root='',
        ann_file=d3_train_ann_file,
        data_prefix=dict(img=d3_image_root, anno=d3_pkl_root),
        test_mode=False,
        pipeline=train_pipeline,
        return_classes=True,
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
    ),
)''' if is_train else 'None'}

test_evaluator = dict(
    _delete_=True,
    type='DODCocoMetric',
    ann_file=d3_val_ann_file if {is_train} else d3_ann_file,
)
val_evaluator = test_evaluator

train_cfg = {f"dict(_delete_=True, type='IterBasedTrainLoop', max_iters={args.max_iter}, val_interval={args.val_interval})" if is_train else 'None'}
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

optim_wrapper = {'''dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=''' + str(args.train_lr) + ''', weight_decay=''' + str(args.weight_decay) + '''),
    clip_grad=dict(max_norm=0.1, norm_type=2),
)''' if is_train else 'None'}
param_scheduler = {'[]' if is_train else 'None'}

custom_hooks = {custom_hooks_cfg}

default_hooks = dict(
    checkpoint=dict(_delete_=True, type='CheckpointHook', by_epoch=False, interval={args.checkpoint_interval}, max_keep_ckpts=3),
    logger=dict(type='LoggerHook', interval={args.log_interval}),
)
log_processor = dict(type='LogProcessor', window_size=50, by_epoch=False)
"""


def _make_run_script(args: argparse.Namespace, config_path: Path) -> str:
    repo_root = Path(args.repo_root).resolve()
    if args.train_ann_file is not None:
        command = 'python tools/train_real_model.py "$CONFIG" --work-dir "$WORK_DIR"'
    else:
        command = 'python tools/test_real_model.py "$CONFIG" "$CHECKPOINT" --work-dir "$WORK_DIR"'
    return f"""#!/usr/bin/env bash
set -euo pipefail

REAL_LOD_DIR={_quote(args.real_lod_root)}
CONFIG={_quote(config_path)}
CHECKPOINT={_quote(args.checkpoint)}
WORK_DIR={_quote(args.work_dir)}
REPO_ROOT={_quote(repo_root)}

cd "$REAL_LOD_DIR"
mkdir -p "$WORK_DIR"
export PYTHONPATH="$REPO_ROOT:$REAL_LOD_DIR:${{PYTHONPATH:-}}"

{command}
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real-lod-root", type=Path, required=True)
    parser.add_argument("--d3-ann-file", type=Path, required=True)
    parser.add_argument("--d3-image-root", type=Path, required=True)
    parser.add_argument("--d3-pkl-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, default=Path("work_dirs/real_lod_d3"))
    parser.add_argument("--checkpoint", default=REAL_MODEL_B_CKPT)
    parser.add_argument(
        "--base-config",
        default="configs/real_model/ours/dod/real-model_swin-b_realdata_dod.py",
    )
    parser.add_argument("--bert-path", default=None)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument("--train-ann-file", type=Path, default=None)
    parser.add_argument("--val-ann-file", type=Path, default=None)
    parser.add_argument(
        "--allow-train-val-same",
        action="store_true",
        help=(
            "Allow training and validation annotations to be the same file. "
            "Use only for intentional smoke/debug runs; paper-facing D3 adaptation "
            "must use disjoint train/val annotations."
        ),
    )
    parser.add_argument("--use-subset-dataset", action="store_true")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-per-img", type=int, default=300)
    parser.add_argument(
        "--inference-mode",
        choices=("concat", "parallel"),
        default="parallel",
    )
    parser.add_argument("--chunked-size", type=int, default=1)
    parser.add_argument("--max-iter", type=int, default=750)
    parser.add_argument("--val-interval", type=int, default=250)
    parser.add_argument("--checkpoint-interval", type=int, default=250)
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--train-lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument(
        "--trainable-scope",
        choices=("selected", "all"),
        default="selected",
        help=(
            "'selected' freezes Real-Model and trains only adapter/DN prefixes "
            "(Stage 1 default); 'all' disables the freeze hook for Stage 2 "
            "same-recipe full-model fine-tuning."
        ),
    )
    parser.add_argument(
        "--adapter-mode",
        choices=("text", "none", "text_negdn"),
        default="text",
        help=(
            "'none' runs the DN-label-only control; 'text_negdn' adds "
            "in-image wrong-phrase DN negatives on top of the text adapter."
        ),
    )
    parser.add_argument("--adapter-bottleneck-dim", type=int, default=64)
    parser.add_argument("--adapter-dropout", type=float, default=0.0)
    parser.add_argument("--adapter-init-scale", type=float, default=1.0)
    parser.add_argument("--negative-dn-box-noise", type=float, default=0.25)
    parser.add_argument("--trainable-prefix", action="append", default=None)
    parser.add_argument("--test-short-edge", type=int, default=800)
    parser.add_argument("--test-long-edge", type=int, default=1333)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _validate_train_val_split(args)
    is_train = args.train_ann_file is not None
    config_name = "real_model_swin-b_d3_adapter_train.py" if is_train else "real_model_swin-b_d3_eval.py"
    run_name = "run_train.sh" if is_train else "run_eval.sh"
    config_path = args.output_dir / config_name
    run_path = args.output_dir / run_name
    manifest_path = args.output_dir / "manifest.json"

    _write_text(config_path, _make_config(args))
    _write_text(run_path, _make_run_script(args, config_path.resolve()))
    run_path.chmod(run_path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

    manifest: Dict[str, Any] = {
        "purpose": (
            "Route A full-model fine-tuning: Real-LOD Real-Model on D3"
            if is_train and args.trainable_scope == "all"
            else "Route A adapter training: Real-LOD Real-Model on D3"
            if is_train
            else "Route A baseline: Real-LOD Real-Model on D3"
        ),
        "config": str(config_path.resolve()),
        "run_script": str(run_path.resolve()),
        "train_mode": is_train,
        "args": {key: _jsonable(value) for key, value in vars(args).items()},
        "expected_reference": {
            "source": "Real-LOD / Real-Model D3 full reported result",
            "real_model_b_full_parallel": 34.1,
        },
    }
    _write_text(manifest_path, json.dumps(manifest, indent=2, sort_keys=True))
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
