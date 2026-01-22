# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.


"""
This file registers pre-defined datasets at hard-coded paths, and their metadata.

We hard-code metadata for common datasets. This will enable:
1. Consistency check when loading the datasets
2. Use models on these standard datasets directly and run demos,
   without having to download the dataset annotations

We hard-code some paths to the dataset that's assumed to
exist in "./datasets/".

Users SHOULD NOT use this file to create new dataset / metadata for new dataset.
To add new dataset, refer to the tutorial "docs/DATASETS.md".
"""

import os

from detectron2.data import DatasetCatalog, MetadataCatalog

from .builtin_meta import ADE20K_SEM_SEG_CATEGORIES, _get_builtin_metadata
from .cityscapes import load_cityscapes_instances, load_cityscapes_semantic
from .cityscapes_panoptic import register_all_cityscapes_panoptic
from .coco import load_sem_seg, register_coco_instances
from .coco_panoptic import register_coco_panoptic, register_coco_panoptic_separated
from .lvis import get_lvis_instances_meta, register_lvis_instances
from .pascal_voc import register_pascal_voc
import json

# ==================================================================
# 直接修改原来的 COCO65 变量，放入 80 类全集
# ==================================================================
COCO80 = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
    "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop",
    "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush"
]

# 同时也必须修改对应的 ID 列表 (这是标准 COCO 的 1-90 ID，对应上面的 80 类)
COCO80_DATASET_IDS = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34,
    35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
    64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90
]
# 顺序必须与 COCO65 名称列表一一对应（你当前的 COCO65 顺序是对的）


# ==== Predefined datasets and splits for COCO ==========

_PREDEFINED_SPLITS_COCO = {}
_PREDEFINED_SPLITS_COCO["coco"] = {
    "coco_2014_train": ("coco/train2014", "coco/annotations/instances_train2014.json"),
    "coco_2014_val": ("coco/val2014", "coco/annotations/instances_val2014.json"),
    "coco_2014_minival": ("coco/val2014", "coco/annotations/instances_minival2014.json"),
    "coco_2014_valminusminival": (
        "coco/val2014",
        "coco/annotations/instances_valminusminival2014.json",
    ),
    "coco_2017_train": ("coco/train2017", "coco/annotations/instances_train2017.json"),
    "coco_2017_val": ("coco/val2017", "coco/annotations/instances_val2017.json"),
    "coco_2017_test": ("coco/test2017", "coco/annotations/image_info_test2017.json"),
    "coco_2017_test-dev": ("coco/test2017", "coco/annotations/image_info_test-dev2017.json"),
    "coco_2017_val_100": ("coco/val2017", "coco/annotations/instances_val2017_100.json"),
}

_PREDEFINED_SPLITS_COCO["coco"].update({

    # 标准 48/17：b=base(48)，t=novel(17)
    "ovcoco_2017_train_b":   ("coco/train2017", "coco/annotations/ovd_ins_train2017_b.json"),
    "ovcoco_2017_train_t":   ("coco/train2017", "coco/annotations/ovd_ins_train2017_t.json"),  # 一般不用训练
    "ovcoco_2017_val_b":     ("coco/val2017",   "coco/annotations/ovd_ins_val2017_b.json"),
    "ovcoco_2017_val_t":     ("coco/val2017",   "coco/annotations/ovd_ins_val2017_t.json"),
    "ovcoco_2017_val_all": ("coco/val2017", "coco/annotations1/instances_val2017.json"),
})

# _PREDEFINED_SPLITS_COCO["obj365v2"] = {
#     "obj365v2_train": ("object365/train/", "object365/annotations/obj365v2_train_filtered.json"),
#     "obj365v2_val": ("object365/val/", "object365/annotations/zhiyuan_objv2_val.json")
# }

_PREDEFINED_SPLITS_COCO["coco_zeroshot"] = {
    "zeroshot_coco_2017_train": ("coco/train2017", "coco/zero-shot/instances_train2017_seen_2_proposal.json"),
    # "zeroshot_coco_2017_train": ("coco/train2017", "coco/zero-shot/instances_train2017_seen.json"),
    # "zeroshot_coco_2017_train": ("coco/train2017", "coco/zero-shot/instances_train2017_all_2.json"),
    "zeroshot_coco_2017_val": ("coco/val2017", "coco/zero-shot/instances_val2017_all_2.json"),
    "zeroshot_coco_2017_val_unseen": ("coco/val2017", "coco/zero-shot/zeroshot_unseen.json"),
}
_PREDEFINED_SPLITS_COCO["coco_zeroshot_seen"] = {
    "zeroshot_coco_2017_train_seen": ("coco/train2017", "coco/zero-shot/instances_train2017_seen_2.json"),
    "zeroshot_coco_2017_val_seen": ("coco/val2017", "coco/zero-shot/instances_val2017_seen_2.json"),
}
_PREDEFINED_SPLITS_COCO["coco_zeroshot_unseen"] = {
    # "zeroshot_coco_2017_unseen": ("coco/val2017", "coco/zero-shot/zeroshot_seen.json"),
    "zeroshot_coco_2017_unseen": ("coco/val2017", "coco/zero-shot/instances_val2017_unseen_2.json"),
}

_PREDEFINED_SPLITS_COCO["coco_person"] = {
    "keypoints_coco_2014_train": (
        "coco/train2014",
        "coco/annotations/person_keypoints_train2014.json",
    ),
    "keypoints_coco_2014_val": ("coco/val2014", "coco/annotations/person_keypoints_val2014.json"),
    "keypoints_coco_2014_minival": (
        "coco/val2014",
        "coco/annotations/person_keypoints_minival2014.json",
    ),
    "keypoints_coco_2014_valminusminival": (
        "coco/val2014",
        "coco/annotations/person_keypoints_valminusminival2014.json",
    ),
    "keypoints_coco_2017_train": (
        "coco/train2017",
        "coco/annotations/person_keypoints_train2017.json",
    ),
    "keypoints_coco_2017_val": ("coco/val2017", "coco/annotations/person_keypoints_val2017.json"),
    "keypoints_coco_2017_val_100": (
        "coco/val2017",
        "coco/annotations/person_keypoints_val2017_100.json",
    ),
}


_PREDEFINED_SPLITS_COCO_PANOPTIC = {
    "coco_2017_train_panoptic": (
        # This is the original panoptic annotation directory
        "coco/panoptic_train2017",
        "coco/annotations/panoptic_train2017.json",
        # This directory contains semantic annotations that are
        # converted from panoptic annotations.
        # It is used by PanopticFPN.
        # You can use the script at detectron2/datasets/prepare_panoptic_fpn.py
        # to create these directories.
        "coco/panoptic_stuff_train2017",
    ),
    "coco_2017_val_panoptic": (
        "coco/panoptic_val2017",
        "coco/annotations/panoptic_val2017.json",
        "coco/panoptic_stuff_val2017",
    ),
    "coco_2017_val_100_panoptic": (
        "coco/panoptic_val2017_100",
        "coco/annotations/panoptic_val2017_100.json",
        "coco/panoptic_stuff_val2017_100",
    ),
}


def _safe_set_thing_classes(dataset_key, classes, force=False):
    """仅当尚未设置过 thing_classes 时设置，避免触发 D2 断言。"""
    meta = MetadataCatalog.get(dataset_key)
    old = getattr(meta, "thing_classes", None)
    if force or old is None:
        meta.thing_classes = classes
    elif list(old) != list(classes):
        import logging
        from detectron2.utils.logger import log_first_n
        log_first_n(
            logging.WARN,
            f"[ovcoco] keep existing thing_classes for '{dataset_key}' (len={len(old)}), "
            f"skip replacing with len={len(classes)}.",
            n=1
        )

def _ovcoco_build_id_map(json_path):
    # """
    # 生成 原始 category_id -> 连续 id(0..64) 的映射。
    # 1) 若 JSON 含 categories：按名字与 COCO65 对齐生成映射（更稳，能校验名字拼写）
    # 2) 若不含 categories：退回固定的 COCO65_DATASET_IDS 常量
    # """
    # with open(json_path, "r") as f:
    #     data = json.load(f)

    # cats = data.get("categories", [])
    # if cats:  # 路径1：严格校验名字
    #     cat_id_to_name = {c["id"]: c["name"] for c in cats}
    #     name_to_contig = {n: i for i, n in enumerate(COCO65)}
    #     id_map, missing = {}, []
    #     for k, v in cat_id_to_name.items():
    #         if v not in name_to_contig:
    #             missing.append((k, v))
    #         else:
    #             id_map[k] = name_to_contig[v]
    #     if missing:
    #         raise ValueError(
    #             "[OVD-COCO] 标注中存在 COCO65 之外或名称不匹配的类别：\n"
    #             + "\n".join([f"  id={k}, name='{v}'" for k, v in missing])
    #             + "\n请确保类别名与 COCO65 完全一致（大小写/空格/连字符）。"
    #         )
    #     return id_map

    # # 路径2：无 categories，用固定 ID 列表兜底
    return {did: i for i, did in enumerate(COCO65_DATASET_IDS)}


def register_all_coco(root):
    for dataset_name, splits_per_dataset in _PREDEFINED_SPLITS_COCO.items():
        for key, (image_root, json_file) in splits_per_dataset.items():
            jf = os.path.join(root, json_file) if "://" not in json_file else json_file
            ir = os.path.join(root, image_root)

            if key.startswith("ovcoco_2017_"):
                register_coco_instances(key, {}, jf, ir)
                meta = MetadataCatalog.get(key)
                meta.evaluator_type = "coco"

                # 统一：所有 split 都用同一套 80类定义
                _safe_set_thing_classes(key, COCO65, force=True)  # 你这里 COCO65=COCO80类名
                meta.thing_dataset_id_to_contiguous_id = _ovcoco_build_id_map()

                continue


            # 其它 COCO 正常 split：沿用内置 meta
            register_coco_instances(
                key,
                _get_builtin_metadata(dataset_name),
                jf,
                ir,
            )

    # panoptic 保持原样
    for (
        prefix,
        (panoptic_root, panoptic_json, semantic_root),
    ) in _PREDEFINED_SPLITS_COCO_PANOPTIC.items():
        prefix_instances = prefix[: -len("_panoptic")]
        instances_meta = MetadataCatalog.get(prefix_instances)
        image_root, instances_json = instances_meta.image_root, instances_meta.json_file

        register_coco_panoptic_separated(
            prefix,
            _get_builtin_metadata("coco_panoptic_separated"),
            image_root,
            os.path.join(root, panoptic_root),
            os.path.join(root, panoptic_json),
            os.path.join(root, semantic_root),
            instances_json,
        )
        register_coco_panoptic(
            prefix,
            _get_builtin_metadata("coco_panoptic_standard"),
            image_root,
            os.path.join(root, panoptic_root),
            os.path.join(root, panoptic_json),
            instances_json,
        )



# ==== Predefined datasets and splits for LVIS ==========


_PREDEFINED_SPLITS_LVIS = {
    "lvis_v1": {
        "lvis_v1_train": ("coco/", "lvis/lvis_v1_train.json"),
        "lvis_v1_train_norare": ("coco/", "lvis/lvis_v1_train_norare.json"),
        "lvis_v1_val": ("coco/", "lvis/lvis_v1_val.json"),
        "lvis_v1_minival": ("coco/", "lvis/lvis_v1_minival.json"),
        "lvis_v1_test_dev": ("coco/", "lvis/lvis_v1_image_info_test_dev.json"),
        "lvis_v1_test_challenge": ("coco/", "lvis/lvis_v1_image_info_test_challenge.json"),
    },
    "lvis_v0.5": {
        "lvis_v0.5_train": ("coco/", "lvis/lvis_v0.5_train.json"),
        "lvis_v0.5_val": ("coco/", "lvis/lvis_v0.5_val.json"),
        "lvis_v0.5_val_rand_100": ("coco/", "lvis/lvis_v0.5_val_rand_100.json"),
        "lvis_v0.5_test": ("coco/", "lvis/lvis_v0.5_image_info_test.json"),
    },
    "lvis_v0.5_cocofied": {
        "lvis_v0.5_train_cocofied": ("coco/", "lvis/lvis_v0.5_train_cocofied.json"),
        "lvis_v0.5_val_cocofied": ("coco/", "lvis/lvis_v0.5_val_cocofied.json"),
    },
}


def register_all_lvis(root):
    for dataset_name, splits_per_dataset in _PREDEFINED_SPLITS_LVIS.items():
        for key, (image_root, json_file) in splits_per_dataset.items():
            register_lvis_instances(
                key,
                get_lvis_instances_meta(dataset_name),
                os.path.join(root, json_file) if "://" not in json_file else json_file,
                os.path.join(root, image_root),
            )


# ==== Predefined splits for raw cityscapes images ===========
_RAW_CITYSCAPES_SPLITS = {
    "cityscapes_fine_{task}_train": ("cityscapes/leftImg8bit/train/", "cityscapes/gtFine/train/"),
    "cityscapes_fine_{task}_val": ("cityscapes/leftImg8bit/val/", "cityscapes/gtFine/val/"),
    "cityscapes_fine_{task}_test": ("cityscapes/leftImg8bit/test/", "cityscapes/gtFine/test/"),
}


def register_all_cityscapes(root):
    for key, (image_dir, gt_dir) in _RAW_CITYSCAPES_SPLITS.items():
        meta = _get_builtin_metadata("cityscapes")
        image_dir = os.path.join(root, image_dir)
        gt_dir = os.path.join(root, gt_dir)

        inst_key = key.format(task="instance_seg")
        DatasetCatalog.register(
            inst_key,
            lambda x=image_dir, y=gt_dir: load_cityscapes_instances(
                x, y, from_json=True, to_polygons=True
            ),
        )
        MetadataCatalog.get(inst_key).set(
            image_dir=image_dir, gt_dir=gt_dir, evaluator_type="cityscapes_instance", **meta
        )

        sem_key = key.format(task="sem_seg")
        DatasetCatalog.register(
            sem_key, lambda x=image_dir, y=gt_dir: load_cityscapes_semantic(x, y)
        )
        MetadataCatalog.get(sem_key).set(
            image_dir=image_dir,
            gt_dir=gt_dir,
            evaluator_type="cityscapes_sem_seg",
            ignore_label=255,
            **meta,
        )


# ==== Predefined splits for PASCAL VOC ===========
def register_all_pascal_voc(root):
    SPLITS = [
        ("voc_2007_trainval", "VOC2007", "trainval"),
        ("voc_2007_train", "VOC2007", "train"),
        ("voc_2007_val", "VOC2007", "val"),
        ("voc_2007_test", "VOC2007", "test"),
        ("voc_2012_trainval", "VOC2012", "trainval"),
        ("voc_2012_train", "VOC2012", "train"),
        ("voc_2012_val", "VOC2012", "val"),
    ]
    for name, dirname, split in SPLITS:
        year = 2007 if "2007" in name else 2012
        register_pascal_voc(name, os.path.join(root, dirname), split, year)
        MetadataCatalog.get(name).evaluator_type = "pascal_voc"


def register_all_ade20k(root):
    root = os.path.join(root, "ADEChallengeData2016")
    for name, dirname in [("train", "training"), ("val", "validation")]:
        image_dir = os.path.join(root, "images", dirname)
        gt_dir = os.path.join(root, "annotations_detectron2", dirname)
        name = f"ade20k_sem_seg_{name}"
        DatasetCatalog.register(
            name, lambda x=image_dir, y=gt_dir: load_sem_seg(y, x, gt_ext="png", image_ext="jpg")
        )
        MetadataCatalog.get(name).set(
            stuff_classes=ADE20K_SEM_SEG_CATEGORIES[:],
            image_root=image_dir,
            sem_seg_root=gt_dir,
            evaluator_type="sem_seg",
            ignore_label=255,
        )

_PREDEFINED_SPLITS_ANP = {
    "ANP_traffic": {
    "ANP_traffic_train": ("ANP_traffic/", "ANP_traffic/annotations/anp_train.json"),
    "ANP_traffic_finetune": ("ANP_traffic/", "ANP_traffic/annotations/anp_train_filter_unknown.json"),
    "ANP_traffic_val": ("ANP_traffic/JIDU_small_testset/", "ANP_traffic/annotations/anp_val_allcone.json"),
    }
}

def register_all_anp(root):
    for dataset_name, splits_per_dataset in _PREDEFINED_SPLITS_ANP.items():
        for key, (image_root, json_file) in splits_per_dataset.items():
            register_coco_instances(
                key,
                _get_builtin_metadata(dataset_name),
                os.path.join(root, json_file) if "://" not in json_file else json_file,
                os.path.join(root, image_root),
            )

_PREDEFINED_SPLITS_VG_PSEUDO = {
    "vg_pseudo": {
        "vg_filter_rare": ("VisualGenome/images", "VisualGenome/vg_filter_rare.json",)
    }
}


def register_all_vg(root):
    for dataset_name, splits_per_dataset in _PREDEFINED_SPLITS_VG_PSEUDO.items():
        for key, (image_root, json_file) in splits_per_dataset.items():
            # Assume pre-defined datasets live in `./datasets`.
            register_coco_instances(
                key,
                _get_builtin_metadata(dataset_name),
                os.path.join(root, json_file) if "://" not in json_file else json_file,
                os.path.join(root, image_root),
                extra_annotation_keys=["score"],
            )


# True for open source;
# Internally at fb, we register them elsewhere
if __name__.endswith(".builtin"):
    # Assume pre-defined datasets live in `./datasets`.
    _root = os.path.expanduser(os.getenv("DETECTRON2_DATASETS", "dataset"))
    register_all_coco(_root)
    register_all_lvis(_root)
    register_all_cityscapes(_root)
    register_all_cityscapes_panoptic(_root)
    register_all_pascal_voc(_root)
    register_all_ade20k(_root)
    register_all_anp(_root)
    register_all_vg(_root)
