"""D3 dataset helpers for MMDetection Route-A experiments."""

from __future__ import annotations

import os.path as osp
from typing import List

import numpy as np
from mmdet.datasets.api_wrappers import COCO
from mmdet.datasets.dod import DODDataset
from mmdet.registry import DATASETS


def _coco_img_ids(coco: COCO) -> set[int]:
    if hasattr(coco, "get_img_ids"):
        return set(coco.get_img_ids())
    return set(coco.getImgIds())


@DATASETS.register_module()
class D3SubsetDODDataset(DODDataset):
    """DODDataset variant that respects the image ids in ``ann_file``.

    OpenMMLab's DODDataset iterates over all image ids from the D3 pkl. That is
    correct for official full eval, but not for local train/val splits stored as
    COCO JSON files. This subclass filters images by the JSON and maps global D3
    sentence category ids to image-local phrase labels for GroundingDINO loss.
    """

    def load_data_list(self) -> List[dict]:
        coco = COCO(self.ann_file)
        allowed_img_ids = _coco_img_ids(coco)
        data_list: List[dict] = []

        for img_id in self.d3.get_img_ids():
            if int(img_id) not in allowed_img_ids:
                continue

            img_info = self.d3.load_imgs(img_id)[0]
            group_ids = self.d3.get_group_ids(img_ids=[img_id])
            sent_ids = self.d3.get_sent_ids(group_ids=group_ids)
            sent_id_to_group_id = {}
            for group_id in group_ids:
                group_sent_ids = self.d3.get_sent_ids(group_ids=[group_id])
                for sent_id in group_sent_ids:
                    sent_id_to_group_id[int(sent_id)] = int(group_id)
            sent_list = self.d3.load_sents(sent_ids=sent_ids)
            sent_ids_array = np.asarray([int(sent_id) for sent_id in sent_ids])
            sent_group_ids_array = np.asarray(
                [sent_id_to_group_id.get(int(sent_id), -1) for sent_id in sent_ids]
            )
            sent_id_to_local = {int(sent_id): idx for idx, sent_id in enumerate(sent_ids_array.tolist())}

            ann_ids = coco.get_ann_ids(img_ids=[img_id])
            anns = coco.load_anns(ann_ids)

            instances = []
            for ann in anns:
                category_id = int(ann["category_id"])
                if category_id in sent_id_to_local:
                    local_label = sent_id_to_local[category_id]
                elif category_id - 1 in sent_id_to_local:
                    local_label = sent_id_to_local[category_id - 1]
                else:
                    continue

                x1, y1, w, h = ann["bbox"]
                instances.append(
                    {
                        "ignore_flag": 0,
                        "bbox": [x1, y1, x1 + w, y1 + h],
                        "bbox_label": local_label,
                    }
                )

            data_list.append(
                {
                    "img_path": osp.join(self.img_root, img_info["file_name"]),
                    "img_id": int(img_id),
                    "height": img_info["height"],
                    "width": img_info["width"],
                    "text": [sent["raw_sent"] for sent in sent_list],
                    "sent_ids": sent_ids_array,
                    "sent_group_ids": sent_group_ids_array,
                    "custom_entities": True,
                    "instances": instances,
                }
            )

        return data_list
