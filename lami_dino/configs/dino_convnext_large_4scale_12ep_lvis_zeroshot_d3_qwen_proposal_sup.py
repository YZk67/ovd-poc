import detectron2.data.transforms as T
from detectron2.config import LazyCall as L

from .dino_convnext_large_4scale_12ep_lvis_zeroshot_d3 import (
    dataloader,
    lr_multiplier,
    model,
    optimizer,
    train,
)


# Strict split datasets are registered from D3_STRICT_TRAIN_JSON and
# D3_STRICT_VAL_JSON. Generate those COCO files with
# tools/filter_coco_annotations_by_image_ids.py before launching training.
dataloader.train.dataset.names = "d3_strict_train"
dataloader.test.dataset.names = "d3_strict_val"
dataloader.evaluator.dataset_name = "${..test.dataset.names}"

# First Qwen-supervision smoke keeps geometry simple: no crop and no random flip.
# Resize-only preserves the normalized coordinates of offline Qwen proposal boxes.
dataloader.train.mapper.augmentation = [
    L(T.ResizeShortestEdge)(
        short_edge_length=(480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800),
        max_size=1333,
        sample_style="choice",
    ),
]
dataloader.train.mapper.augmentation_with_crop = None

model.qwen_proposal_supervision_enabled = True
model.qwen_proposal_jsonl = "dataset/d3/qwen_proposal_train.jsonl"
model.qwen_proposal_topk_per_image = 100
model.qwen_proposal_match_iou = 0.5
model.qwen_proposal_positive_threshold = 0.6
model.qwen_proposal_negative_threshold = 0.3
model.qwen_proposal_rank_margin = 0.2
model.qwen_proposal_max_rank_pairs = 1024
model.qwen_proposal_use_sample_weight = True

model.criterion.weight_dict["loss_qwen_soft"] = 0.1
model.criterion.weight_dict["loss_qwen_rank"] = 0.2

dataloader.train.total_batch_size = 1
dataloader.train.num_workers = 4
train.max_iter = 2000
train.eval_period = 1000
train.log_period = 20
train.checkpointer.period = 1000
train.output_dir = "./output/lami_convnext_large_12ep_lvis_zeroshot_d3_qwen_proposal_sup"
dataloader.evaluator.output_dir = train.output_dir
