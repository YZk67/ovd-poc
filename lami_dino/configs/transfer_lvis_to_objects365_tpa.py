"""
Zero-shot transfer: LVIS-trained model -> Objects365 v2 val (all classes novel).

Run:
  CUDA_VISIBLE_DEVICES=0 python tools/train_net.py \
    --config-file lami_dino/configs/transfer_lvis_to_objects365_tpa.py \
    --num-gpus 1 --eval-only \
    train.init_checkpoint=/root/autodl-tmp/model_final.pth

Prerequisites (run on the training server before launch):
  1. Download Objects365 val:
       dataset/object365/val/*.jpg
       dataset/object365/annotations/zhiyuan_objv2_val.json
  2. Generate obj_cats.json + class names:
       python tools/extract_obj365_cats.py
  3. scp embeddings from local:
       dataset/metadata/obj_365_openclip_convnextl_prompts8.npy   (365, 8, 768)
"""

from detrex.config import get_config
from detectron2.config import LazyCall as L
from detectron2.evaluation import COCOEvaluator

from .dino_convnext_large_4scale_12ep_lvis import (
    model,
    optimizer,
    lr_multiplier,
    train,
)

# ── dataloader: Objects365 v2 val ────────────────────────────────────────────
dataloader = get_config("common/data/obj365_detr.py").dataloader
dataloader.evaluator = L(COCOEvaluator)(
    dataset_name="obj365v2_val",
    output_dir=train.output_dir,
)

# ── model: 365 classes, all novel ────────────────────────────────────────────
model.num_classes = 365

_OBJ365_EMBED = "dataset/metadata/obj_365_openclip_convnextl_prompts8.npy"  # (365,8,768)
model.vlm_query_path        = _OBJ365_EMBED   # 3-D auto mean-pooled inside DINO
model.query_path            = _OBJ365_EMBED
model.eval_query_path       = _OBJ365_EMBED
model.classifier.text_embed_path      = _OBJ365_EMBED
model.classifier.eval_text_embed_path = _OBJ365_EMBED

model.seen_classes = "dataset/metadata/empty_seen_classes.json"
model.all_classes  = "dataset/metadata/obj_365_classes.json"

# zero-shot score-ensemble: all classes novel
model.score_ensemble = True
model.backbone.score_ensemble = True
model.alpha       = 0.0
model.beta        = 0.10
model.novel_scale = 1.0

# disable LVIS-specific fed-loss
model.use_fed_loss     = False
model.cluster_fed_loss = False

# ── train: eval-only ─────────────────────────────────────────────────────────
train.init_checkpoint          = "/root/autodl-tmp/model_final.pth"
train.init_checkpoint_scope    = "full"
train.eval_period              = 999999999
train.checkpointer.period      = 999999999
train.output_dir               = "/root/autodl-tmp/transfer_lvis_to_obj365"
dataloader.evaluator.output_dir = train.output_dir
