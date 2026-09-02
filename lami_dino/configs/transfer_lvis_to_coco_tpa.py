"""
Zero-shot transfer: LVIS-trained model → COCO 80 (all classes novel).
Inherits all model/optimizer/schedule from the LVIS training config and
overrides only dataset, class embeddings, and eval-only settings.

Run:
  CUDA_VISIBLE_DEVICES=0,1,2,3 python tools/train_net.py \
    --config-file lami_dino/configs/transfer_lvis_to_coco_tpa.py \
    --num-gpus 4 --eval-only \
    train.init_checkpoint=/root/autodl-tmp/model_final.pth
"""

from detrex.config import get_config
from detectron2.config import LazyCall as L
from detectron2.evaluation import COCOEvaluator

# ── base config (model / optimizer / schedule) ───────────────────────────────
from .dino_convnext_large_4scale_12ep_lvis import (
    model,
    optimizer,
    lr_multiplier,
    train,
)

# ── dataloader: switch to COCO val2017 ───────────────────────────────────────
dataloader = get_config("common/data/coco_detr.py").dataloader
dataloader.evaluator = L(COCOEvaluator)(
    dataset_name="coco_2017_val",
    output_dir=train.output_dir,
)

# ── model: COCO 80 classes, all novel ────────────────────────────────────────
model.num_classes = 80

_COCO_EMBED = "dataset/metadata/coco_80_openclip_convnextl_prompts8.npy"  # (80,8,768)
model.vlm_query_path        = _COCO_EMBED   # model auto-mean-pools 3-D arrays
model.query_path            = _COCO_EMBED
model.eval_query_path       = _COCO_EMBED
model.classifier.text_embed_path      = _COCO_EMBED
model.classifier.eval_text_embed_path = _COCO_EMBED

model.seen_classes = "dataset/metadata/empty_seen_classes.json"   # [] → all novel
model.all_classes  = "dataset/metadata/coco_80_classes.json"

# zero-shot score-ensemble settings: all classes treated as novel
model.score_ensemble = True
model.backbone.score_ensemble = True
model.alpha       = 0.0   # no seen-class branch contribution
model.beta        = 0.10  # novel-class VLM weight
model.novel_scale = 1.0

# disable fed-loss (LVIS-specific, not needed for eval)
model.use_fed_loss     = False
model.cluster_fed_loss = False

# ── train: eval-only mode ────────────────────────────────────────────────────
train.init_checkpoint          = "/root/autodl-tmp/model_final.pth"
train.init_checkpoint_scope    = "full"
train.eval_period              = 999999999   # never mid-run
train.checkpointer.period      = 999999999   # never save
train.output_dir               = "/root/autodl-tmp/transfer_lvis_to_coco"
dataloader.evaluator.output_dir = train.output_dir
