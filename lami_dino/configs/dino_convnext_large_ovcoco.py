"""OV-COCO on LaMI-DETR: Open Vocabulary Detection (48 base + 17 novel).

Standard OV-COCO benchmark: train on 48 base classes, evaluate on all 65.
Uses CLIP ConvNeXt-L backbone with frozen features.
"""

from detrex.config import get_config
from .models.dino_convnextl import model

# === OVD settings ===
model.score_ensemble = True
model.backbone.score_ensemble = True
model.vlm_temperature = 100.0
model.alpha = 0.0
model.beta = 0.4
model.novel_scale = 5.0
model.seen_classes = "dataset/metadata/ovcoco_seen_classes.json"
model.all_classes = "dataset/metadata/ovcoco_all_classes.json"

# === Freeze backbone (preserve CLIP features) ===
model.freeze_backbone = True

# === Disable HAUF (not needed for standard OVD) ===
model.hauf_enabled = False

# === Common configs ===
dataloader = get_config("common/data/coco_detr.py").dataloader
optimizer = get_config("common/optim.py").AdamW
lr_multiplier = get_config("common/coco_schedule.py").lr_multiplier_12ep_warmup
train = get_config("common/train.py").train
train.ddp.find_unused_parameters = True

# === Training config ===
train.init_checkpoint = "./pretrained_models/clip_convnext_large_trans.pth"
train.output_dir = "./output/dino_convnext_large_ovcoco"
# OV-COCO base train: ~107k images / batch 16 = ~6735 iter per epoch
# Train 12 epochs (matching newOvd/ovdcocoeasy branches)
train.max_iter = 85200
train.eval_period = 7100
train.log_period = 200
train.checkpointer.period = 7100

train.clip_grad.enabled = True
train.clip_grad.params.max_norm = 0.1
train.clip_grad.params.norm_type = 2

train.device = "cuda"
model.device = train.device

# === Model settings (65 classes for OV-COCO) ===
model.num_classes = 65
model.query_path = "dataset/metadata/ovdcoco_visual_desc_convnextl.npy"
model.eval_query_path = "dataset/metadata/ovdcoco_visual_desc_convnextl.npy"
model.vlm_query_path = "dataset/metadata/ovdcoco_confuse_convnextl.npy"

model.use_fed_loss = True
model.cluster_fed_loss = True
model.cluster_label_path = "dataset/cluster/ovdcoco_cluster4_16.npy"
model.cat_freq_path = "dataset/coco/ovd_ins_train2017_all_cat_info.json"
model.fed_loss_num_cat = 32
model.select_box_nums_for_evaluation = 2000

# === Optimizer ===
optimizer.lr = 1e-4
optimizer.betas = (0.9, 0.999)
optimizer.weight_decay = 1e-4
optimizer.params.lr_factor_func = lambda module_name: 0.1 if "backbone" in module_name else 1

# === Dataloader (OV-COCO: train on base 48, eval on all 65) ===
dataloader.train.num_workers = 4
dataloader.train.total_batch_size = 16
dataloader.evaluator.output_dir = train.output_dir
dataloader.train.dataset.names = "ovdcoco65_2017_train_b"
dataloader.test.dataset.names = "ovdcoco65_2017_val_all"
