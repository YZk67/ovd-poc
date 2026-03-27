"""OW-OVD on LaMI-DETR: Open World + Open Vocabulary Detection.

Based on LaMI-DETR with HAUF unknown detection at inference.
Trains on M-OWODB task splits with incremental class introduction.
"""

from detrex.config import get_config
from .models.dino_convnextl import model

# === OVD settings ===
model.score_ensemble = False
model.backbone.score_ensemble = True  # needed for HAUF ROI feature extraction
model.vlm_temperature = 100.0
model.alpha = 0.0
model.beta = 0.4
model.novel_scale = 5.0

# === Freeze backbone (preserve CLIP features, core of OW-OVD) ===
model.freeze_backbone = True

# === OW-OVD: HAUF settings (enable after VSAS selection) ===
model.hauf_enabled = False
model.hauf_att_path = "dataset/metadata/coco_vsas_selected.pth"
model.hauf_top_k = 10
model.hauf_threshold = 0.55
model.unknown_class_id = 80

# === OW-OVD: VSAS distribution logging ===
model.vsas_log_distributions = False  # collect on single GPU separately
model.vsas_all_att_path = "dataset/metadata/coco_att_embeddings.pth"
model.vsas_dist_save_path = "dataset/metadata/att_distributions.pth"

# === Common configs ===
dataloader = get_config("common/data/coco_detr.py").dataloader
optimizer = get_config("common/optim.py").AdamW
lr_multiplier = get_config("common/coco_schedule.py").lr_multiplier_12ep_warmup
train = get_config("common/train.py").train
train.ddp.find_unused_parameters = True  # needed: frozen backbone params used in forward but no grad

# === Training config ===
train.init_checkpoint = "./pretrained_models/clip_convnext_large_trans.pth"
train.output_dir = "./output/dino_convnext_large_owovd"
# 1 epoch: M-OWODB T1 ~95k images / batch 16 = ~5950 iter
train.max_iter = 5950
train.eval_period = 5950
train.log_period = 200
train.checkpointer.period = 5950

train.clip_grad.enabled = True
train.clip_grad.params.max_norm = 0.1
train.clip_grad.params.norm_type = 2

train.device = "cuda"
model.device = train.device

# === Model settings (COCO 80 classes) ===
model.num_classes = 80
model.query_path = "dataset/metadata/coco_80_text_convnextl.npy"
model.eval_query_path = "dataset/metadata/coco_80_text_convnextl.npy"

model.use_fed_loss = False
model.cluster_fed_loss = False
model.select_box_nums_for_evaluation = 300

# === Optimizer ===
# Backbone frozen -> its lr_factor doesn't matter, but keep for clarity
optimizer.lr = 1e-4
optimizer.betas = (0.9, 0.999)
optimizer.weight_decay = 1e-4
optimizer.params.lr_factor_func = lambda module_name: 0.1 if "backbone" in module_name else 1

# === Dataloader ===
dataloader.train.num_workers = 4
dataloader.train.total_batch_size = 16
dataloader.evaluator.output_dir = train.output_dir
# M-OWODB Task 1: 20 known classes, eval on all 80
dataloader.train.dataset.names = "owodb_m_t1_train"
dataloader.test.dataset.names = "owodb_m_t1_test"
