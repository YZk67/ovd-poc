"""OW-OVD on LaMI-DETR: Open World + Open Vocabulary Detection.

Based on LaMI-DETR with HAUF unknown detection at inference.
Trains on M-OWODB task splits with incremental class introduction.
"""

from detrex.config import get_config
from .models.dino_convnextl import model

# === OVD settings ===
# Using ovdcoco VLM embeddings for score_ensemble
model.vlm_query_path = "dataset/metadata/ovdcoco_confuse_convnextl.npy"
model.score_ensemble = True
model.backbone.score_ensemble = model.score_ensemble
model.seen_classes = 'dataset/metadata/ovcoco_seen_classes.json'
model.all_classes = 'dataset/metadata/ovcoco_all_classes.json'
model.vlm_temperature = 100.0
model.alpha = 0.0
model.beta = 0.4
model.novel_scale = 5.0

# === OW-OVD: HAUF settings (enable after VSAS selection) ===
model.hauf_enabled = False  # set True after VSAS attribute selection
model.hauf_att_path = "dataset/metadata/coco_vsas_selected.pth"
model.hauf_top_k = 10
model.hauf_threshold = 0.5
model.unknown_class_id = 65  # class ID for "unknown"

# === OW-OVD: VSAS distribution logging (Phase 1: collect distributions) ===
model.vsas_log_distributions = True
model.vsas_all_att_path = "dataset/metadata/coco_att_embeddings.pth"
model.vsas_dist_save_path = "dataset/metadata/att_distributions.pth"

# === Common configs ===
dataloader = get_config("common/data/coco_detr.py").dataloader
optimizer = get_config("common/optim.py").AdamW
lr_multiplier = get_config("common/coco_schedule.py").lr_multiplier_12ep_warmup
train = get_config("common/train.py").train
train.ddp.find_unused_parameters = True

# === Training config ===
train.init_checkpoint = "./pretrained_models/clip_convnext_large_trans.pth"
train.output_dir = "./output/dino_convnext_large_owovd"
train.max_iter = 85200
train.eval_period = 7100
train.log_period = 200
train.checkpointer.period = 7100

train.clip_grad.enabled = True
train.clip_grad.params.max_norm = 0.1
train.clip_grad.params.norm_type = 2

train.device = "cuda"
model.device = train.device

# === Model settings ===
# NOTE: using 65-class OV-COCO embeddings for now to validate pipeline.
# TODO: generate 80-class COCO embeddings for full OWODB training.
model.num_classes = 65
model.query_path = "dataset/metadata/ovdcoco_visual_desc_convnextl.npy"
model.eval_query_path = "dataset/metadata/ovdcoco_visual_desc_convnextl.npy"

model.use_fed_loss = True
model.cluster_fed_loss = True
model.cluster_label_path = 'dataset/cluster/ovdcoco_cluster4_16.npy'
model.cat_freq_path = "dataset/coco/ovd_ins_train2017_all_cat_info.json"
model.fed_loss_num_cat = 32
model.select_box_nums_for_evaluation = 2000

# === Optimizer ===
optimizer.lr = 1e-4
optimizer.betas = (0.9, 0.999)
optimizer.weight_decay = 1e-4
optimizer.params.lr_factor_func = lambda module_name: 0.1 if "backbone" in module_name else 1

# === Dataloader ===
dataloader.train.num_workers = 4
dataloader.train.total_batch_size = 16
dataloader.evaluator.output_dir = train.output_dir
# Use standard OV-COCO to validate VSAS pipeline first
# Switch to owodb_m_t1_train after generating 80-class embeddings
dataloader.test.dataset.names = "ovcoco_2017_val_all"
