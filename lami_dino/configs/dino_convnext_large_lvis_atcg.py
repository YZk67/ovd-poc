"""
LVIS config with ATCG (Analogical Textual Concept Generator).

Based on dino_convnext_large_4scale_12ep_lvis.py, adds analogical
cross-attention distillation for rare (novel) class improvement.

Key addition: during training, for each GT box, ATCG extracts CLIP ROI
features, computes cross-attention over base-class visual prototypes,
and generates pseudo text embeddings as soft classification targets.
"""
from detrex.config import get_config
from .models.dino_convnextl import model
from datetime import datetime

# Remove 'language' key from model config
if "language" in model:
    del model["language"]

# === Score ensemble (required for ATCG — provides CLIP ROI features) ===
model.vlm_query_path = "dataset/metadata/vodcoco_tpa_prompts_convnextl.npy"
model.score_ensemble = True
model.backbone.score_ensemble = model.score_ensemble
model.seen_classes = 'dataset/metadata/ovcoco_seen_classes.json'
model.all_classes = 'dataset/metadata/ovcoco_all_classes.json'
model.vlm_temperature = 100.0
model.alpha = 0.0
model.beta = 0.4
model.novel_scale = 5.0

# === ATCG configuration ===
model.atcg_enabled = True
# Text embeddings for all classes — used to build the knowledge base.
# Uses the same TPA prompts (averaged to single embedding per class inside ATCG).
model.atcg_text_embeddings_path = "dataset/metadata/vodcoco_tpa_prompts_convnextl.npy"
model.atcg_temperature = 0.05       # Cross-attention temperature (lower = sharper)
model.atcg_momentum = 0.999         # EMA momentum for visual prototype updates
model.atcg_loss_temperature = 0.1   # Temperature for KL divergence soft targets
model.atcg_novel_only = True        # Only supervise novel-class dimensions
model.atcg_num_stacked_layers = 0   # Start with initial layer only (no extra params)

# === Standard configs ===
dataloader = get_config("common/data/coco_detr.py").dataloader
optimizer = get_config("common/optim.py").AdamW
lr_multiplier = get_config("common/coco_schedule.py").lr_multiplier_12ep_warmup
train = get_config("common/train.py").train

train.seed = 42
train.init_checkpoint = "./pretrained_models/clip_convnext_large_trans.pth"
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
train.output_dir = f"/root/dino_convnext_large_lvis_atcg_{timestamp}"

# LVIS: 100,170 images, batch_size=16 → 6,260 iter/epoch
train.max_iter = 85200       # 12 epochs (could adjust)
train.eval_period = 99999999 # Evaluate manually or at end
train.log_period = 50
train.checkpointer.period = 7100

train.clip_grad.enabled = True
train.clip_grad.params.max_norm = 0.1
train.clip_grad.params.norm_type = 2
train.sync_batchnorm = True
train.device = "cuda"
model.device = train.device

# === Classes ===
model.num_classes = 65

# Text embedding paths for TPA
model.query_path = "dataset/metadata/vodcoco_tpa_prompts_convnextl.npy"
model.eval_query_path = "dataset/metadata/vodcoco_tpa_prompts_convnextl.npy"

# Fed Loss — disabled for now (same as baseline)
model.use_fed_loss = False
model.cluster_fed_loss = False
model.cluster_label_path = None
model.cat_freq_path = None
model.fed_loss_num_cat = 0
model.select_box_nums_for_evaluation = 300

# === Optimizer ===
base_lr = 1e-4
world_size = 1.5
optimizer.lr = base_lr * world_size
optimizer.betas = (0.9, 0.999)
optimizer.weight_decay = 1e-4
optimizer.params.lr_factor_func = lambda module_name: 0.1 if "backbone" in module_name else 1

# === Dataloader ===
dataloader.train.num_workers = 4
dataloader.train.total_batch_size = 16
dataloader.evaluator.output_dir = train.output_dir
dataloader.test.dataset.names = "lvis_v1_val"

# === TPA (same as baseline) ===
model.classifier.use_tpa = True
model.classifier.text_embed_path = "dataset/metadata/vodcoco_tpa_prompts_convnextl.npy"
model.classifier.eval_text_embed_path = "dataset/metadata/vodcoco_tpa_prompts_convnextl.npy"
model.classifier.tpa_num_prototypes = 5
model.classifier.tpa_hidden_dim = 256
model.classifier.tpa_dropout = 0.05
model.classifier.tpa_tau = 0.07
model.classifier.tpa_log_interval = 200

# Soft-attention aggregation
model.use_soft_attention = True
model.soft_attention_tau = 0.08

# AMP
train.amp.enabled = True

# Multiprocessing stability
import torch.multiprocessing as mp
import os

try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    pass

os.environ['PYTHONUNBUFFERED'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'OFF'
