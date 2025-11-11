"""
Fixed TPA configuration with improved diversity loss and hyperparameters
This config addresses the diversity issues identified in the visualization.
"""

from detrex.config import get_config
from .models.dino_convnextl import model
from datetime import datetime

# Remove 'language' key from model config as it's not a parameter for DINO.__init__()
if "language" in model:
    del model["language"]

model.vlm_query_path = "dataset/metadata/lvis_visual_desc_confuse_lvis_convnextl.npy"
model.score_ensemble = True
model.backbone.score_ensemble = model.score_ensemble
model.seen_classes = 'dataset/lvis/lvis_v1_seen_classes.json'
model.all_classes = 'dataset/lvis/lvis_v1_all_classes.json'
model.vlm_temperature = 100.0
model.alpha = 0.0
model.beta = 0.4
model.novel_scale = 5.0

# get default config
dataloader = get_config("common/data/lvis_detr.py").dataloader
optimizer = get_config("common/optim.py").AdamW
lr_multiplier = get_config("common/lvis_schedule.py").lr_multiplier_12ep_warmup
train = get_config("common/train.py").train

# Set random seed for reproducibility
train.seed = 42

# modify training config
train.init_checkpoint = "./pretrained_models/lami_convnext_large_12ep_lvis/model_final.pth"

# Add timestamp to output directory
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
train.output_dir = f"/root/lami_convnext_large_12ep_lvis_fixed_{timestamp}"

train.max_iter = 92300
train.eval_period = 99999999
train.log_period = 50
train.checkpointer.period = 7100

# gradient clipping
train.clip_grad.enabled = True
train.clip_grad.params.max_norm = 0.1
train.clip_grad.params.norm_type = 2

train.sync_batchnorm = True
train.device = "cuda"
model.device = train.device

model.num_classes = 1203
model.query_path = "dataset/metadata/lvis_claude_prompts_convnextl.npy"
model.eval_query_path = "dataset/metadata/lvis_claude_prompts_convnextl.npy"

# modify optimizer config
base_lr = 1e-4
world_size = 1.5
optimizer.lr = base_lr * world_size
optimizer.betas = (0.9, 0.999)
optimizer.weight_decay = 1e-4
optimizer.params.lr_factor_func = lambda module_name: 0.1 if "backbone" in module_name else 1

# modify dataloader config
dataloader.train.num_workers = 4
dataloader.train.total_batch_size = 16
dataloader.evaluator.output_dir = train.output_dir
dataloader.test.dataset.names = "lvis_v1_val"

# Fed Loss
model.use_fed_loss = True
model.cluster_fed_loss = False
model.cat_freq_path = "dataset/lvis/lvis_v1_train_norare_cat_info.json"
model.fed_loss_num_cat = 100
model.select_box_nums_for_evaluation = 300

# ====== FIXED TPA Configuration ======
# Enable TPA with improved hyperparameters
model.classifier.use_tpa = True
model.classifier.text_embed_path = "dataset/metadata/lvis_claude_prompts_convnextl.npy"
model.classifier.eval_text_embed_path = "dataset/metadata/lvis_claude_prompts_convnextl.npy"
model.classifier.tpa_num_prototypes = 5

# Improved hyperparameters based on analysis:
# 1. Increased diversity loss weight to encourage different prototypes
model.classifier.tpa_lambda_div = 0.12  # ↑ from 0.03 (4x increase)
# 2. Increased orthogonality loss weight for better separation
model.classifier.tpa_lambda_orth = 0.20  # ↑ from 0.10 (2x increase)
# 3. Increased temperature for softer attention (allows more exploration)
model.classifier.tpa_tau = 0.10  # ↑ from 0.07 (allows more exploration)

model.classifier.tpa_hidden_dim = 256
model.classifier.tpa_dropout = 0.05
model.classifier.tpa_log_interval = 200

# Query initialization
model.use_soft_attention = True
model.soft_attention_tau = 0.10  # Match TPA tau

# ========================= RPSA V3 =========================
model.transformer.use_rpsa = True
model.criterion.weight_dict["loss_rpsa"] = 0.05
model.transformer.rpsa_module.K = 8
model.transformer.rpsa_module.tau_align = 0.06
model.transformer.rpsa_module.sigma = 1.0
model.transformer.rpsa_module.bg_thresh = 0.05
model.transformer.rpsa_module.bg_percentile = 0.60
model.transformer.rpsa_module.subsample_tokens = 2048
model.transformer.rpsa_module.subsample_method = "confidence"
model.transformer.rpsa_module.detach_pi = False
model.transformer.rpsa_module.stop_grad_vision = False
model.transformer.rpsa_module.stop_grad_text = False
model.transformer.rpsa_token_topk = 1024
model.transformer.rpsa_confidence_threshold = 0.3
model.transformer.rpsa_warmup_start = 20000
model.transformer.rpsa_warmup_iters = 8000
model.transformer.rpsa_warmup_init_scale = 0.0
model.transformer.rpsa_warmup_power = 1.0

# Enable AMP
train.amp.enabled = True

# Multiprocessing settings
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

