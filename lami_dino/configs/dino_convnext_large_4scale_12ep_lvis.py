from detrex.config import get_config
from .models.dino_convnextl import model
from datetime import datetime

# Remove 'language' key from model config as it's not a parameter for DINO.__init__()
# The language config is only used for TextClassifier via ${..language.xxx} references
if "language" in model:
    del model["language"]

model.vlm_query_path = "dataset/metadata/lvis_visual_desc_confuse_lvis_convnextl.npy"
model.score_ensemble = True
model.backbone.score_ensemble = model.score_ensemble
model.seen_classes = 'dataset/lvis/lvis_v1_seen_classes.json'
model.all_classes = 'dataset/lvis/lvis_v1_all_classes.json'
model.vlm_temperature = 100.0 # keep same with f-vlm
model.alpha = 0.0
model.beta = 0.4
model.novel_scale = 5.0

# get default config
dataloader = get_config("common/data/lvis_detr.py").dataloader
optimizer = get_config("common/optim.py").AdamW
lr_multiplier = get_config("common/lvis_schedule.py").lr_multiplier_12ep_warmup
# lr_multiplier = get_config("common/lvis_schedule.py").lr_multiplier_12ep_64bs  # Use 64bs scheduler for batch size 64
train = get_config("common/train.py").train

# Set random seed for reproducibility
train.seed = 42  # Fixed seed for fair comparison

# modify training config
# train.init_checkpoint = "clip_convnext_large_trans.pth"
train.init_checkpoint = "./pretrained_models/lami_convnext_large_12ep_lvis/model_final.pth"

# Add timestamp to output directory
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
# resume training from the last checkpoint
train.output_dir = f"/root/lami_convnext_large_12ep_lvis_{timestamp}"

# max training iterations
# - Original COCO dataset: 113600 images
# Calculation logic:
# - LVIS dataset: 100,170 images
# - Formula: total_iterations = (dataset_size / total_batch_size) * num_epochs
# - Batch size 4: 100,170 ÷ 4 = 25,042.5 iter/epoch → 300,510 total (12 epochs) Single GPU A100
# - Batch size 16: 100,170 ÷ 16 = 6,260 iter/epoch → 75,120 total (12 epochs)
# - Batch size 32: 100,170 ÷ 32 = 3,130 iter/epoch → 37,560 total (12 epochs)
# - Batch size 64: 100,170 ÷ 64 = 1,565 iter/epoch → 18,780 total (12 epochs)
# - Standardized values: 7,100 (bs16), 3,550 (bs32), 1,775 (bs64)
# - LR scheduler: use lr_multiplier_12ep_warmup for batch size 32
train.max_iter = 92300 #85200 #85200  # Single GPU A100 4 epochs 12 epochs with batch size 32: 100170/32*12 -- 85200

# run evaluation every ~4 epochs (28400 ≈ 4 × 7100)
# was 99999999 (never) — without intermediate eval there is no signal that the
# auxiliary heads are actually helping until training finishes.
train.eval_period = 28400

# log training infomation every 20 iters
train.log_period = 50

# save checkpoint every 3130 iters
train.checkpointer.period = 7100  # 1 epoch worth of iterations

# gradient clipping for training
# was 0.1 — too tight once APR + RPSA are added on top of the DETR losses.
# 0.5 keeps DETR-style stability while letting the auxiliary heads actually learn.
train.clip_grad.enabled = True
train.clip_grad.params.max_norm = 0.5
train.clip_grad.params.norm_type = 2

train.sync_batchnorm = True

# set training devices
train.device = "cuda"
model.device = train.device

model.num_classes = 1203
# Set the text embedding paths for TPA (using Claude-generated 8 prompts per class)
model.query_path = "dataset/metadata/lvis_claude_prompts_convnextl.npy"
model.eval_query_path = "dataset/metadata/lvis_claude_prompts_convnextl.npy"

# modify optimizer config
# was: base_lr * world_size with world_size=1.5 (= 1.5e-4) — the 1.5 was a hand-tuned
# multiplier with no clear linear-scaling justification and made auxiliary losses
# more likely to blow up early. Use the LaMI-DETR default 1e-4 for total_batch_size=16.
base_lr = 1e-4
optimizer.lr = base_lr
optimizer.betas = (0.9, 0.999)
optimizer.weight_decay = 1e-4
# TPA is ~0.8M parameters sitting behind the whole detector, and at the shared
# 1e-4 it barely moves: in a controlled run on the real LVIS prompt bank, the
# prototypes' effective rank reached 2.46 at lr 1e-3 but only 1.50 at 1e-4 --
# after task structure, this is the single largest lever on whether the K
# prototypes separate at all. The backbone and detector keep their own rates, so
# this only speeds up the text-side module.
# lr_factor_func receives the full parameter path (detectron2/solver/build.py:231),
# e.g. "transformer.decoder.class_embed.6.tpa.prototype_queries".
# Not CLI-overridable (it is a callable, not a config scalar) -- edit this to sweep.
tpa_lr_multiplier = 10.0


def _lr_factor(param_name: str) -> float:
    if "backbone" in param_name:
        return 0.1
    if ".tpa." in param_name:
        return tpa_lr_multiplier
    return 1.0


optimizer.params.lr_factor_func = _lr_factor

# modify dataloader config
# Start with conservative setting, can be increased if stable
dataloader.train.num_workers = 4  # 1 worker per GPU for 4GPU training

# please notice that this is total batch size.
# surpose you're using 4 gpus for training and the batch size for
# each gpu is 16/4 = 4
# Note: Using Option 3 (averaged embeddings), batch_size can remain at 4
# If using Option 2 (6015 queries), reduce to batch_size=1
dataloader.train.total_batch_size = 16  # Can use 4 with Option 3

# dump the testing results into output_dir for visualization
dataloader.evaluator.output_dir = train.output_dir
dataloader.test.dataset.names = "lvis_v1_val"

# ====== 测试 Claude Prompts ======
# Fed Loss是LVIS必须的基础设施，所有实验都需要使用
model.use_fed_loss = True  # ✅ 必须开启，处理长尾分布
model.cluster_fed_loss = False
# model.cluster_label_path = 'dataset/cluster/lvis_cluster_128.npy'
model.cat_freq_path = "dataset/lvis/lvis_v1_train_norare_cat_info.json"
model.fed_loss_num_cat = 100  # 每次采样100个类别计算loss
model.select_box_nums_for_evaluation = 300

# Enable TPA (Text Prototype Aggregator) by modifying the classifier
model.classifier.use_tpa = True
model.classifier.text_embed_path = "dataset/metadata/lvis_claude_prompts_convnextl.npy"
model.classifier.eval_text_embed_path = "dataset/metadata/lvis_claude_prompts_convnextl.npy"
model.classifier.tpa_num_prototypes = 5  # 8 prompts -> 5 prototypes (better utilization)
model.classifier.tpa_hidden_dim = 256
# was 0.05 — bumped to 0.1 (the value PRIORITY_DECISION.md recommends) for stronger
# prototype regularization and to discourage attention collapse onto a single prompt.
model.classifier.tpa_dropout = 0.1
model.classifier.tpa_tau = 0.07  # Optimal temperature for attention aggregation (τ ≈ 0.05–0.1 recommended) 0.1 is original
model.classifier.tpa_log_interval = 200
# Derive the APR warmup from this run's schedule rather than the aggregator's
# built-in default, which is sized for max_iter=85200 and would be 4.6% here.
model.classifier.tpa_warmup_steps = int(train.max_iter * 0.05)

# Query initialization: Soft-attention aggregation parameters for multi-prototype query initialization
model.use_soft_attention = True  # Enable soft-attention aggregation in query initialization
# was 0.08 — too sharp, caused soft-attention to behave like top-1 and waste the
# extra prototypes. 0.15 keeps attention soft enough to actually mix prototypes.
model.soft_attention_tau = 0.15


# ========================= RPSA V1 =========================
# RPSA parameters V1
# model.transformer.use_rpsa = True
# model.criterion.weight_dict["loss_rpsa"] = 0.08
# model.transformer.rpsa_module.K = 6
# model.transformer.rpsa_module.tau_align = 0.07
# model.transformer.rpsa_module.sigma = 1.0
# model.transformer.rpsa_module.bg_thresh = 0.0
# model.transformer.rpsa_module.subsample_tokens = 2048  # random subsample tokens to stabilize cost
# model.transformer.rpsa_module.subsample_method = "confidence"
# model.transformer.rpsa_module.detach_pi = True
# model.transformer.rpsa_module.stop_grad_vision = False
# model.transformer.rpsa_module.stop_grad_text = False     # warm-up: also freeze prototype branch initially
# model.transformer.rpsa_module.bg_percentile = 0.65

# model.transformer.rpsa_token_topk = 768
# model.transformer.rpsa_confidence_threshold = 0.3
# model.transformer.rpsa_warmup_start = 0
# model.transformer.rpsa_warmup_iters = 3000
# model.transformer.rpsa_warmup_init_scale = 0.0
# model.transformer.rpsa_warmup_power = 1.0

# ========================= RPSA V2 =========================
# # --- 基本启用 ---
# model.transformer.use_rpsa = True
# # --- 损失权重（降低约 40%）---
# model.criterion.weight_dict["loss_rpsa"] = 0.03
# # --- 聚类设置 ---
# model.transformer.rpsa_module.K = 6
# # --- 对齐温度：soft 化 InfoNCE ---
# model.transformer.rpsa_module.tau_align = 0.12  # 由 0.07 ↑
# model.transformer.rpsa_module.sigma = 1.0
# # --- 背景筛选策略：只保留分位数控制 ---
# model.transformer.rpsa_module.bg_thresh = None        # 禁用固定阈
# model.transformer.rpsa_module.bg_percentile = 0.55    # 从 0.65 ↓，保留更多前景
# # --- 采样策略 ---
# model.transformer.rpsa_module.subsample_tokens = 2048
# model.transformer.rpsa_module.subsample_method = "confidence"
# # --- 聚类权重锐化（路由更聚焦）---
# model.transformer.rpsa_module.alpha_pi = 1.5          # 新增，控制 soft cluster 纯度
# # --- 梯度流向 ---
# model.transformer.rpsa_module.detach_pi = False       # 允许 π 反传，修正错误聚类
# model.transformer.rpsa_module.stop_grad_vision = False
# model.transformer.rpsa_module.stop_grad_text = False  # 后续再考虑 warm-up text
# # --- token gating ---
# model.transformer.rpsa_token_topk = 768
# model.transformer.rpsa_confidence_threshold = 0.3
# # --- warm-up 策略（关键）---
# model.transformer.rpsa_warmup_start = 0
# model.transformer.rpsa_warmup_iters = 1500             # 由 3000 ↓，加快生效
# model.transformer.rpsa_warmup_init_scale = 0.0
# model.transformer.rpsa_warmup_power = 1.0


# ========================= RPSA V3 (stable preset) =========================
model.transformer.use_rpsa = True
model.criterion.weight_dict["loss_rpsa"] = 0.05
model.transformer.rpsa_module.K = 8
model.transformer.rpsa_module.tau_align = 0.06
model.transformer.rpsa_module.sigma = 1.0
model.transformer.rpsa_module.bg_thresh = 0.05
model.transformer.rpsa_module.bg_percentile = 0.60
model.transformer.rpsa_module.subsample_tokens = 2048
model.transformer.rpsa_module.subsample_method = "confidence"
# was False — letting π gradients flow back through the encoder pseudo-mask is
# unstable when enc_outputs_class is still noisy. Detach for the first stable run;
# flip to False later only if it clearly helps.
model.transformer.rpsa_module.detach_pi = True
model.transformer.rpsa_module.stop_grad_vision = False
model.transformer.rpsa_module.stop_grad_text = False
model.transformer.rpsa_token_topk = 1024
model.transformer.rpsa_confidence_threshold = 0.3
# --- warm-up by iteration ---
model.transformer.rpsa_warmup_start = 20000
model.transformer.rpsa_warmup_iters = 8000
model.transformer.rpsa_warmup_init_scale = 0.0
model.transformer.rpsa_warmup_power = 1.0

# ========================= APR weight tune =========================
# was 1.0 (inherited from dino_convnextl.py). The TPA already applies internal
# λ_orth=0.10 and λ_div=0.03 with cosine warmup; multiplying by 1.0 again pushes
# the regularizer to the same magnitude as loss_class and crowds out the main
# task. 0.1 keeps the regularizer present without dominating.
model.criterion.weight_dict["loss_apr"] = 0.1


# Enable Automatic Mixed Precision (AMP) for faster training
train.amp.enabled = True

# Add multiprocessing stability settings to prevent BrokenPipeError
import torch.multiprocessing as mp
import os

# Set multiprocessing start method to spawn for better stability with multiple GPUs
try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    # If already set, continue
    pass

# Additional environment variables for stability
os.environ['PYTHONUNBUFFERED'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'OFF'
