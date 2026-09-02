from detrex.config import get_config
from .models.dino_convnextl import model

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

# Paper-aligned initialization: restore only the CLIP ConvNeXt-L visual trunk.
# The DINO detector, transformer, classification head, TPA, APR, and RPSA all
# start fresh; tools/train_net.py enforces this scope even if a checkpoint file
# unexpectedly contains additional detector tensors.
train.init_checkpoint = "./pretrained_models/clip_convnext_large_trans.pth"
train.init_checkpoint_scope = "backbone_only"
# Match the LaMI ConvNeXt protocol precisely: the visual trunk stages are
# frozen, while p1/p2/p3 output normalization layers remain trainable at the
# declared 0.1x backbone learning rate. train_net.py validates this at startup.
train.backbone_trainable_scope = "output_norm_only"

# Keep the directory stable so ``--resume`` finds its ``last_checkpoint`` file.
# Use a CLI override for each ablation, e.g. train.output_dir=output/..._k1.
train.output_dir = "./output/instructdet_clip_convnext_large_12ep_lvis"

# The LVIS RepeatFactorTrainingSampler yields about 7,100 iterations per epoch
# at total batch size 16. Keep max_iter identical to the 12-epoch scheduler's
# endpoint; the previous 92,300 value silently added roughly one extra epoch.
iterations_per_epoch = 7100
train.max_iter = 12 * iterations_per_epoch

# run evaluation every ~4 epochs (28400 ≈ 4 × 7100)
# was 99999999 (never) — without intermediate eval there is no signal that the
# auxiliary heads are actually helping until training finishes.
train.eval_period = 4 * iterations_per_epoch

# log training infomation every 20 iters
train.log_period = 50

# save checkpoint every 3130 iters
train.checkpointer.period = iterations_per_epoch

# gradient clipping for training
# was 0.1 — too tight once APR + RPSA are added on top of the DETR losses.
# 0.5 keeps DETR-style stability while letting the auxiliary heads actually learn.
train.clip_grad.enabled = True
train.clip_grad.params.max_norm = 0.5
train.clip_grad.params.norm_type = 2
# The revised APR barrier can have a deliberately large gradient near collapse.
# Give TPA and the remaining detector separate 0.5-norm clipping budgets so
# this corrective signal cannot suppress every detector parameter update.
train.separate_tpa_grad_clip = True

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
train.tpa_lr_multiplier = tpa_lr_multiplier  # persist in the resolved config


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
# Paper Eq. (1): sqrt(256) * tau_p = 16 * 0.004375 = 0.07. This is
# numerically identical to the validated no-sqrt/0.07 setup while restoring the
# equation's explicit scaled-dot-product form.
model.classifier.tpa_tau = 0.004375
model.classifier.tpa_cls_tau = 0.07  # Eq. (2) temperature-controlled log-mean-exp
model.classifier.tpa_diversity_barrier_eps = 1e-4
model.classifier.tpa_log_interval = 200
# Let the barrier reach full strength while task gradients are still detached.
model.classifier.tpa_warmup_steps = 100
model.tpa_stabilization_steps = 500
# Keep forward prototypes unchanged but attenuate the collapse-inducing detector
# gradient into TPA after stabilization. APR retains its full gradient.
model.tpa_task_gradient_scale = 0.1

# Query initialization: Soft-attention aggregation parameters for multi-prototype query initialization
model.use_soft_attention = True  # Enable soft-attention aggregation in query initialization
# was 0.08 — too sharp, caused soft-attention to behave like top-1 and waste the
# extra prototypes. 0.15 keeps attention soft enough to actually mix prototypes.
model.soft_attention_tau = 0.15
model.soft_category_topk = 3  # Eq. (3): retain top-R category hypotheses
model.soft_category_tau = 1.0


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
# Transformer-level top-M selection below already removes padding and retains
# the highest-confidence valid tokens; do not subsample a second time.
model.transformer.rpsa_module.subsample_tokens = 0
model.transformer.rpsa_module.subsample_method = "confidence"
# was False — letting π gradients flow back through the encoder pseudo-mask is
# unstable when enc_outputs_class is still noisy. Detach for the first stable run;
# flip to False later only if it clearly helps.
model.transformer.rpsa_module.detach_pi = True
model.transformer.rpsa_module.stop_grad_vision = False
model.transformer.rpsa_module.stop_grad_text = False
model.transformer.rpsa_token_topk = 1024
# Top-M itself provides high-confidence selection. A hard 0.3 softmax threshold
# is too strict early in a 100-way federated classifier; thresholded mode is
# supported but now fails fast instead of duplicating one fallback token.
model.transformer.rpsa_confidence_threshold = 0.0
# --- warm-up by iteration ---
model.transformer.rpsa_warmup_start = 20000
model.transformer.rpsa_warmup_iters = 8000
model.transformer.rpsa_warmup_init_scale = 0.0
model.transformer.rpsa_warmup_power = 1.0

# ========================= APR weight tune =========================
# Keep the base model's outer weight of 1.0. The TPA already applies the Eq. (5)
# coefficients internally (λ_div=0.10 and λ_bal=0.03), so the raw APR term is
# only about 0.1 versus a roughly 35-point total detector loss in the 500-step
# CLIP-only smoke run. An additional 0.1 multiplier reduced its contribution to
# about 0.01 and the measured prototype rank regressed from 1.568 to 1.519.
model.criterion.weight_dict["loss_apr"] = 1.0


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
