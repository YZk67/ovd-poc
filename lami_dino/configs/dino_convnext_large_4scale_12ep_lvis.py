from detrex.config import get_config
from .models.dino_convnextl import model
from datetime import datetime

# Remove 'language' key from model config as it's not a parameter for DINO.__init__()
# The language config is only used for TextClassifier via ${..language.xxx} references
if "language" in model:
    del model["language"]

model.vlm_query_path = "dataset/metadata/vodcoco_tpa_prompts_convnextl.npy"
model.score_ensemble = True
model.backbone.score_ensemble = model.score_ensemble
model.seen_classes = None
model.all_classes = None
model.vlm_temperature = 100.0 # keep same with f-vlm
model.alpha = 0.0
model.beta = 0.4
model.novel_scale = 5.0

# get default config
dataloader = get_config("common/data/coco_detr.py").dataloader
optimizer = get_config("common/optim.py").AdamW
lr_multiplier = get_config("common/coco_schedule.py").lr_multiplier_12ep_warmup
# lr_multiplier = get_config("common/lvis_schedule.py").lr_multiplier_12ep_64bs  # Use 64bs scheduler for batch size 64
train = get_config("common/train.py").train

# Set random seed for reproducibility
train.seed = 42  # Fixed seed for fair comparison


# modify training config
# train.init_checkpoint = "clip_convnext_large_trans.pth"
train.init_checkpoint = "./pretrained_models/clip_convnext_large_trans.pth"
# Add timestamp to output directory
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
# resume training from the last checkpoint
train.output_dir = f"/root/dino_convnext_large_ovcoco65_{timestamp}"

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
train.max_iter = 88700 #85200  # Single GPU A100 4 epochs 12 epochs with batch size 32: 100170/32*12 -- 85200

# run evaluation every 3130 iters
train.eval_period = 5000  # Evaluate after each epoch 7100//2

# log training infomation every 20 iters
train.log_period = 50

# save checkpoint every 3130 iters
train.checkpointer.period = 7400  # 1 epoch worth of iterations

# gradient clipping for training
train.clip_grad.enabled = True
train.clip_grad.params.max_norm = 0.1
train.clip_grad.params.norm_type = 2

train.sync_batchnorm = True

# set training devices
train.device = "cuda"
model.device = train.device

model.num_classes = 65
# Set the text embedding paths for TPA (using Claude-generated 8 prompts per class)
model.query_path = "dataset/metadata/vodcoco_tpa_prompts_convnextl.npy"
model.eval_query_path = "dataset/metadata/vodcoco_tpa_prompts_convnextl.npy"

# model.use_fed_loss = True
# model.cluster_fed_loss = True
# model.cluster_label_path = 'dataset/cluster/lvis_cluster_128.npy'
# model.cat_freq_path = "dataset/lvis/lvis_v1_train_norare_cat_info.json"
# model.fed_loss_num_cat=100
# model.select_box_nums_for_evaluation = 300

# modify optimizer config
# 假设你单卡用 1e-4 (1GPU); 1e-4 使用原来成功的学习率
base_lr = 1e-4
world_size = 1.5  # GPU 数
optimizer.lr = base_lr * world_size
optimizer.betas = (0.9, 0.999)
optimizer.weight_decay = 1e-4
optimizer.params.lr_factor_func = lambda module_name: 0.1 if "backbone" in module_name else 1

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
dataloader.test.dataset.names = "ovcoco_2017_val_all"
dataloader.train.dataset.names = "ovcoco_2017_train_all"

# ====== Phase 1：测试 Claude Prompts 单独效果 ======
# Fed Loss是LVIS必须的基础设施，所有实验都需要使用
model.use_fed_loss = False  # ✅ 必须开启，处理长尾分布
model.cluster_fed_loss = False
model.cluster_label_path = None
model.cat_freq_path = None
model.fed_loss_num_cat = 0  # 每次采样100个类别计算loss
model.select_box_nums_for_evaluation = 300

# Enable TPA (Text Prototype Aggregator) by modifying the classifier
model.classifier.use_tpa = True
model.classifier.text_embed_path = "dataset/metadata/vodcoco_tpa_prompts_convnextl.npy"
model.classifier.eval_text_embed_path = "dataset/metadata/vodcoco_tpa_prompts_convnextl.npy"
model.classifier.tpa_num_prototypes = 5  # 8 prompts -> 5 prototypes (better utilization)
model.classifier.tpa_hidden_dim = 256
model.classifier.tpa_dropout = 0.05  # Add dropout for regularization
model.classifier.tpa_tau = 0.07  # Optimal temperature for attention aggregation (τ ≈ 0.05–0.1 recommended) 0.1 is original
model.classifier.tpa_log_interval = 200


# Soft-attention aggregation parameters for multi-prototype query initialization
model.use_soft_attention = True  # Enable soft-attention aggregation in query initialization
model.soft_attention_tau = 0.08  # Temperature parameter for soft-attention (τ ≈ 0.05–0.1 recommended)

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

from detectron2.config import CfgNode as CN
if not hasattr(model, "cfg_overrides"):
    model.cfg_overrides = CN()
model.cfg_overrides.DATASETS = CN()
model.cfg_overrides.DATASETS.TRAIN = ("ovcoco_2017_train_all",)
model.cfg_overrides.DATASETS.TEST  = ("ovcoco_2017_val_all",)

from detectron2.data import MetadataCatalog
import numpy as np
_ov_meta = MetadataCatalog.get("ovcoco_2017_train_all")
assert len(_ov_meta.thing_classes) == 65, "thing_classes 不是 65！"
assert np.load(model.query_path).shape[0] == 65, ".npy 行数不是 65！"