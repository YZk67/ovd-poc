from detrex.config import get_config
from .models.dino_convnextl import model
from datetime import datetime
from copy import deepcopy

# Remove 'language' key from model config as it's not a parameter for DINO.__init__()
# The language config is only used for TextClassifier via ${..language.xxx} references
if "language" in model:
    del model["language"]

model.vlm_query_path = "dataset/metadata/vodcoco_tpa_prompts_convnextl.npy"
model.score_ensemble = True
model.backbone.score_ensemble = model.score_ensemble
model.seen_classes = "dataset/metadata/ovcoco_seen_classes.json"
model.all_classes = "dataset/metadata/ovcoco_all_classes.json"
model.vlm_temperature = 100.0 # keep same with f-vlm
model.alpha = 0.0
model.beta = 0.4
model.novel_scale = 5.0

# --- 1) 确保注册代码被执行（很关键，不然 Dataset/Metadata 都是空的） ---
try:
    import lami_dino.data.builtin as _ov_builtin  # 按你项目实际注册模块路径
except Exception:
    pass

# --- 2) 若 metadata 为空，用 all_classes.json 手动补上 thing_classes ---
import json
from detectron2.data import MetadataCatalog, DatasetCatalog

assert "ovcoco_2017_val_all" in DatasetCatalog.list(), "未注册 ovcoco_2017_val_all（检查 builtin 注册是否被导入）"

_meta = MetadataCatalog.get("ovcoco_2017_val_all")
if not hasattr(_meta, "thing_classes") or not _meta.thing_classes:
    with open(model.all_classes, "r") as f:
        names = json.load(f)  # 期望为 65 个类名，顺序要与你的 .npy 一致
    assert len(names) == 65, "all_classes.json 中的类数不是 65！"
    _meta.thing_classes = names  # ★ 填入 65 类，解除后续断言/评测报错

# get default config
#dataloader = get_config("common/data/coco_detr.py").dataloader
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

model.num_classes = 65  # OV-Coco 65 classes
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
world_size = 4  # GPU 数
optimizer.lr = base_lr * world_size
optimizer.betas = (0.9, 0.999)
optimizer.weight_decay = 1e-4
optimizer.params.lr_factor_func = lambda module_name: 0.1 if "backbone" in module_name else 1

# modify dataloader config
# Start with conservative setting, can be increased if stable
#dataloader.train.num_workers = 4  # 1 worker per GPU for 4GPU training

# please notice that this is total batch size.
# surpose you're using 4 gpus for training and the batch size for
# each gpu is 16/4 = 4
# Note: Using Option 3 (averaged embeddings), batch_size can remain at 4
# If using Option 2 (6015 queries), reduce to batch_size=1
#dataloader.train.total_batch_size = 16  # Can use 4 with Option 3

# dump the testing results into output_dir for visualization
#dataloader.evaluator.output_dir = train.output_dir
dataloader = {}

# ---- dataloader: train + two separate tests (no .clone()) ----
_base_dl = get_config("common/data/coco_detr.py").dataloader

# 训练集：从基模板拷一份，并改成 ov-coco base
dataloader["train"] = deepcopy(_base_dl.train)
dataloader["train"].dataset.names = "ovcoco_2017_train_b"
dataloader["train"].num_workers = 4
dataloader["train"].total_batch_size = 16  # 4卡×每卡4

# 验证集1：base(48)
# tb = deepcopy(_base_dl.test)
# tb.dataset.names = "ovcoco_2017_val_b"
# tb.num_workers = 0   # 单进程评测更稳（可按需改）

# # 验证集2：novel(17)
# tt = deepcopy(_base_dl.test)
# tt.dataset.names = "ovcoco_2017_val_t"
# tt.num_workers = 0

# dataloader["test"] = tt

# 验证集：all(65)
ta = deepcopy(_base_dl.test)
ta.dataset.names = "ovcoco_2017_val_all"   # 统一 65 类评测
ta.num_workers = 0
dataloader["test"] = ta

# ★★★ 关键：给出 evaluator（否则会评成 {}）
dataloader["evaluator"] = deepcopy(_base_dl.evaluator)
# 有些 LazyConfig 里 evaluator 需要 dataset_name
#（如果你的 _base_dl.evaluator 是 CocoEvaluator 风格，这行可有可无）
try:
    dataloader["evaluator"].dataset_name = "ovcoco_2017_val_all"
except Exception:
    pass
# 指定输出目录，方便保存 metrics.json
dataloader["evaluator"].output_dir = train.output_dir

# —— 一致性断言，避免 silent failure —— #
from detectron2.data import MetadataCatalog
import numpy as np
_meta = MetadataCatalog.get("ovcoco_2017_val_all")
assert len(_meta.thing_classes) == 65, "ovcoco_2017_val_all 的 thing_classes 不是 65！"
assert np.load(model.eval_query_path).shape[0] == 65, "eval_text_embed 的 .npy 行数不是 65！"

# —— 训练/评测类别顺序自检与对齐 —— #
# 放在你两个断言之后：
#   assert len(_meta.thing_classes) == 65
#   assert np.load(model.eval_query_path).shape[0] == 65

# 1) 触发加载，拿到 train_b 的真实顺序（长度=48）
_ = DatasetCatalog.get("ovcoco_2017_train_b")
tc_train = MetadataCatalog.get("ovcoco_2017_train_b").thing_classes  # 训练端 48 类名（顺序=Detectron2的连续 id 0..47）

# 2) 读取全局 65 顺序（与你 .npy 行顺序一致）
with open(model.all_classes, "r") as f:
    coco65 = json.load(f)
assert len(coco65) == 65

# 3) 构造 “训练48类 -> 65类下标” 映射
name2idx65 = {n: i for i, n in enumerate(coco65)}
try:
    train48_to_65 = [name2idx65[n] for n in tc_train]  # len=48
except KeyError as e:
    raise ValueError(f"[类名不匹配] {e}. 请确保 train_b JSON 的类名与 all_classes(=COCO65) 完全一致（空格/连字符/大小写）。")

# 4) 将映射交给模型/分类头（按你的代码结构，这里放到 model/classifier 最稳妥）
model.classifier.train_class_indices = train48_to_65   # 训练时只取这 48 列进行监督
model.classifier.eval_class_indices  = list(range(65)) # 评测时走全 65 列
model.label_map_48to65               = {i:k for i,k in enumerate(train48_to_65)}  # 可供 loss/可视化使用

# （可选）如果你在别处读取 seen_classes.json 来切 .npy，
# 建议把 seen 改成与 tc_train 完全一致，避免顺序错位：
try:
    with open(model.seen_classes, "r") as f:
        seen_list = json.load(f)
    if seen_list != tc_train:
        print("[WARN] seen_classes.json 的顺序 ≠ 训练端实际顺序；建议用 tc_train 覆盖或统一以 tc_train 为准。")
except Exception:
    pass

# evaluator：给一个就够，你的 do_test() 会在数量不匹配时复用
#dataloader["evaluator"] = get_config("common/data/coco_detr.py").dataloader.evaluator



# dataloader.evaluator = [
#     get_config("common/data/coco_detr.py").dataloader.evaluator.clone(),
#     get_config("common/data/coco_detr.py").dataloader.evaluator.clone(),
# ]
#dataloader.test.dataset.names = ("ovcoco_2017_val_b", "ovcoco_2017_val_t")
# dataloader.test = [
#     get_config("common/data/coco_detr.py").dataloader.test.clone()
# ]
#dataloader.test[0].dataset.names = "ovcoco_2017_val_b"

# dataloader.test += [
#     get_config("common/data/coco_detr.py").dataloader.test.clone()
# ]
#dataloader.test[1].dataset.names = "ovcoco_2017_val_t"
#dataloader["train"].dataset.names = "ovcoco_2017_train_b"

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

# from detectron2.config import CfgNode as CN
# if not hasattr(model, "cfg_overrides"):
#     model.cfg_overrides = CN()
# model.cfg_overrides.DATASETS = CN()
# model.cfg_overrides.DATASETS.TRAIN = ("ovcoco_2017_train_b",)
# model.cfg_overrides.DATASETS.TEST  = ("ovcoco_2017_val_b", "ovcoco_2017_val_t")

# from detectron2.data import MetadataCatalog
# import numpy as np
# #_ov_meta = MetadataCatalog.get("ovcoco_2017_train_all")
# for _name in ["ovcoco_2017_train_b", "ovcoco_2017_val_b", "ovcoco_2017_val_t"]:
#     _meta = MetadataCatalog.get(_name)
#     assert len(_meta.thing_classes) == 65, f"{_name} thing_classes 不是 65！"
# #assert len(_ov_meta.thing_classes) == 65, "thing_classes 不是 65！"
# assert np.load(model.query_path).shape[0] == 65, ".npy 行数不是 65！"