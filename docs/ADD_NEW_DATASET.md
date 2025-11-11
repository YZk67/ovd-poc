# 添加新数据集指南

## 概述

在Detectron2框架中添加新数据集（如 `ovd-coco`）**非常简单**，只需要修改**2-3个文件**即可。

## 前提条件

你的数据集需要是以下格式之一：
- **COCO格式**：标准COCO JSON标注格式（推荐，最简单）
- **LVIS格式**：LVIS JSON标注格式

## 快速开始：3步添加新数据集

### 步骤1：注册数据集（2种方式）

#### 方式A：在代码中直接注册（推荐，最简单）

在你的训练脚本或配置文件中添加：

```python
from detectron2.data.datasets import register_coco_instances

# 注册训练集
register_coco_instances(
    "ovd_coco_train",
    {},  # 元数据（可以为空）
    "path/to/ovd_coco/annotations/train.json",  # 标注文件路径
    "path/to/ovd_coco/images/train"  # 图片目录
)

# 注册验证集
register_coco_instances(
    "ovd_coco_val",
    {},
    "path/to/ovd_coco/annotations/val.json",
    "path/to/ovd_coco/images/val"
)
```

#### 方式B：在 builtin.py 中注册（永久注册）

编辑 `detectron2/detectron2/data/datasets/builtin.py`：

```python
# 在文件末尾的 register_all_* 函数之前添加
_PREDEFINED_SPLITS_OVD_COCO = {
    "ovd_coco": {
        "ovd_coco_train": ("ovd_coco/images/train", "ovd_coco/annotations/train.json"),
        "ovd_coco_val": ("ovd_coco/images/val", "ovd_coco/annotations/val.json"),
    },
}

def register_all_ovd_coco(root):
    for dataset_name, splits_per_dataset in _PREDEFINED_SPLITS_OVD_COCO.items():
        for key, (image_root, json_file) in splits_per_dataset.items():
            register_coco_instances(
                key,
                _get_builtin_metadata("coco"),  # 使用COCO元数据
                os.path.join(root, json_file) if "://" not in json_file else json_file,
                os.path.join(root, image_root),
            )

# 在文件末尾的 if __name__ 块中添加
if __name__.endswith(".builtin"):
    _root = os.path.expanduser(os.getenv("DETECTRON2_DATASETS", "dataset"))
    register_all_coco(_root)
    register_all_lvis(_root)
    register_all_ovd_coco(_root)  # 添加这一行
    # ... 其他注册
```

### 步骤2：创建数据配置文件

创建 `configs/common/data/ovd_coco_detr.py`：

```python
from omegaconf import OmegaConf

import detectron2.data.transforms as T
from detectron2.config import LazyCall as L
from detectron2.data import (
    build_detection_test_loader,
    build_detection_train_loader,
    get_detection_dataset_dicts,
)
from detectron2.data.samplers import RepeatFactorTrainingSampler
from detectron2.evaluation import COCOEvaluator  # 使用COCO评估器

from detrex.data import DetrDatasetMapper

dataloader = OmegaConf.create()

dataloader.train = L(build_detection_train_loader)(
    dataset=L(get_detection_dataset_dicts)(names="ovd_coco_train"),  # 修改数据集名称
    sampler="RepeatFactorTrainingSampler",
    repeat_threshold=0.001,
    mapper=L(DetrDatasetMapper)(
        augmentation=[
            L(T.RandomFlip)(),
            L(T.ResizeShortestEdge)(
                short_edge_length=(480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800),
                max_size=1333,
                sample_style="choice",
            ),
        ],
        augmentation_with_crop=[
            L(T.RandomFlip)(),
            L(T.ResizeShortestEdge)(
                short_edge_length=(400, 500, 600),
                sample_style="choice",
            ),
            L(T.RandomCrop)(
                crop_type="absolute_range",
                crop_size=(384, 600),
            ),
            L(T.ResizeShortestEdge)(
                short_edge_length=(480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800),
                max_size=1333,
                sample_style="choice",
            ),
        ],
        is_train=True,
        mask_on=False,
        img_format="RGB",
    ),
    total_batch_size=16,
    num_workers=4,
)

dataloader.test = L(build_detection_test_loader)(
    dataset=L(get_detection_dataset_dicts)(names="ovd_coco_val", filter_empty=False),  # 修改数据集名称
    mapper=L(DetrDatasetMapper)(
        augmentation=[
            L(T.ResizeShortestEdge)(
                short_edge_length=800,
                max_size=1333,
            ),
        ],
        augmentation_with_crop=None,
        is_train=False,
        mask_on=False,
        img_format="RGB",
    ),
    num_workers=1,
)

# 使用COCO评估器（如果是LVIS格式，则使用LVISEvaluator）
dataloader.evaluator = L(COCOEvaluator)(
    dataset_name="${..test.dataset.names}",
)
```

### 步骤3：创建训练配置文件

创建 `lami_dino/configs/dino_convnext_large_4scale_12ep_ovd_coco.py`：

```python
from detrex.config import get_config
from .models.dino_convnextl import model
from datetime import datetime

# 移除language配置（如果不是必需的）
if "language" in model:
    del model["language"]

# 配置模型参数
model.vlm_query_path = "dataset/metadata/ovd_coco_visual_desc_convnextl.npy"  # 如果有VLM embeddings
model.score_ensemble = True
model.backbone.score_ensemble = model.score_ensemble

# 如果是开放词汇检测，需要定义seen和unseen类
model.seen_classes = 'dataset/ovd_coco/ovd_coco_seen_classes.json'  # 如果有
model.all_classes = 'dataset/ovd_coco/ovd_coco_all_classes.json'    # 如果有
model.vlm_temperature = 100.0
model.alpha = 0.0
model.beta = 0.4
model.novel_scale = 5.0

# 获取数据配置
dataloader = get_config("common/data/ovd_coco_detr.py").dataloader  # 使用新创建的数据配置
optimizer = get_config("common/optim.py").AdamW
lr_multiplier = get_config("common/coco_schedule.py").lr_multiplier_12ep_warmup  # 或自定义
train = get_config("common/train.py").train

# 训练配置
train.seed = 42
train.init_checkpoint = "./pretrained_models/your_pretrained_model.pth"

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
train.output_dir = f"/path/to/output/ovd_coco_{timestamp}"

# 训练迭代次数（根据数据集大小调整）
# 假设数据集有50,000张图片，batch size 32
# 迭代次数 = (50000 / 32) * num_epochs
train.max_iter = 18750  # 12 epochs with batch size 32
train.eval_period = 1562  # 每个epoch评估一次

# 合并配置
cfg = get_config()
cfg.model = model
cfg.dataloader = dataloader
cfg.optimizer = optimizer
cfg.lr_multiplier = lr_multiplier
cfg.train = train

# 设置类别数量（重要！）
# 这会在模型配置中自动设置，但确保类别数量正确
# 如果你的数据集有80个类别，确保模型配置中的num_classes=80
```

## 完整示例

### 示例1：最简单的COCO格式数据集

假设你的数据集结构如下：
```
dataset/
  ovd_coco/
    images/
      train/
        img1.jpg
        img2.jpg
        ...
      val/
        img1.jpg
        ...
    annotations/
      train.json
      val.json
```

**只需要2个文件修改：**

1. **在训练脚本开头注册数据集**（或在builtin.py中）：
```python
from detectron2.data.datasets import register_coco_instances

register_coco_instances("ovd_coco_train", {}, 
    "dataset/ovd_coco/annotations/train.json",
    "dataset/ovd_coco/images/train")
register_coco_instances("ovd_coco_val", {},
    "dataset/ovd_coco/annotations/val.json", 
    "dataset/ovd_coco/images/val")
```

2. **创建数据配置文件** `configs/common/data/ovd_coco_detr.py`（复制 `coco_detr.py` 并修改数据集名称）

3. **创建训练配置**（复制现有的训练配置，修改数据加载器路径）

### 示例2：LVIS格式数据集

如果使用LVIS格式，只需要将 `register_coco_instances` 改为 `register_lvis_instances`，评估器改为 `LVISEvaluator`：

```python
from detectron2.data.datasets import register_lvis_instances

register_lvis_instances("ovd_coco_train", {},
    "dataset/ovd_coco/annotations/train.json",
    "dataset/ovd_coco/images/train")
```

评估器：
```python
from detectron2.evaluation import LVISEvaluator

dataloader.evaluator = L(LVISEvaluator)(
    dataset_name="${..test.dataset.names}",
)
```

## 需要修改的文件总结

| 任务 | 文件 | 修改内容 | 必需性 |
|------|------|----------|--------|
| 数据集注册 | `builtin.py` 或训练脚本 | 添加注册代码 | ✅ 必需 |
| 数据配置 | `configs/common/data/ovd_coco_detr.py` | 创建新文件，修改数据集名称 | ✅ 必需 |
| 训练配置 | `lami_dino/configs/xxx_ovd_coco.py` | 创建新文件，引用数据配置 | ✅ 必需 |
| 类别数量 | 模型配置 | 修改 `num_classes` | ⚠️ 如果类别数不同 |
| 元数据 | `builtin_meta.py` | 添加类别信息 | ❌ 可选 |

## 常见问题

### Q1: 需要修改多少代码？
**A:** 最少只需要**2-3个文件**：
1. 数据集注册（1个地方）
2. 数据配置文件（1个新文件）
3. 训练配置文件（1个新文件）

### Q2: 如果我的标注格式不是COCO/LVIS怎么办？
**A:** 你需要：
1. 编写一个加载函数，将你的格式转换为Detectron2格式
2. 使用 `DatasetCatalog.register()` 注册

### Q3: 如何设置类别数量？
**A:** Detectron2会自动从JSON文件中读取类别数量。如果需要在模型中手动设置：
```python
model.num_classes = 80  # 你的类别数量
```

### Q4: 需要修改模型代码吗？
**A:** 通常不需要。如果类别数量相同，可以直接使用现有模型。如果类别数量不同，只需要在配置中设置正确的 `num_classes`。

### Q5: 评估指标会自动计算吗？
**A:** 是的，使用 `COCOEvaluator` 或 `LVISEvaluator` 会自动计算AP、AP50、AP75等指标。

## 快速检查清单

- [ ] 数据集是COCO或LVIS格式
- [ ] 注册了训练集和验证集
- [ ] 创建了数据配置文件
- [ ] 创建了训练配置文件
- [ ] 设置了正确的类别数量
- [ ] 设置了正确的评估器（COCOEvaluator或LVISEvaluator）
- [ ] 检查了数据集路径是否正确

## 参考

- Detectron2数据集教程: `detectron2/docs/tutorials/datasets.md`
- COCO格式说明: http://cocodataset.org/#format-data
- LVIS格式说明: https://www.lvisdataset.org/

## 总结

添加新数据集**非常容易**，主要步骤：
1. ✅ 注册数据集（1行代码）
2. ✅ 创建数据配置文件（复制现有文件并修改名称）
3. ✅ 创建训练配置（复制现有配置并修改数据加载器）

**总共只需要修改2-3个文件，无需修改核心代码！**

