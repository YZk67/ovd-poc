# 从LVIS训练转换到OVD-COCO训练 - 详细计划

## 📋 概述

本计划详细说明如何将当前基于LVIS的训练配置转换为OVD-COCO训练配置。

## 🎯 目标

- 从LVIS数据集（1203类，866 seen + 337 unseen）切换到COCO zero-shot数据集（80类，48 base + 17 novel）
- 保持所有模型功能和训练流程不变
- 确保base/novel类正确区分

---

## 📁 阶段1：准备数据集文件

### 1.1 数据集目录结构

```
dataset/
  coco/
    images/
      train2017/          ← COCO训练图片
      val2017/            ← COCO验证图片
    zero-shot/            ← 新建：zero-shot标注目录
      instances_train2017_seen_2.json      ← 训练集（只包含48个base类）
      instances_val2017_all_2.json         ← 验证集（包含所有80类）
      instances_val2017_unseen_2.json      ← 验证集（只包含17个novel类，可选）
```

### 1.2 需要的标注文件

**如果已有COCO数据集，需要创建zero-shot标注文件：**

1. **训练集标注** (`instances_train2017_seen_2.json`)
   - 只包含48个base类的标注
   - 从原始COCO训练集中过滤出base类
   
2. **验证集标注** (`instances_val2017_all_2.json`)
   - 包含所有80个类别的标注（base + novel）
   - 用于完整评估

3. **验证集unseen标注** (`instances_val2017_unseen_2.json`, 可选)
   - 只包含17个novel类的标注
   - 用于单独评估novel类性能

### 1.3 创建类别列表文件

**需要创建3个JSON文件：**

```
dataset/coco/
  coco_seen_classes.json      ← 48个base类名称列表
  coco_unseen_classes.json    ← 17个novel类名称列表（可选）
  coco_all_classes.json       ← 80个所有类名称列表
```

**文件格式（字符串列表）：**

```json
// coco_seen_classes.json
[
  "toilet",
  "bicycle",
  "apple",
  "train",
  ...
]

// coco_all_classes.json
[
  "toilet",
  "bicycle",
  ...,
  "umbrella",  // novel类
  "cow",
  ...
]
```

---

## 🔧 阶段2：创建数据配置文件

### 2.1 创建数据加载配置

**文件：`configs/common/data/coco_zeroshot_detr.py`**

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
from detectron2.evaluation import COCOEvaluator
from detrex.data import DetrDatasetMapper

dataloader = OmegaConf.create()

dataloader.train = L(build_detection_train_loader)(
    dataset=L(get_detection_dataset_dicts)(names="zeroshot_coco_2017_train"),  # 使用zero-shot训练集
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
    dataset=L(get_detection_dataset_dicts)(names="zeroshot_coco_2017_val", filter_empty=False),
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

dataloader.evaluator = L(COCOEvaluator)(
    dataset_name="${..test.dataset.names}",
)
```

### 2.2 验证数据集注册

**检查 `detectron2/detectron2/data/datasets/builtin.py` 中是否已有zero-shot注册：**

```python
_PREDEFINED_SPLITS_COCO["coco_zeroshot"] = {
    "zeroshot_coco_2017_train": ("coco/train2017", "coco/zero-shot/instances_train2017_seen_2_proposal.json"),
    "zeroshot_coco_2017_val": ("coco/val2017", "coco/zero-shot/instances_val2017_all_2.json"),
    "zeroshot_coco_2017_val_unseen": ("coco/val2017", "coco/zero-shot/zeroshot_unseen.json"),
}
```

**如果没有，需要添加或使用自定义注册。**

---

## 🤖 阶段3：准备文本Embeddings

### 3.1 生成类别文本Embeddings

**⚠️ 重要：需要为COCO的所有80个类别生成文本embeddings，包括48个base类和17个novel类！**

**原因：**
- 开放词汇检测的核心是通过文本embeddings理解所有类别的语义
- 训练时虽然只有48个base类的标注，但模型需要知道所有80个类别的语义信息
- 推理时模型需要对所有80个类别（base + novel）进行预测
- 文本embeddings提供类别的语义表示，使得模型能够识别训练时未见过的novel类

**步骤：**

1. **生成类别prompts**
   - 使用 `tools/generate_class_prompts.py` 或类似工具
   - **输入：`coco_all_classes.json`（包含所有80个类别）**
   - 为每个类别生成多个prompts（如8个）
   - 保存为JSON格式

2. **生成文本embeddings**
   - 使用 `tools/generate_text_embeddings.py`
   - **输入：包含所有80个类别prompts的JSON文件**
   - **输出：`.npy`文件，形状为 `[80, num_prompts, embed_dim]`**
   - 例如：`dataset/metadata/coco_claude_prompts_convnextl.npy`
   - **⚠️ 注意：必须是80个类别，不是48个！**

3. **生成VLM embeddings（如果使用score_ensemble）**
   - 生成视觉描述embeddings
   - **同样需要包含所有80个类别**
   - 例如：`dataset/metadata/coco_visual_desc_convnextl.npy`

### 3.2 文件清单

```
dataset/metadata/
  coco_claude_prompts_convnextl.npy        ← 所有80个类别的文本embeddings（必需）
  coco_visual_desc_convnextl.npy           ← 所有80个类别的VLM embeddings（如果使用score_ensemble）
  coco_prompts_claude.json                 ← 所有80个类别的原始prompts（可选）
```

### 3.3 验证Embeddings

**验证embeddings文件是否正确：**

```python
import numpy as np
import json

# 加载embeddings
emb = np.load('dataset/metadata/coco_claude_prompts_convnextl.npy')
print(f"Embeddings形状: {emb.shape}")  # 应该是 [80, num_prompts, embed_dim]

# 加载类别列表
with open('dataset/coco/coco_all_classes.json', 'r') as f:
    all_classes = json.load(f)

# 验证类别数量
assert emb.shape[0] == len(all_classes) == 80, \
    f"Embeddings类别数({emb.shape[0]}) != 类别列表数量({len(all_classes)}) != 80"
print("✅ Embeddings包含所有80个类别")
```

---

## ⚙️ 阶段4：创建训练配置文件

### 4.1 创建训练配置

**文件：`lami_dino/configs/dino_convnext_large_4scale_12ep_coco_zeroshot.py`**

**关键修改点：**

```python
from detrex.config import get_config
from .models.dino_convnextl import model
from datetime import datetime

# Remove 'language' key
if "language" in model:
    del model["language"]

# ====== 1. 修改文本embedding路径 ======
model.vlm_query_path = "dataset/metadata/coco_visual_desc_convnextl.npy"  # 改为COCO
model.score_ensemble = True
model.backbone.score_ensemble = model.score_ensemble

# ====== 2. 修改类别列表路径 ======
model.seen_classes = 'dataset/coco/coco_seen_classes.json'      # 48个base类
model.all_classes = 'dataset/coco/coco_all_classes.json'        # 80个所有类
# model.unseen_classes = 'dataset/coco/coco_unseen_classes.json'  # 可选：17个novel类

model.vlm_temperature = 100.0
model.alpha = 0.0
model.beta = 0.4
model.novel_scale = 5.0

# ====== 3. 修改数据加载器 ======
dataloader = get_config("common/data/coco_zeroshot_detr.py").dataloader  # 使用COCO zero-shot配置
optimizer = get_config("common/optim.py").AdamW
lr_multiplier = get_config("common/coco_schedule.py").lr_multiplier_12ep_warmup  # 使用COCO schedule
train = get_config("common/train.py").train

# ====== 4. 训练配置 ======
train.seed = 42
train.init_checkpoint = "./pretrained_models/your_pretrained_model.pth"  # 可能需要修改

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
train.output_dir = f"/path/to/output/coco_zeroshot_{timestamp}"

# ====== 5. 修改迭代次数（COCO有118K训练图片） ======
# COCO: 118,287 training images
# Batch size 32: 118,287 ÷ 32 = 3,696 iter/epoch
# 12 epochs: 3,696 × 12 = 44,352 iterations
train.max_iter = 44352  # 12 epochs with batch size 32
train.eval_period = 3696  # 每个epoch评估一次
train.log_period = 50
train.checkpointer.period = 3696  # 每个epoch保存一次

# ====== 6. 修改类别数量 ======
model.num_classes = 80  # COCO有80个类别（不是1203）

# ====== 7. 修改文本embedding路径 ======
model.query_path = "dataset/metadata/coco_claude_prompts_convnextl.npy"
model.eval_query_path = "dataset/metadata/coco_claude_prompts_convnextl.npy"

# ====== 8. Fed Loss配置（COCO不需要，但可以保留） ======
# COCO类别数量少（80类），通常不需要Fed Loss
# 但如果想保持一致性，可以保留
model.use_fed_loss = False  # COCO不需要Fed Loss
# model.use_fed_loss = True
# model.fed_loss_num_cat = 50  # 可以设置为较小的值

# ====== 9. TPA配置 ======
model.classifier.use_tpa = True
model.classifier.text_embed_path = "dataset/metadata/coco_claude_prompts_convnextl.npy"
model.classifier.eval_text_embed_path = "dataset/metadata/coco_claude_prompts_convnextl.npy"
model.classifier.tpa_num_prototypes = 5
model.classifier.tpa_hidden_dim = 256
model.classifier.tpa_dropout = 0.05
model.classifier.tpa_tau = 0.07

# ====== 10. 其他配置（保持与LVIS相同） ======
model.use_soft_attention = True
model.soft_attention_tau = 0.08

# RPSA配置（如果使用）
model.transformer.use_rpsa = True
model.criterion.weight_dict["loss_rpsa"] = 0.05
# ... 其他RPSA参数

# 优化器配置
base_lr = 1e-4
world_size = 1.5
optimizer.lr = base_lr * world_size
optimizer.betas = (0.9, 0.999)
optimizer.weight_decay = 1e-4
optimizer.params.lr_factor_func = lambda module_name: 0.1 if "backbone" in module_name else 1

# 数据加载器配置
dataloader.train.num_workers = 4
dataloader.train.total_batch_size = 16

# 评估器配置
dataloader.evaluator.output_dir = train.output_dir
dataloader.test.dataset.names = "zeroshot_coco_2017_val"

# AMP配置
train.amp.enabled = True
```

---

## 📝 阶段5：创建辅助脚本

### 5.1 创建类别列表文件生成脚本

**文件：`tools/create_coco_zeroshot_class_lists.py`**

```python
#!/usr/bin/env python3
"""
创建COCO zero-shot的类别列表文件
"""
import json
from detectron2.data.datasets.builtin_meta import COCO_SEEN_CLASSES, COCO_UNSEEN_CLASSES

# 获取所有COCO类别
from detectron2.data.datasets.builtin_meta import COCO_CATEGORIES
all_coco_classes = [cat['name'] for cat in COCO_CATEGORIES if cat['isthing'] == 1]

# 创建seen classes列表
seen_classes = list(COCO_SEEN_CLASSES)

# 创建unseen classes列表
unseen_classes = list(COCO_UNSEEN_CLASSES)

# 创建all classes列表（按COCO顺序）
all_classes = all_coco_classes

# 保存文件
output_dir = "dataset/coco"
import os
os.makedirs(output_dir, exist_ok=True)

with open(f"{output_dir}/coco_seen_classes.json", 'w') as f:
    json.dump(seen_classes, f, indent=2)

with open(f"{output_dir}/coco_unseen_classes.json", 'w') as f:
    json.dump(unseen_classes, f, indent=2)

with open(f"{output_dir}/coco_all_classes.json", 'w') as f:
    json.dump(all_classes, f, indent=2)

print(f"✅ 创建类别列表文件:")
print(f"   - {output_dir}/coco_seen_classes.json ({len(seen_classes)} 类)")
print(f"   - {output_dir}/coco_unseen_classes.json ({len(unseen_classes)} 类)")
print(f"   - {output_dir}/coco_all_classes.json ({len(all_classes)} 类)")
```

### 5.2 创建zero-shot标注文件过滤脚本

**文件：`tools/filter_coco_zeroshot_annotations.py`**

```python
#!/usr/bin/env python3
"""
从COCO标注文件中过滤出zero-shot标注
- 训练集：只保留seen类的标注
- 验证集：保留所有类的标注
"""
import json
from detectron2.data.datasets.builtin_meta import COCO_SEEN_CLASSES, COCO_UNSEEN_CLASSES

def filter_annotations(input_json, output_json, keep_classes=None):
    """
    过滤标注文件，只保留指定类别的标注
    
    Args:
        input_json: 输入JSON文件路径
        output_json: 输出JSON文件路径
        keep_classes: 要保留的类别名称集合，如果为None则保留所有
    """
    with open(input_json, 'r') as f:
        data = json.load(f)
    
    # 获取类别ID到名称的映射
    cat_id_to_name = {cat['id']: cat['name'] for cat in data['categories']}
    
    if keep_classes:
        # 只保留指定类别的categories
        data['categories'] = [cat for cat in data['categories'] 
                             if cat['name'] in keep_classes]
        keep_cat_ids = {cat['id'] for cat in data['categories']}
        
        # 过滤annotations
        data['annotations'] = [ann for ann in data['annotations']
                              if ann['category_id'] in keep_cat_ids]
        
        # 获取有标注的图片ID
        image_ids_with_annotations = {ann['image_id'] for ann in data['annotations']}
        
        # 过滤images（只保留有标注的图片）
        data['images'] = [img for img in data['images']
                         if img['id'] in image_ids_with_annotations]
    
    # 保存过滤后的文件
    with open(output_json, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"✅ 过滤完成: {output_json}")
    print(f"   类别数: {len(data['categories'])}")
    print(f"   图片数: {len(data['images'])}")
    print(f"   标注数: {len(data['annotations'])}")

# 使用示例
if __name__ == "__main__":
    # 过滤训练集（只保留seen类）
    filter_annotations(
        "dataset/coco/annotations/instances_train2017.json",
        "dataset/coco/zero-shot/instances_train2017_seen_2.json",
        keep_classes=set(COCO_SEEN_CLASSES)
    )
    
    # 验证集保留所有类（不需要过滤）
    # 如果需要，可以创建只包含unseen类的验证集
    filter_annotations(
        "dataset/coco/annotations/instances_val2017.json",
        "dataset/coco/zero-shot/instances_val2017_unseen_2.json",
        keep_classes=set(COCO_UNSEEN_CLASSES)
    )
```

---

## ✅ 阶段6：检查清单

### 6.1 文件检查清单

- [ ] **数据集文件**
  - [ ] `dataset/coco/zero-shot/instances_train2017_seen_2.json` (训练集，48类)
  - [ ] `dataset/coco/zero-shot/instances_val2017_all_2.json` (验证集，80类)
  - [ ] `dataset/coco/images/train2017/` (训练图片)
  - [ ] `dataset/coco/images/val2017/` (验证图片)

- [ ] **类别列表文件**
  - [ ] `dataset/coco/coco_seen_classes.json` (48个base类)
  - [ ] `dataset/coco/coco_all_classes.json` (80个所有类)
  - [ ] `dataset/coco/coco_unseen_classes.json` (17个novel类，可选)

- [ ] **文本Embeddings**
  - [ ] `dataset/metadata/coco_claude_prompts_convnextl.npy` (类别embeddings)
  - [ ] `dataset/metadata/coco_visual_desc_convnextl.npy` (VLM embeddings，如果使用)

- [ ] **配置文件**
  - [ ] `configs/common/data/coco_zeroshot_detr.py` (数据配置)
  - [ ] `lami_dino/configs/dino_convnext_large_4scale_12ep_coco_zeroshot.py` (训练配置)

### 6.2 配置检查清单

- [ ] **模型配置**
  - [ ] `model.num_classes = 80` (COCO有80类)
  - [ ] `model.seen_classes` 路径正确
  - [ ] `model.all_classes` 路径正确
  - [ ] `model.query_path` 路径正确
  - [ ] `model.vlm_query_path` 路径正确（如果使用）

- [ ] **数据配置**
  - [ ] 训练集名称：`zeroshot_coco_2017_train`
  - [ ] 验证集名称：`zeroshot_coco_2017_val`
  - [ ] 评估器：`COCOEvaluator`

- [ ] **训练配置**
  - [ ] `train.max_iter` 根据数据集大小调整
  - [ ] `train.eval_period` 设置合理
  - [ ] `dataloader.train.total_batch_size` 设置合理
  - [ ] `model.use_fed_loss` 根据需求设置（COCO通常不需要）

---

## 🚀 阶段7：执行步骤

### 步骤1：准备数据集

```bash
# 1. 创建目录结构
mkdir -p dataset/coco/zero-shot
mkdir -p dataset/coco/images

# 2. 下载COCO数据集（如果还没有）
# 下载 train2017.zip, val2017.zip, annotations_trainval2017.zip

# 3. 解压文件
unzip train2017.zip -d dataset/coco/images/
unzip val2017.zip -d dataset/coco/images/
unzip annotations_trainval2017.zip -d dataset/coco/

# 4. 创建类别列表文件
python tools/create_coco_zeroshot_class_lists.py

# 5. 过滤标注文件
python tools/filter_coco_zeroshot_annotations.py
```

### 步骤2：生成文本Embeddings

```bash
# 1. 生成类别prompts（如果还没有）
python tools/generate_class_prompts.py \
    --classes dataset/coco/coco_all_classes.json \
    --output dataset/metadata/coco_prompts_claude.json

# 2. 生成文本embeddings
python tools/generate_text_embeddings.py \
    --prompts dataset/metadata/coco_prompts_claude.json \
    --output dataset/metadata/coco_claude_prompts_convnextl.npy

# 3. 生成VLM embeddings（如果使用score_ensemble）
# 根据你的VLM embedding生成流程
```

### 步骤3：创建配置文件

```bash
# 1. 创建数据配置文件
# 复制并修改 configs/common/data/coco_zeroshot_detr.py

# 2. 创建训练配置文件
# 复制 lami_dino/configs/dino_convnext_large_4scale_12ep_lvis.py
# 修改为 lami_dino/configs/dino_convnext_large_4scale_12ep_coco_zeroshot.py
```

### 步骤4：验证配置

```bash
# 1. 验证数据集注册
python -c "from detectron2.data import DatasetCatalog; print('zeroshot_coco_2017_train' in DatasetCatalog.list())"

# 2. 验证类别列表文件
python -c "import json; print(len(json.load(open('dataset/coco/coco_seen_classes.json'))))"  # 应该输出48
python -c "import json; print(len(json.load(open('dataset/coco/coco_all_classes.json'))))"    # 应该输出80

# 3. 验证embeddings文件
python -c "import numpy as np; emb = np.load('dataset/metadata/coco_claude_prompts_convnextl.npy'); print(emb.shape)"  # 应该输出 [80, num_prompts, embed_dim]
```

### 步骤5：开始训练

```bash
# 使用训练配置文件
python tools/train_net.py --config-file lami_dino/configs/dino_convnext_large_4scale_12ep_coco_zeroshot.py
```

---

## 📊 关键差异总结

| 项目 | LVIS | OVD-COCO |
|------|------|----------|
| **总类别数** | 1203 | 80 |
| **Base/Seen类** | 866 | 48 |
| **Novel/Unseen类** | 337 | 17 |
| **训练图片数** | 100,170 | 118,287 |
| **验证图片数** | 19,809 | 5,000 |
| **Fed Loss** | ✅ 必需 | ❌ 通常不需要 |
| **评估器** | LVISEvaluator | COCOEvaluator |
| **文本Embeddings** | lvis_claude_prompts_convnextl.npy | coco_claude_prompts_convnextl.npy |
| **类别列表** | lvis_v1_seen_classes.json | coco_seen_classes.json |

---

## 🔍 常见问题

### Q1: 训练集标注文件应该包含哪些类别？

**A:** 训练集标注文件应该**只包含48个base类的标注**。虽然`categories`字段可以包含所有80个类别，但`annotations`字段中只应该有base类的标注。

### Q2: 验证集标注文件应该包含哪些类别？

**A:** 验证集标注文件应该**包含所有80个类别的标注**（base + novel），用于完整评估模型的开放词汇检测能力。

### Q3: 是否需要Fed Loss？

**A:** COCO数据集类别数量较少（80类），通常不需要Fed Loss。但如果想保持与LVIS训练的一致性，可以启用Fed Loss，但`fed_loss_num_cat`应该设置为较小的值（如50）。

### Q4: 如何验证base/novel类划分是否正确？

**A:** 
1. 检查训练集标注文件中是否只包含base类的标注
2. 检查验证集标注文件中是否包含所有类的标注
3. 检查类别列表文件中的类别数量是否正确

### Q5: 文本embeddings的维度需要修改吗？

**A:** 不需要。文本embeddings的维度（embed_dim）由模型决定，与类别数量无关。只需要确保embeddings的第一维是类别数量（80）。

### Q6: 文本embeddings需要为所有类别生成，还是只为base类生成？

**A:** **必须为所有80个类别生成！** 这是开放词汇检测的核心机制：
- 训练数据只包含48个base类的标注
- 但文本embeddings需要包含所有80个类别（base + novel）
- 模型通过文本语义理解所有类别，从而能够识别训练时未见过的novel类
- 推理时模型对所有80个类别进行预测

**验证方法：**
```python
import numpy as np
emb = np.load('dataset/metadata/coco_claude_prompts_convnextl.npy')
assert emb.shape[0] == 80, f"应该是80个类别，但得到{emb.shape[0]}个"
```

---

## 📚 参考资料

- COCO Zero-Shot类别定义：`detectron2/detectron2/data/datasets/builtin_meta.py`
- COCO数据集注册：`detectron2/detectron2/data/datasets/builtin.py`
- LVIS训练配置参考：`lami_dino/configs/dino_convnext_large_4scale_12ep_lvis.py`

---

## ✅ 完成检查

完成所有步骤后，确认：

- [ ] 所有文件路径正确
- [ ] 类别数量正确（80类）
- [ ] 训练集只包含base类标注
- [ ] 验证集包含所有类标注
- [ ] 文本embeddings文件存在且格式正确
- [ ] 配置文件中的路径都正确
- [ ] 可以成功加载数据集
- [ ] 可以成功开始训练

