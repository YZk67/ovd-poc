# 生成LVIS类别Text Embeddings指南

## 📋 概述

要生成 `lvis_visual_desc_convnextl.npy` (1203, 768) 文件，你需要：

1. ✅ LVIS类别信息JSON文件 (`lvis_v1_train_norare_cat_info.json`)
2. ✅ CLIP模型权重 (`clip_convnext_large_head.pth`)
3. ✅ 两个Python脚本：
   - `generate_class_prompts.py` - 生成文本prompts
   - `generate_text_embeddings.py` - 用CLIP编码文本

## 🚀 完整流程

### 步骤1: 安装依赖

```bash
cd /Users/zhengjiankang/Downloads/research/research/ovd-poc
source .venv/bin/activate

# 安装CLIP (选择一个)
# 方案A: OpenAI官方CLIP
pip install git+https://github.com/openai/CLIP.git

# 方案B: OpenCLIP (推荐，支持更多模型)
pip install open_clip_torch

# 其他依赖
pip install tqdm
```

### 步骤2: 生成文本Prompts

```bash
python tools/generate_class_prompts.py \
  --ann dataset2/lvis/lvis_v1_train_norare_cat_info.json \
  --prompt-output dataset2/metadata/lvis_prompts.json \
  --max-synonyms 5
```

**输出**: `dataset2/metadata/lvis_prompts.json`
- 格式: `{class_name: [prompt1, prompt2, ...]}`
- 每个类别有多个prompts (使用同义词和不同模板)

### 步骤3: 用CLIP生成Embeddings

```bash
python tools/generate_text_embeddings.py \
  --prompt-json dataset2/metadata/lvis_prompts.json \
  --clip-model pretrained_models/clip_convnext_large_head.pth \
  --output dataset2/metadata/lvis_visual_desc_convnextl.npy \
  --aggregate mean \
  --normalize
```

**参数说明**:
- `--prompt-json`: 输入的prompts文件
- `--clip-model`: CLIP文本编码器权重 (可选，不提供则使用默认预训练权重)
- `--output`: 输出的npy文件路径
- `--aggregate`: 如何聚合多个prompts
  - `mean`: 平均 (推荐)
  - `max`: 最大值
  - `first`: 只用第一个prompt
  - `none`: 保存所有prompts (输出shape: [1203, num_prompts, 768])
- `--normalize`: L2归一化 (CLIP推荐使用)

**输出**: `dataset2/metadata/lvis_visual_desc_convnextl.npy`
- Shape: `(1203, 768)`
- Dtype: `float32`

### 步骤4: 验证生成的文件

```bash
python -c "import numpy as np; arr=np.load('dataset2/metadata/lvis_visual_desc_convnextl.npy'); print(f'Shape: {arr.shape}, Dtype: {arr.dtype}')"
```

## 📝 文件说明

### 输入文件

#### `lvis_v1_train_norare_cat_info.json`
```json
[
  {
    "name": "aerosol_can",
    "synonyms": ["aerosol_can", "spray_can"],
    "def": "a dispenser that holds a substance under pressure",
    "id": 1,
    "frequency": "c",
    "image_count": 64,
    "instance_count": 109
  },
  ...
]
```

#### 生成的 `lvis_prompts.json`
```json
{
  "aerosol_can": [
    "a photo of a aerosol_can",
    "a close-up photo of a aerosol_can",
    "a photo of a spray_can",
    "a close-up photo of a spray_can",
    ...
  ],
  ...
}
```

### 输出文件

#### `lvis_visual_desc_convnextl.npy`
- **Shape**: `(1203, 768)`
  - 1203: LVIS类别数量
  - 768: CLIP text embedding维度 (ConvNeXt-Large)
- **Dtype**: `float32`
- **归一化**: L2 normalized (每个向量的L2范数为1)

## 🔍 进阶选项

### 1. 生成多个prompts而不聚合

如果你想保存所有prompts的embeddings (用于Text Prototype Aggregator):

```bash
python tools/generate_text_embeddings.py \
  --prompt-json dataset2/metadata/lvis_prompts.json \
  --output dataset2/metadata/lvis_all_prompts.npy \
  --aggregate none
```

输出shape: `(1203, num_prompts, 768)`

### 2. 从简单的类名列表生成

如果你只有类名列表 (`lvis_v1_all_classes.json`):

```bash
python tools/generate_text_embeddings.py \
  --prompt-json dataset2/lvis/lvis_v1_all_classes.json \
  --output dataset2/metadata/lvis_simple.npy \
  --aggregate first
```

这会直接使用类名，不使用模板和同义词。

### 3. 使用不同的CLIP模型

如果没有 `clip_convnext_large_head.pth`，脚本会自动使用预训练的CLIP:

```bash
# 使用默认OpenAI CLIP ViT-L/14
python tools/generate_text_embeddings.py \
  --prompt-json dataset2/metadata/lvis_prompts.json \
  --output dataset2/metadata/lvis_vitl14.npy \
  --aggregate mean
```

## 🎯 使用场景

### LVIS OV-Detection训练

在配置文件中使用生成的embedding:

```python
# lami_dino/configs/dino_convnext_large_4scale_12ep_lvis.py
model.query_path = "dataset2/metadata/lvis_visual_desc_convnextl.npy"
model.eval_query_path = "dataset2/metadata/lvis_visual_desc_convnextl.npy"
```

### Text Prototype Aggregator

如果使用TPA (Text Prototype Aggregator)，需要3D array:

```python
model.language.text_embed_path = "dataset2/metadata/lvis_all_prompts.npy"  # shape: [1203, N, 768]
```

## ⚠️ 注意事项

1. **CLIP模型必须匹配**: 
   - 训练和推理时使用的CLIP text encoder必须一致
   - `clip_convnext_large_head.pth` 对应 768维输出

2. **归一化很重要**:
   - CLIP embeddings应该L2归一化
   - 与visual features计算余弦相似度

3. **Prompt模板的影响**:
   - 不同的prompt模板会影响性能
   - 推荐使用多个模板并平均

4. **类别顺序**:
   - 确保类别ID顺序与annotation文件一致
   - 脚本会按照 `"id"` 字段排序

## 🔗 相关文件

- `tools/generate_class_prompts.py` - 生成prompts
- `tools/generate_text_embeddings.py` - 生成embeddings
- `lami_dino/modeling/text_classifier.py` - 使用embeddings的分类器
- `lami_dino/models/text_prototype_aggregator.py` - TPA模块

## 📚 参考

- CLIP论文: [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020)
- LaMI-DETR论文: [Open-Vocabulary Detection with Language Model Instruction](https://arxiv.org/abs/2407.11335)

