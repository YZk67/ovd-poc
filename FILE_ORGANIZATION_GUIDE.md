# 文件组织指南

## ✅ 软链接已删除

所有软链接已经删除，并创建了实际的目录结构。

---

## 📁 目录结构

```
ovd-poc/
├── dataset/                          # 主数据集目录（需要填充）
│   ├── metadata/                     # 元数据文件
│   │   ├── ovcoco_seen_classes.json       [需要复制]
│   │   ├── ovcoco_all_classes.json        [需要复制]
│   │   ├── ovdcoco_prompts_list8_v2.npy   [需要生成或复制]
│   │   └── ovdcoco_vlm_query_convnextl.npy [需要生成或复制]
│   └── coco/                         # COCO 数据集
│       ├── annotations/              # 标注文件
│       │   ├── ovd_ins_train2017_b.json   [需要复制]
│       │   └── ovd_ins_val2017_all.json   [需要复制]
│       ├── train2017/                # 训练图像
│       └── val2017/                  # 验证图像
│
├── dataset2/                         # 现有数据源
│   ├── metadata/                     # 源文件位置
│   │   ├── ovcoco_seen_classes.json       ✅ 已有
│   │   ├── ovcoco_all_classes.json        ✅ 已有
│   │   └── vodcoco_tpa_prompts_convnextl.npy ✅ 已有
│   ├── coco/                         
│   └── lvis/                         
│
├── pretrained_models/                # 预训练模型（需要填充）
│   └── clip_convnext_large_head.pth       [需要复制]
│
├── pretrained_models2/               # 现有预训练模型
│   └── clip_convnext_large_head.pth       ✅ 已有
│
└── output/                           # 训练输出（训练后生成）
    ├── model_final.pth               [训练后生成]
    └── checkpoint_*.pth              [训练后生成]
```

---

## 📝 需要复制的文件清单

### 1️⃣ 类别定义文件（已有，需要复制）

```bash
# 从 dataset2 复制到 dataset
cp dataset2/metadata/ovcoco_seen_classes.json dataset/metadata/
cp dataset2/metadata/ovcoco_all_classes.json dataset/metadata/
```

**验证**:
```bash
ls -lh dataset/metadata/ovcoco*.json
```

---

### 2️⃣ 文本嵌入文件（需要生成或复制）

#### 选项A: 使用现有文件（临时测试）

如果 `vodcoco_tpa_prompts_convnextl.npy` 可以作为替代：

```bash
# 复制并重命名
cp dataset2/metadata/vodcoco_tpa_prompts_convnextl.npy \
   dataset/metadata/ovdcoco_prompts_list8_v2.npy

cp dataset2/metadata/vodcoco_tpa_prompts_convnextl.npy \
   dataset/metadata/ovdcoco_vlm_query_convnextl.npy
```

#### 选项B: 生成新的嵌入文件（推荐）

参见下面的"生成文本嵌入"章节。

**验证**:
```bash
ls -lh dataset/metadata/*.npy
```

---

### 3️⃣ CLIP 预训练模型

```bash
# 从 pretrained_models2 复制到 pretrained_models
cp pretrained_models2/clip_convnext_large_head.pth \
   pretrained_models/
```

**验证**:
```bash
ls -lh pretrained_models/*.pth
```

---

### 4️⃣ COCO 数据集标注文件

如果你有 COCO OVD 标注文件，需要放在：

```bash
dataset/coco/annotations/ovd_ins_train2017_b.json      # 训练集（base类）
dataset/coco/annotations/ovd_ins_val2017_all.json     # 验证集（全部类）
```

如果在 `dataset2/coco/annotations/` 中有这些文件：

```bash
# 检查源文件
ls dataset2/coco/annotations/ovd_ins_*.json

# 复制
cp dataset2/coco/annotations/ovd_ins_train2017_b.json \
   dataset/coco/annotations/

cp dataset2/coco/annotations/ovd_ins_val2017_all.json \
   dataset/coco/annotations/
```

**验证**:
```bash
ls -lh dataset/coco/annotations/*.json
```

---

### 5️⃣ COCO 图像（如果需要）

```bash
# COCO 训练图像（可选，如果要重新训练）
dataset/coco/train2017/

# COCO 验证图像（推荐，用于评估和可视化）
dataset/coco/val2017/
```

如果 `dataset2` 中有图像：

```bash
# 方案1: 复制（需要大量空间）
cp -r dataset2/coco/train2017/* dataset/coco/train2017/
cp -r dataset2/coco/val2017/* dataset/coco/val2017/

# 方案2: 在配置中直接指向 dataset2（推荐）
# 修改配置文件中的图像路径
```

---

## 🔧 生成文本嵌入文件

如果你需要生成新的嵌入文件：

### 步骤1: 创建 COCO prompts JSON

创建 `dataset/metadata/ovdcoco_prompts_list8.json`:

```json
{
  "person": [
    "a photo of a person",
    "a person in the scene",
    "there is a person",
    "a picture of a person",
    "a person standing",
    "a human",
    "people",
    "someone"
  ],
  "bicycle": [
    "a photo of a bicycle",
    "a bicycle in the scene",
    ...
  ],
  ...
}
```

或者使用类别列表生成简单版本：

```bash
python tools/generate_class_prompts.py \
  --classes dataset/metadata/ovcoco_all_classes.json \
  --prompt-output dataset/metadata/ovdcoco_prompts_list8.json \
  --num-prompts 8
```

### 步骤2: 生成嵌入

```bash
# 生成多prompt版本 (用于 TPA)
python tools/generate_text_embeddings.py \
  --prompt-json dataset/metadata/ovdcoco_prompts_list8.json \
  --clip-model pretrained_models/clip_convnext_large_head.pth \
  --output dataset/metadata/ovdcoco_prompts_list8_v2.npy \
  --aggregate none \
  --normalize

# 生成聚合版本 (用于 VLM query)
python tools/generate_text_embeddings.py \
  --prompt-json dataset/metadata/ovdcoco_prompts_list8.json \
  --clip-model pretrained_models/clip_convnext_large_head.pth \
  --output dataset/metadata/ovdcoco_vlm_query_convnextl.npy \
  --aggregate mean \
  --normalize
```

---

## ✅ 验证所有文件

运行这个脚本检查所有必需文件是否就绪：

```bash
python3 << 'EOF'
import os
from pathlib import Path

base = Path("/Users/zhengjiankang/Downloads/research/research/ovd-poc")

files_to_check = {
    "配置文件": base / "lami_dino/configs/dino_convnext_large_4scale_12ep_lvis.py",
    "类别定义 - seen": base / "dataset/metadata/ovcoco_seen_classes.json",
    "类别定义 - all": base / "dataset/metadata/ovcoco_all_classes.json",
    "文本嵌入 - prompts": base / "dataset/metadata/ovdcoco_prompts_list8_v2.npy",
    "文本嵌入 - vlm": base / "dataset/metadata/ovdcoco_vlm_query_convnextl.npy",
    "CLIP 模型": base / "pretrained_models/clip_convnext_large_head.pth",
    "训练标注": base / "dataset/coco/annotations/ovd_ins_train2017_b.json",
    "验证标注": base / "dataset/coco/annotations/ovd_ins_val2017_all.json",
}

print("=" * 70)
print("文件组织检查")
print("=" * 70)

missing = []
for name, path in files_to_check.items():
    if path.exists():
        size = path.stat().st_size / 1024 / 1024  # MB
        print(f"✅ {name:<25} ({size:.1f} MB)")
    else:
        print(f"❌ {name:<25} [缺失]")
        missing.append(name)

# Check output directory
output_dir = base / "output"
if output_dir.exists():
    ckpts = list(output_dir.glob("*.pth"))
    if ckpts:
        print(f"✅ {'模型权重':<25} ({len(ckpts)} 个文件)")
    else:
        print(f"⚠️  {'模型权重':<25} [output/目录存在但无.pth文件]")
        missing.append("模型权重")
else:
    print(f"❌ {'模型权重':<25} [output/目录不存在]")
    missing.append("模型权重")

print("=" * 70)

if missing:
    print(f"\n⚠️  还需要准备 {len(missing)} 项:")
    for item in missing:
        print(f"   - {item}")
else:
    print("\n✅ 所有必需文件已就绪！")

print("=" * 70)
EOF
```

---

## 🚀 快速复制命令（一键执行）

如果所有源文件都在 `dataset2/` 和 `pretrained_models2/` 中：

```bash
#!/bin/bash
cd /Users/zhengjiankang/Downloads/research/research/ovd-poc

echo "开始复制文件..."

# 1. 类别定义
cp dataset2/metadata/ovcoco_seen_classes.json dataset/metadata/
cp dataset2/metadata/ovcoco_all_classes.json dataset/metadata/

# 2. 文本嵌入（使用现有文件作为临时方案）
cp dataset2/metadata/vodcoco_tpa_prompts_convnextl.npy \
   dataset/metadata/ovdcoco_prompts_list8_v2.npy
cp dataset2/metadata/vodcoco_tpa_prompts_convnextl.npy \
   dataset/metadata/ovdcoco_vlm_query_convnextl.npy

# 3. CLIP 模型
cp pretrained_models2/clip_convnext_large_head.pth \
   pretrained_models/

# 4. COCO 标注（如果存在）
if [ -f dataset2/coco/annotations/ovd_ins_train2017_b.json ]; then
    cp dataset2/coco/annotations/ovd_ins_train2017_b.json \
       dataset/coco/annotations/
fi
if [ -f dataset2/coco/annotations/ovd_ins_val2017_all.json ]; then
    cp dataset2/coco/annotations/ovd_ins_val2017_all.json \
       dataset/coco/annotations/
fi

echo "✅ 文件复制完成！"
echo "运行验证脚本检查："
echo "python3 [上面的验证脚本]"
```

保存为 `copy_files.sh` 并运行：
```bash
chmod +x copy_files.sh
./copy_files.sh
```

---

## 📋 配置文件路径参考

确保配置文件 `lami_dino/configs/dino_convnext_large_4scale_12ep_lvis.py` 中的路径都指向 `dataset/` 而不是 `dataset2/`：

```python
# 应该是这样 ✅
model.vlm_query_path = "dataset/metadata/ovdcoco_vlm_query_convnextl.npy"
model.seen_classes = 'dataset/metadata/ovcoco_seen_classes.json'
model.all_classes = 'dataset/metadata/ovcoco_all_classes.json'
model.query_path = "dataset/metadata/ovdcoco_prompts_list8_v2.npy"
model.eval_query_path = "dataset/metadata/ovdcoco_prompts_list8_v2.npy"
model.classifier.text_embed_path = "dataset/metadata/ovdcoco_prompts_list8_v2.npy"
model.classifier.eval_text_embed_path = "dataset/metadata/ovdcoco_prompts_list8_v2.npy"
```

这些路径已经在配置中正确设置。

---

## 🎯 下一步

1. ✅ 软链接已删除
2. ✅ 目录结构已创建
3. ⏳ 按照上面的指南复制文件
4. ⏳ 验证所有文件
5. ⏳ 开始训练或运行可视化

---

**更新时间**: 2026-02-03
**状态**: 目录结构已就绪，等待文件复制
