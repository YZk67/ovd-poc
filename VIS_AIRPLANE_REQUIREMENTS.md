# 运行 vis_airplane.py 所需文件清单

## 📋 当前状态

### ✅ 已有的文件
- ✅ 配置文件: `lami_dino/configs/dino_convnext_large_4scale_12ep_lvis.py`
- ✅ 类别定义: `dataset/metadata/ovcoco_seen_classes.json`
- ✅ 类别定义: `dataset/metadata/ovcoco_all_classes.json`
- ✅ CLIP 模型: `pretrained_models2/clip_convnext_large_head.pth`
- ✅ 一个相关的嵌入文件: `dataset2/metadata/vodcoco_tpa_prompts_convnextl.npy` (可能是拼写错误)

### ❌ 缺失的文件

#### 1. **训练好的模型权重** (最关键!)
```
output/model_final.pth
或
output/model_best.pth
或
output/checkpoint_XXXX.pth
```

**状态**: ❌ **必需 - 你需要先训练模型**

**如何获取**:
- **选项A**: 训练模型
  ```bash
  python tools/train_net.py \
      --config-file lami_dino/configs/dino_convnext_large_4scale_12ep_lvis.py \
      --num-gpus 4
  ```
  训练完成后会在 `output/` 目录生成 checkpoint 文件

- **选项B**: 使用预训练模型（如果有的话）
  - 从其他地方复制已训练好的模型到 `output/` 目录

---

#### 2. **文本嵌入文件**
```
dataset/metadata/ovdcoco_prompts_list8_v2.npy
dataset/metadata/ovdcoco_vlm_query_convnextl.npy
```

**状态**: ❌ **必需 - 模型加载时需要**

**问题**: 
- 配置文件引用这两个文件，但不存在
- 有一个类似的文件: `vodcoco_tpa_prompts_convnextl.npy` (注意是 vodcoco 不是 ovdcoco)

**解决方案**:

##### 方案1: 生成新的嵌入文件 (推荐)

**步骤1**: 创建 COCO 类别的 prompts 文件

创建文件 `dataset2/metadata/ovdcoco_prompts_list8.json`:
```json
{
  "person": ["a photo of a person", "a person in the scene", ...],
  "bicycle": ["a photo of a bicycle", "a bicycle in the scene", ...],
  ...
}
```

或者使用脚本生成（如果有 COCO 类别信息文件）:
```bash
python tools/generate_class_prompts.py \
  --classes dataset/metadata/ovcoco_all_classes.json \
  --prompt-output dataset2/metadata/ovdcoco_prompts_list8.json \
  --num-prompts 8
```

**步骤2**: 生成文本嵌入

```bash
# 生成 ovdcoco_prompts_list8_v2.npy
python tools/generate_text_embeddings.py \
  --prompt-json dataset2/metadata/ovdcoco_prompts_list8.json \
  --clip-model pretrained_models2/clip_convnext_large_head.pth \
  --output dataset2/metadata/ovdcoco_prompts_list8_v2.npy \
  --aggregate none \
  --normalize

# 生成 ovdcoco_vlm_query_convnextl.npy (聚合版本)
python tools/generate_text_embeddings.py \
  --prompt-json dataset2/metadata/ovdcoco_prompts_list8.json \
  --clip-model pretrained_models2/clip_convnext_large_head.pth \
  --output dataset2/metadata/ovdcoco_vlm_query_convnextl.npy \
  --aggregate mean \
  --normalize
```

##### 方案2: 使用现有文件（临时方案）

如果 `vodcoco_tpa_prompts_convnextl.npy` 实际上就是你需要的文件（可能只是命名错误），可以复制并重命名：

```bash
cd /Users/zhengjiankang/Downloads/research/research/ovd-poc

# 复制并重命名
cp dataset2/metadata/vodcoco_tpa_prompts_convnextl.npy \
   dataset/metadata/ovdcoco_prompts_list8_v2.npy
   
cp dataset2/metadata/vodcoco_tpa_prompts_convnextl.npy \
   dataset/metadata/ovdcoco_vlm_query_convnextl.npy
```

**注意**: 这只有在文件格式和形状匹配时才有效。

##### 方案3: 使用一键复制脚本（最简单）

```bash
cd /Users/zhengjiankang/Downloads/research/research/ovd-poc
./copy_files.sh
```

这会自动复制所有文件到正确位置。

---

#### 3. **测试图像**
```
任意 COCO 图像或你自己的图像
```

**状态**: ⚠️ **需要准备**

**如何准备**:
- 使用 COCO 验证集图像: `dataset2/coco/val2017/`
- 或使用你自己的测试图像

---

## 🚀 完整运行命令

### 前提条件检查

```bash
cd /Users/zhengjiankang/Downloads/research/research/ovd-poc

# 1. 检查配置文件
ls -lh lami_dino/configs/dino_convnext_large_4scale_12ep_lvis.py

# 2. 检查模型权重
ls -lh output/*.pth

# 3. 检查文本嵌入
ls -lh dataset/metadata/ovdcoco*.npy

# 4. 检查测试图像
ls dataset2/coco/val2017/ | head -5
```

### 最简运行示例

```bash
python tools/vis_airplane.py \
    --config lami_dino/configs/dino_convnext_large_4scale_12ep_lvis.py \
    --weights output/model_final.pth \
    --input dataset2/coco/val2017/ \
    --output visualization_output/ \
    --limit 10 \
    --score-thresh 0.3
```

### 可视化特定类别

```bash
# 只可视化 "airplane" 类别
python tools/vis_airplane.py \
    --config lami_dino/configs/dino_convnext_large_4scale_12ep_lvis.py \
    --weights output/model_final.pth \
    --input dataset2/coco/val2017/ \
    --output visualization_output/airplane/ \
    --class-name airplane \
    --score-thresh 0.5 \
    --limit 50

# 可视化 novel 类别（如 "cat", "dog", "elephant" 等）
python tools/vis_airplane.py \
    --config lami_dino/configs/dino_convnext_large_4scale_12ep_lvis.py \
    --weights output/model_final.pth \
    --input dataset2/coco/val2017/ \
    --output visualization_output/novel/ \
    --class-name cat \
    --score-thresh 0.3
```

### Debug 模式

```bash
python tools/vis_airplane.py \
    --config lami_dino/configs/dino_convnext_large_4scale_12ep_lvis.py \
    --weights output/model_final.pth \
    --input /path/to/single_image.jpg \
    --output visualization_output/ \
    --debug \
    --device cuda
```

---

## 📝 任务优先级

### 🔴 高优先级（必需）

1. **训练模型获取权重文件**
   - 这是最关键的步骤
   - 没有训练好的模型，无法进行可视化

2. **生成或准备文本嵌入文件**
   - `ovdcoco_prompts_list8_v2.npy`
   - `ovdcoco_vlm_query_convnextl.npy`

### 🟡 中优先级（推荐）

3. **准备测试图像**
   - 下载 COCO val2017 数据集
   - 或准备自己的测试图像

### 🟢 低优先级（可选）

4. **安装依赖**
   ```bash
   pip install open_clip_torch  # 或 pip install git+https://github.com/openai/CLIP.git
   pip install tqdm
   ```

---

## 🔍 快速检查脚本

运行这个脚本检查所有依赖：

```bash
python3 << 'EOF'
import os
from pathlib import Path

base = Path("/Users/zhengjiankang/Downloads/research/research/ovd-poc")

print("=" * 70)
print("VIS_AIRPLANE.PY 依赖检查")
print("=" * 70)

checks = {
    "配置文件": base / "lami_dino/configs/dino_convnext_large_4scale_12ep_lvis.py",
    "模型权重": list((base / "output").glob("*.pth")) if (base / "output").exists() else [],
    "文本嵌入1": base / "dataset/metadata/ovdcoco_prompts_list8_v2.npy",
    "文本嵌入2": base / "dataset/metadata/ovdcoco_vlm_query_convnextl.npy",
    "CLIP模型": base / "pretrained_models2/clip_convnext_large_head.pth",
    "类别定义1": base / "dataset/metadata/ovcoco_seen_classes.json",
    "类别定义2": base / "dataset/metadata/ovcoco_all_classes.json",
}

for name, path in checks.items():
    if isinstance(path, list):
        status = f"✅ 找到 {len(path)} 个文件" if path else "❌ 不存在"
        print(f"{name:<15} {status}")
        if path:
            for p in path[:3]:
                print(f"                  - {p.name}")
    else:
        status = "✅ 存在" if path.exists() else "❌ 不存在"
        print(f"{name:<15} {status}")

print("\n" + "=" * 70)
print("优先任务:")
if not list((base / "output").glob("*.pth")) if (base / "output").exists() else True:
    print("  1. ❌ 训练模型或准备模型权重文件")
if not (base / "dataset/metadata/ovdcoco_prompts_list8_v2.npy").exists():
    print("  2. ❌ 生成文本嵌入文件")
print("=" * 70)
EOF
```

---

## 📚 相关文档

- `OVD_VERIFICATION_REPORT.md` - OVD 配置验证报告
- `tools/README_generate_embeddings.md` - 生成文本嵌入指南
- `TRAINING_PROCESS.md` - 训练流程文档（如果存在）

---

**更新时间**: 2026-02-03
**状态**: 等待模型训练完成
