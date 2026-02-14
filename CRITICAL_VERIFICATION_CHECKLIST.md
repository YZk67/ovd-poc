# OVD-COCO 训练与推理的关键验证清单

## ⚠️ 诚实声明

我**不能100%确定代码没有bug**，因为有些关键点我还**没有实际验证**。以下是需要检查的所有潜在风险点。

---

## 🔴 高风险：必须立即验证

### 1. 文本嵌入文件的类别顺序 ⚠️ 

**文件**: `dataset/metadata/ovdcoco_prompts_list8_v2.npy`

**风险**: 如果这个文件的类别顺序与`OVDCOCO65`不一致，会导致**所有类别预测错乱**。

**当前状态**: ❓ **未验证**

**验证方法**:
```python
import numpy as np
from detectron2.data.datasets.builtin import OVDCOCO65

# 1. 检查shape
text_embed = np.load("dataset/metadata/ovdcoco_prompts_list8_v2.npy")
print(f"Shape: {text_embed.shape}")  # 必须是 (65, K, D)

# 2. 检查生成时的类别顺序
# 需要查看生成这个.npy文件时使用的JSON
import json
with open("dataset/metadata/ovdcoco_prompts_list8_rich_v2.json") as f:
    prompts = json.load(f)

# 3. 验证顺序
if isinstance(prompts, dict):
    # 如果是dict，顺序取决于JSON的key顺序
    prompt_classes = list(prompts.keys())
elif isinstance(prompts, list):
    # 如果是list，顺序取决于list的顺序
    prompt_classes = [p["class"] for p in prompts]

print(f"OVDCOCO65[:5] = {OVDCOCO65[:5]}")
print(f"Prompt classes[:5] = {prompt_classes[:5]}")

# 必须完全一致！
for i in range(65):
    if OVDCOCO65[i] != prompt_classes[i]:
        print(f"❌ MISMATCH at index {i}: {OVDCOCO65[i]} != {prompt_classes[i]}")
        break
else:
    print("✅ All 65 classes match in order!")
```

**如果不匹配**: 需要重新生成文本嵌入文件，按照`OVDCOCO65`的顺序。

---

### 2. seen_classes.json 和 all_classes.json 的顺序 ⚠️

**文件**: 
- `dataset/metadata/ovcoco_seen_classes.json`
- `dataset/metadata/ovcoco_all_classes.json`

**当前内容**:
```json
// ovcoco_seen_classes.json (48类)
["person", "bicycle", "car", "motorcycle", "train", "truck", "boat", "bench", 
 "bird", "horse", "sheep", "bear", "zebra", "giraffe", "backpack", "handbag", 
 "suitcase", "frisbee", "skis", "kite", "surfboard", "bottle", "fork", "spoon", 
 "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "pizza", 
 "donut", "chair", "bed", "toilet", "tv", "laptop", "mouse", "remote", 
 "microwave", "oven", "toaster", "refrigerator", "book", "clock", "vase", "toothbrush"]

// ovcoco_all_classes.json (65类)
["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", 
 "boat", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", 
 "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", 
 "frisbee", "skis", "snowboard", "kite", "skateboard", "surfboard", "bottle", 
 "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", 
 "broccoli", "carrot", "pizza", "donut", "cake", "chair", "couch", "bed", 
 "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "microwave", "oven", 
 "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "toothbrush"]
```

**风险**: 这两个文件的顺序**必须与`OVDCOCO65`完全一致**，否则seen/novel分类会错误。

**当前状态**: ⚠️ **发现潜在问题！**

**问题**: `ovcoco_all_classes.json`的顺序与`OVDCOCO65`**可能不一致**！

**验证**:
```python
import json
from detectron2.data.datasets.builtin import OVDCOCO65

with open("dataset/metadata/ovcoco_all_classes.json") as f:
    all_classes = json.load(f)

print(f"OVDCOCO65 length: {len(OVDCOCO65)}")
print(f"all_classes length: {len(all_classes)}")

for i in range(min(len(OVDCOCO65), len(all_classes))):
    if OVDCOCO65[i] != all_classes[i]:
        print(f"❌ MISMATCH at {i}: OVDCOCO65[{i}]='{OVDCOCO65[i]}' vs all_classes[{i}]='{all_classes[i]}'")
```

**预期问题**: `OVDCOCO65`定义在`builtin.py`第66-74行，需要确认顺序是否一致。

---

### 3. 数据集JSON文件是否存在 🔴

**需要的文件**:
- `dataset/coco/annotations/ovd_ins_train2017_b.json`
- `dataset/coco/annotations/ovd_ins_val2017_all.json`

**当前状态**: ❓ **未验证是否存在**

**验证**:
```bash
ls -lh dataset/coco/annotations/ovd_ins_train2017_b.json
ls -lh dataset/coco/annotations/ovd_ins_val2017_all.json
```

**如果不存在**: 训练和评估都会失败。需要准备这些标注文件。

---

### 4. JSON文件中的category_id是否正确 🔴

**风险**: JSON文件必须使用**COCO原始ID (1-90)**，而不是contiguous ID (0-64)。

**当前状态**: ❓ **未验证**

**验证方法**:
```python
import json

with open("dataset/coco/annotations/ovd_ins_train2017_b.json") as f:
    data = json.load(f)

# 检查annotations中的category_id
cat_ids = {ann["category_id"] for ann in data["annotations"]}
print(f"Category IDs in JSON: {sorted(cat_ids)}")

# 应该是COCO原始ID的子集
expected_ids = {1, 2, 3, 4, 5, 6, 7, 8, 9, 15, 16, 17, ...}  # COCO ID
print(f"Expected IDs (COCO): {sorted(expected_ids)}")

# 如果JSON中是0-64，说明是错误的contiguous ID
if 0 in cat_ids:
    print("❌ ERROR: JSON使用了contiguous ID (0-64)，应该使用COCO ID (1-90)")
```

---

### 5. 文本嵌入文件是否存在且shape正确 🔴

**需要的文件**:
- `dataset/metadata/ovdcoco_prompts_list8_v2.npy` - TPA用
- `dataset/metadata/ovdcoco_vlm_query_convnextl.npy` - VLM ensemble用（如果启用）

**当前状态**: ❓ **未验证文件是否存在于正确位置**

**注意**: 配置文件指向`dataset/metadata/`，但实际文件在`dataset2/metadata/`！

**验证**:
```bash
# 检查文件是否存在
ls -lh dataset/metadata/ovdcoco_prompts_list8_v2.npy
ls -lh dataset/metadata/ovdcoco_vlm_query_convnextl.npy

# 如果不存在，需要从dataset2复制或生成
python << 'EOF'
import numpy as np

# TPA文件
try:
    tpa = np.load("dataset/metadata/ovdcoco_prompts_list8_v2.npy")
    print(f"✅ TPA shape: {tpa.shape}")  # 应该是 (65, K, D)
    assert tpa.shape[0] == 65, f"Expected 65 classes, got {tpa.shape[0]}"
except FileNotFoundError:
    print("❌ TPA file not found!")

# VLM文件
try:
    vlm = np.load("dataset/metadata/ovdcoco_vlm_query_convnextl.npy")
    print(f"✅ VLM shape: {vlm.shape}")  # 应该是 (65, D) 或 (65, K, D)
    assert vlm.shape[0] == 65, f"Expected 65 classes, got {vlm.shape[0]}"
except FileNotFoundError:
    print("❌ VLM file not found!")
EOF
```

---

## 🟡 中等风险：需要验证

### 6. model.num_classes 配置 🟡

**配置文件**: `lami_dino/configs/models/dino_convnextl.py`

**期望值**: `model.num_classes = 65`

**风险**: 如果设置为80或其他值，会导致维度不匹配。

**验证**:
```python
from lami_dino.configs.models.dino_convnextl import model

print(f"model.num_classes = {model.num_classes}")
assert model.num_classes == 65, f"Expected 65, got {model.num_classes}"
```

---

### 7. TPA配置是否正确 🟡

**配置文件**: `lami_dino/configs/dino_convnext_large_4scale_12ep_lvis.py`

**关键配置**:
```python
model.classifier.use_tpa = True
model.classifier.text_embed_path = "dataset/metadata/ovdcoco_prompts_list8_v2.npy"
model.classifier.tpa_num_prototypes = 5
```

**验证**:
```python
from lami_dino.configs.dino_convnext_large_4scale_12ep_lvis import model

assert model.classifier.use_tpa == True
assert model.classifier.tpa_num_prototypes == 5
print(f"✅ TPA enabled with {model.classifier.tpa_num_prototypes} prototypes")
```

---

### 8. Score Ensemble配置（如果启用）🟡

**当前配置**:
```python
model.score_ensemble = True
model.vlm_query_path = "dataset/metadata/ovdcoco_vlm_query_convnextl.npy"
```

**风险**: 如果`score_ensemble=True`但文件不存在，推理会失败。

**建议**: 如果初期调试，可以先设置`model.score_ensemble = False`简化配置。

---

### 9. OVDCOCO65定义与实际类别数一致 🟡

**代码位置**: `detectron2/detectron2/data/datasets/builtin.py` 第66-74行

**验证**:
```python
from detectron2.data.datasets.builtin import OVDCOCO65, OVDCOCO65_IDMAP

print(f"OVDCOCO65 length: {len(OVDCOCO65)}")  # 必须是65
print(f"OVDCOCO65_IDMAP length: {len(OVDCOCO65_IDMAP)}")  # 必须是65

assert len(OVDCOCO65) == 65
assert len(OVDCOCO65_IDMAP) == 65
assert OVDCOCO65_IDMAP[1] == 0  # person
assert OVDCOCO65_IDMAP[90] == 64  # toothbrush
```

---

### 10. 数据集注册是否正确 🟡

**验证**:
```python
from detectron2.data import MetadataCatalog

# 训练集
train_meta = MetadataCatalog.get("ovdcoco65_2017_train_b")
assert train_meta.thing_classes == OVDCOCO65
assert len(train_meta.thing_dataset_id_to_contiguous_id) == 65
print("✅ Train dataset registered correctly")

# 测试集
test_meta = MetadataCatalog.get("ovdcoco65_2017_val_all")
assert test_meta.thing_classes == OVDCOCO65
assert len(test_meta.thing_dataset_id_to_contiguous_id) == 65
print("✅ Test dataset registered correctly")
```

---

## 🟢 低风险：建议验证

### 11. seen/novel类别分类是否正确 🟢

**代码位置**: `lami_dino/modeling/dino.py` `__init__()`

这个在初始化时会从`seen_classes`和`all_classes`构建seen/novel的索引。

**验证**:
```python
# 需要在模型初始化后检查
# model.seen_idx 和 model.novel_idx
```

---

### 12. 预训练权重是否存在 🟢

**需要的文件**: 预训练的backbone和其他组件权重

**验证**: 根据配置检查相应的权重文件是否存在。

---

## 🔧 完整验证脚本

```python
#!/usr/bin/env python3
"""
完整的OVD-COCO设置验证脚本
运行这个脚本来检查所有潜在问题
"""

import os
import sys
import json
import numpy as np
from pathlib import Path

print("=" * 80)
print("🔍 OVD-COCO 完整验证")
print("=" * 80)
print()

errors = []
warnings = []

# ============================================================================
# 1. 检查OVDCOCO65定义
# ============================================================================
print("1️⃣ 检查OVDCOCO65定义...")
try:
    from detectron2.data.datasets.builtin import OVDCOCO65, OVDCOCO65_IDMAP
    
    assert len(OVDCOCO65) == 65, f"OVDCOCO65应该有65个类，实际{len(OVDCOCO65)}"
    assert len(OVDCOCO65_IDMAP) == 65, f"OVDCOCO65_IDMAP应该有65个映射，实际{len(OVDCOCO65_IDMAP)}"
    assert OVDCOCO65_IDMAP[1] == 0, "person (COCO ID 1)应该映射到0"
    assert OVDCOCO65_IDMAP[90] == 64, "toothbrush (COCO ID 90)应该映射到64"
    
    print("  ✅ OVDCOCO65定义正确")
except AssertionError as e:
    errors.append(f"OVDCOCO65定义错误: {e}")
    print(f"  ❌ {e}")
except Exception as e:
    errors.append(f"无法加载OVDCOCO65: {e}")
    print(f"  ❌ {e}")

print()

# ============================================================================
# 2. 检查seen_classes和all_classes的顺序
# ============================================================================
print("2️⃣ 检查seen_classes和all_classes...")
try:
    # 检查文件是否存在
    seen_path = "dataset/metadata/ovcoco_seen_classes.json"
    all_path = "dataset/metadata/ovcoco_all_classes.json"
    
    if not os.path.exists(seen_path):
        seen_path = "dataset2/metadata/ovcoco_seen_classes.json"
    if not os.path.exists(all_path):
        all_path = "dataset2/metadata/ovcoco_all_classes.json"
    
    if not os.path.exists(seen_path):
        errors.append(f"ovcoco_seen_classes.json不存在")
        print(f"  ❌ seen_classes.json不存在")
    else:
        with open(seen_path) as f:
            seen_classes = json.load(f)
        print(f"  ✅ seen_classes: {len(seen_classes)}个类")
    
    if not os.path.exists(all_path):
        errors.append(f"ovcoco_all_classes.json不存在")
        print(f"  ❌ all_classes.json不存在")
    else:
        with open(all_path) as f:
            all_classes = json.load(f)
        
        assert len(all_classes) == 65, f"all_classes应该有65个类，实际{len(all_classes)}"
        
        # 检查顺序是否与OVDCOCO65一致
        mismatches = []
        for i in range(65):
            if OVDCOCO65[i] != all_classes[i]:
                mismatches.append((i, OVDCOCO65[i], all_classes[i]))
        
        if mismatches:
            errors.append(f"all_classes顺序与OVDCOCO65不匹配: {len(mismatches)}处不同")
            print(f"  ❌ all_classes顺序不匹配OVDCOCO65:")
            for i, expected, actual in mismatches[:5]:
                print(f"     位置{i}: 期望'{expected}', 实际'{actual}'")
            if len(mismatches) > 5:
                print(f"     ... 还有{len(mismatches)-5}处不同")
        else:
            print(f"  ✅ all_classes顺序与OVDCOCO65一致")

except Exception as e:
    errors.append(f"检查seen/all classes失败: {e}")
    print(f"  ❌ {e}")

print()

# ============================================================================
# 3. 检查文本嵌入文件
# ============================================================================
print("3️⃣ 检查文本嵌入文件...")

# TPA文本嵌入
tpa_paths = [
    "dataset/metadata/ovdcoco_prompts_list8_v2.npy",
    "dataset2/metadata/vodcoco_tpa_prompts_convnextl.npy",
]

tpa_found = False
for path in tpa_paths:
    if os.path.exists(path):
        try:
            tpa = np.load(path)
            assert tpa.shape[0] == 65, f"TPA第一维应该是65，实际{tpa.shape[0]}"
            print(f"  ✅ TPA文件: {path}, shape={tpa.shape}")
            tpa_found = True
            break
        except Exception as e:
            warnings.append(f"TPA文件加载失败 {path}: {e}")
            print(f"  ⚠️  TPA文件存在但加载失败 {path}: {e}")

if not tpa_found:
    errors.append("未找到TPA文本嵌入文件")
    print(f"  ❌ 未找到TPA文本嵌入文件")
    print(f"     尝试过的路径: {tpa_paths}")

# VLM文本嵌入 (可选)
vlm_path = "dataset/metadata/ovdcoco_vlm_query_convnextl.npy"
if not os.path.exists(vlm_path):
    vlm_path = "dataset2/metadata/ovdcoco_vlm_query_convnextl.npy"

if os.path.exists(vlm_path):
    try:
        vlm = np.load(vlm_path)
        assert vlm.shape[0] == 65, f"VLM第一维应该是65，实际{vlm.shape[0]}"
        print(f"  ✅ VLM文件: {vlm_path}, shape={vlm.shape}")
    except Exception as e:
        warnings.append(f"VLM文件加载失败: {e}")
        print(f"  ⚠️  VLM文件存在但加载失败: {e}")
else:
    warnings.append("未找到VLM文本嵌入文件（如果score_ensemble=True需要）")
    print(f"  ⚠️  未找到VLM文本嵌入文件（如果score_ensemble=True需要）")

print()

# ============================================================================
# 4. 检查数据集JSON文件
# ============================================================================
print("4️⃣ 检查数据集JSON文件...")

json_files = {
    "train": "dataset/coco/annotations/ovd_ins_train2017_b.json",
    "val": "dataset/coco/annotations/ovd_ins_val2017_all.json",
}

for name, path in json_files.items():
    if os.path.exists(path):
        try:
            with open(path) as f:
                data = json.load(f)
            
            # 检查category_id
            if "annotations" in data:
                cat_ids = {ann["category_id"] for ann in data["annotations"][:100]}
                
                # 检查是否使用COCO ID (应该有1)
                if 1 in cat_ids:
                    print(f"  ✅ {name} JSON存在，使用COCO ID")
                elif 0 in cat_ids:
                    errors.append(f"{name} JSON使用了contiguous ID (0)，应该使用COCO ID (1)")
                    print(f"  ❌ {name} JSON使用了错误的ID类型（contiguous而非COCO）")
                else:
                    warnings.append(f"{name} JSON的category_id无法判断类型")
                    print(f"  ⚠️  {name} JSON的category_id类型无法判断")
        except Exception as e:
            errors.append(f"{name} JSON加载失败: {e}")
            print(f"  ❌ {name} JSON加载失败: {e}")
    else:
        errors.append(f"{name} JSON不存在: {path}")
        print(f"  ❌ {name} JSON不存在: {path}")

print()

# ============================================================================
# 5. 检查数据集注册
# ============================================================================
print("5️⃣ 检查数据集注册...")
try:
    from detectron2.data import MetadataCatalog
    
    for ds_name in ["ovdcoco65_2017_train_b", "ovdcoco65_2017_val_all"]:
        meta = MetadataCatalog.get(ds_name)
        
        if not hasattr(meta, "thing_classes"):
            errors.append(f"{ds_name}未设置thing_classes")
            print(f"  ❌ {ds_name}未设置thing_classes")
        elif meta.thing_classes != OVDCOCO65:
            errors.append(f"{ds_name}的thing_classes与OVDCOCO65不匹配")
            print(f"  ❌ {ds_name}的thing_classes不匹配")
        else:
            print(f"  ✅ {ds_name}注册正确")
        
        if not hasattr(meta, "thing_dataset_id_to_contiguous_id"):
            errors.append(f"{ds_name}未设置ID映射")
            print(f"  ❌ {ds_name}未设置ID映射")
        elif len(meta.thing_dataset_id_to_contiguous_id) != 65:
            errors.append(f"{ds_name}的ID映射数量不是65")
            print(f"  ❌ {ds_name}的ID映射数量错误")

except Exception as e:
    errors.append(f"检查数据集注册失败: {e}")
    print(f"  ❌ {e}")

print()

# ============================================================================
# 总结
# ============================================================================
print("=" * 80)
print("📊 验证总结")
print("=" * 80)
print()

if not errors and not warnings:
    print("🎉 所有检查通过！代码应该没有明显bug。")
elif not errors:
    print(f"✅ 无严重错误，但有 {len(warnings)} 个警告:")
    for w in warnings:
        print(f"  ⚠️  {w}")
else:
    print(f"❌ 发现 {len(errors)} 个错误:")
    for e in errors:
        print(f"  ❌ {e}")
    
    if warnings:
        print()
        print(f"⚠️  还有 {len(warnings)} 个警告:")
        for w in warnings:
            print(f"  ⚠️  {w}")
    
    print()
    print("❌ 需要修复这些错误才能正常训练和评估！")

print()
print("=" * 80)

sys.exit(0 if not errors else 1)
```

---

## 🎯 我的诚实回答

### 我**不能100%确定**代码没有bug，因为：

1. ❌ **文本嵌入文件的类别顺序** - 这是最高风险点，我无法验证
2. ❌ **seen/all_classes.json的顺序** - 需要与OVDCOCO65完全一致
3. ❌ **数据集JSON文件** - 不知道是否存在，category_id是否正确
4. ❌ **文件路径问题** - 配置指向`dataset/`但文件在`dataset2/`

### 理论上代码逻辑是正确的：

✅ ID映射机制正确
✅ 训练loss计算正确
✅ 评估反向映射正确

### 但实际运行可能出现问题：

⚠️ 文件不存在
⚠️ 文件内容顺序错误
⚠️ 配置与实际不匹配

---

## 🔧 建议立即执行

1. **运行上面的完整验证脚本**
2. **检查所有文件是否存在于正确位置**
3. **验证文本嵌入的类别顺序**
4. **确认seen/all_classes.json与OVDCOCO65顺序一致**

只有完成这些验证，我才能确定代码可以正常工作！
