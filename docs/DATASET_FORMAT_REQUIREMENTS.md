# 数据集文件格式要求

## 概述

本文档详细说明训练和验证数据集需要哪些文件，以及这些文件的格式要求。

## 📁 数据集目录结构

### COCO格式（推荐）

```
dataset/
  ovd_coco/
    images/
      train/
        img_001.jpg
        img_002.jpg
        img_003.jpg
        ...
      val/
        img_001.jpg
        img_002.jpg
        ...
    annotations/
      train.json      ← 训练集标注文件（必需）
      val.json        ← 验证集标注文件（必需）
```

### LVIS格式

```
dataset/
  ovd_coco/
    images/           ← 所有图片放在一个目录
      img_001.jpg
      img_002.jpg
      ...
    annotations/
      train.json      ← 训练集标注文件
      val.json        ← 验证集标注文件
```

## 📄 JSON标注文件格式

### COCO格式（标准格式）

#### 完整结构示例

```json
{
  "info": {
    "description": "OVD-COCO Dataset",
    "version": "1.0",
    "year": 2024,
    "contributor": "Your Name",
    "date_created": "2024-01-01"
  },
  "licenses": [
    {
      "id": 1,
      "name": "Unknown",
      "url": ""
    }
  ],
  "categories": [
    {
      "id": 1,
      "name": "person",
      "supercategory": "none"
    },
    {
      "id": 2,
      "name": "bicycle",
      "supercategory": "vehicle"
    },
    {
      "id": 3,
      "name": "car",
      "supercategory": "vehicle"
    }
  ],
  "images": [
    {
      "id": 1,
      "width": 640,
      "height": 480,
      "file_name": "img_001.jpg",
      "license": 1,
      "flickr_url": "",
      "coco_url": "",
      "date_captured": ""
    },
    {
      "id": 2,
      "width": 800,
      "height": 600,
      "file_name": "img_002.jpg",
      "license": 1,
      "flickr_url": "",
      "coco_url": "",
      "date_captured": ""
    }
  ],
  "annotations": [
    {
      "id": 1,
      "image_id": 1,
      "category_id": 1,
      "bbox": [100, 150, 200, 300],
      "area": 60000.0,
      "iscrowd": 0,
      "segmentation": []
    },
    {
      "id": 2,
      "image_id": 1,
      "category_id": 2,
      "bbox": [300, 200, 150, 180],
      "area": 27000.0,
      "iscrowd": 0,
      "segmentation": []
    }
  ]
}
```

#### 字段详细说明

##### 1. `info`（可选但推荐）

```json
{
  "description": "数据集描述",
  "version": "版本号",
  "year": 年份,
  "contributor": "贡献者",
  "date_created": "创建日期"
}
```

##### 2. `licenses`（可选）

```json
[
  {
    "id": 1,
    "name": "许可证名称",
    "url": "许可证URL"
  }
]
```

##### 3. `categories`（必需）

```json
[
  {
    "id": 1,                    // 类别ID，必须是整数，通常从1开始
    "name": "person",           // 类别名称（字符串）
    "supercategory": "none"     // 父类别（可选，可以是"none"）
  }
]
```

**重要要求：**
- `id` 必须是唯一的整数
- `id` 通常从1开始（也可以是其他正整数）
- `name` 必须是唯一的字符串
- 所有类别必须在此处定义

##### 4. `images`（必需）

```json
[
  {
    "id": 1,                    // 图片ID，必须是唯一的整数
    "width": 640,               // 图片宽度（像素）
    "height": 480,              // 图片高度（像素）
    "file_name": "img_001.jpg"  // 图片文件名（相对于image_root）
  }
]
```

**重要要求：**
- `id` 必须是唯一的整数
- `file_name` 必须与实际图片文件名匹配
- `width` 和 `height` 必须与实际图片尺寸匹配
- 如果图片在子目录中，`file_name` 应包含相对路径，如 `"train/img_001.jpg"`

##### 5. `annotations`（必需）

```json
[
  {
    "id": 1,                    // 标注ID，必须是唯一的整数
    "image_id": 1,              // 对应的图片ID（必须在images中存在）
    "category_id": 1,           // 类别ID（必须在categories中存在）
    "bbox": [100, 150, 200, 300],  // 边界框 [x, y, width, height]
    "area": 60000.0,            // 边界框面积（width * height）
    "iscrowd": 0,               // 是否为crowd（0或1）
    "segmentation": []          // 分割标注（可选，做检测时可以为空数组）
  }
]
```

**边界框格式说明：**
- `bbox`: `[x, y, width, height]`
  - `x`: 左上角x坐标
  - `y`: 左上角y坐标
  - `width`: 边界框宽度
  - `height`: 边界框高度
- `area`: 必须是 `width * height`
- `iscrowd`: 通常为0，如果为1表示这是一个crowd区域

## 📋 文件要求总结

### 训练集文件（train.json）

**必需字段：**
- ✅ `categories`: 所有类别的定义
- ✅ `images`: 训练集所有图片的信息
- ✅ `annotations`: 训练集所有标注

**可选字段：**
- ⚠️ `info`: 数据集信息（推荐）
- ⚠️ `licenses`: 许可证信息（可选）

### 验证集文件（val.json）

**必需字段：**
- ✅ `categories`: 所有类别的定义（必须与训练集一致）
- ✅ `images`: 验证集所有图片的信息
- ✅ `annotations`: 验证集所有标注

**重要：**
- 验证集的 `categories` 必须与训练集完全相同（相同的ID和name）
- 验证集的图片ID不应与训练集重复

## 🔍 验证数据集格式

### 方法1：使用Python脚本验证

```python
import json
from pycocotools.coco import COCO

# 验证训练集
train_json = "dataset/ovd_coco/annotations/train.json"
coco = COCO(train_json)

print(f"类别数量: {len(coco.cats)}")
print(f"图片数量: {len(coco.imgs)}")
print(f"标注数量: {len(coco.anns)}")

# 检查是否有错误
coco.getImgIds()  # 如果格式正确，不会报错
coco.getCatIds()  # 如果格式正确，不会报错
```

### 方法2：使用Detectron2验证

```python
from detectron2.data.datasets import register_coco_instances

# 尝试注册数据集
try:
    register_coco_instances(
        "ovd_coco_train",
        {},
        "dataset/ovd_coco/annotations/train.json",
        "dataset/ovd_coco/images/train"
    )
    print("✅ 数据集格式正确！")
except Exception as e:
    print(f"❌ 数据集格式错误: {e}")
```

## 📝 最小示例

### 最简单的train.json

```json
{
  "categories": [
    {"id": 1, "name": "person", "supercategory": "none"},
    {"id": 2, "name": "car", "supercategory": "none"}
  ],
  "images": [
    {"id": 1, "width": 640, "height": 480, "file_name": "img_001.jpg"}
  ],
  "annotations": [
    {
      "id": 1,
      "image_id": 1,
      "category_id": 1,
      "bbox": [100, 100, 200, 300],
      "area": 60000,
      "iscrowd": 0
    }
  ]
}
```

## ⚠️ 常见错误

### 错误1：类别ID不连续

```json
// ❌ 错误：类别ID从2开始
"categories": [
  {"id": 2, "name": "person"},
  {"id": 3, "name": "car"}
]

// ✅ 正确：类别ID从1开始
"categories": [
  {"id": 1, "name": "person"},
  {"id": 2, "name": "car"}
]
```

### 错误2：图片文件名不匹配

```json
// ❌ 错误：file_name与实际文件不匹配
"file_name": "img_001.jpg"  // 实际文件是 "image_001.jpg"

// ✅ 正确：file_name必须与实际文件名完全匹配
"file_name": "image_001.jpg"
```

### 错误3：bbox格式错误

```json
// ❌ 错误：使用[x_min, y_min, x_max, y_max]格式
"bbox": [100, 100, 300, 400]

// ✅ 正确：使用[x, y, width, height]格式
"bbox": [100, 100, 200, 300]  // [x, y, width, height]
```

### 错误4：image_id不存在

```json
// ❌ 错误：annotation中的image_id在images中不存在
"images": [{"id": 1, ...}],
"annotations": [{"image_id": 2, ...}]  // image_id=2不存在

// ✅ 正确：image_id必须在images中存在
"images": [{"id": 1, ...}, {"id": 2, ...}],
"annotations": [{"image_id": 2, ...}]  // image_id=2存在
```

## 🔧 工具和脚本

### 验证脚本

创建 `tools/validate_dataset.py`：

```python
import json
import sys
from pathlib import Path

def validate_coco_json(json_file):
    """验证COCO格式JSON文件"""
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    errors = []
    
    # 检查必需字段
    required_keys = ['categories', 'images', 'annotations']
    for key in required_keys:
        if key not in data:
            errors.append(f"缺少必需字段: {key}")
    
    if errors:
        return False, errors
    
    # 检查categories
    cat_ids = [cat['id'] for cat in data['categories']]
    if len(cat_ids) != len(set(cat_ids)):
        errors.append("类别ID重复")
    
    # 检查images
    img_ids = [img['id'] for img in data['images']]
    if len(img_ids) != len(set(img_ids)):
        errors.append("图片ID重复")
    
    # 检查annotations
    ann_ids = [ann['id'] for ann in data['annotations']]
    if len(ann_ids) != len(set(ann_ids)):
        errors.append("标注ID重复")
    
    # 检查annotation中的image_id是否存在
    for ann in data['annotations']:
        if ann['image_id'] not in img_ids:
            errors.append(f"标注 {ann['id']} 的 image_id {ann['image_id']} 不存在")
        if ann['category_id'] not in cat_ids:
            errors.append(f"标注 {ann['id']} 的 category_id {ann['category_id']} 不存在")
    
    if errors:
        return False, errors
    
    return True, []

if __name__ == "__main__":
    json_file = sys.argv[1]
    is_valid, errors = validate_coco_json(json_file)
    
    if is_valid:
        print(f"✅ {json_file} 格式正确")
    else:
        print(f"❌ {json_file} 格式错误:")
        for error in errors:
            print(f"  - {error}")
        sys.exit(1)
```

## 📚 参考资源

- COCO格式官方文档: http://cocodataset.org/#format-data
- Detectron2数据集教程: https://detectron2.readthedocs.io/tutorials/datasets.html
- pycocotools文档: https://github.com/cocodataset/cocoapi

## 总结

**你需要准备的文件：**

1. ✅ **train.json** - 训练集标注文件（COCO格式）
2. ✅ **val.json** - 验证集标注文件（COCO格式）
3. ✅ **图片文件** - 训练集和验证集的图片

**文件格式要求：**

- JSON文件必须是有效的COCO格式
- 必须包含 `categories`, `images`, `annotations` 字段
- 所有ID必须唯一且正确关联
- 图片文件名必须与JSON中的 `file_name` 匹配

**验证方法：**

- 使用 `pycocotools` 加载JSON文件检查
- 使用Detectron2注册数据集检查
- 使用验证脚本检查

