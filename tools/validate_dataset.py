#!/usr/bin/env python3
"""
数据集格式验证工具

用法:
    python tools/validate_dataset.py path/to/annotations/train.json
    python tools/validate_dataset.py path/to/annotations/val.json
"""

import json
import sys
from pathlib import Path
from typing import List, Tuple


def validate_coco_json(json_file: str) -> Tuple[bool, List[str]]:
    """
    验证COCO格式JSON文件
    
    Args:
        json_file: JSON文件路径
        
    Returns:
        (is_valid, errors): 是否有效和错误列表
    """
    errors = []
    warnings = []
    
    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        return False, [f"JSON格式错误: {e}"]
    except FileNotFoundError:
        return False, [f"文件不存在: {json_file}"]
    
    # 检查必需字段
    required_keys = ['categories', 'images', 'annotations']
    for key in required_keys:
        if key not in data:
            errors.append(f"缺少必需字段: {key}")
    
    if errors:
        return False, errors
    
    # 检查categories
    categories = data['categories']
    if not isinstance(categories, list) or len(categories) == 0:
        errors.append("categories必须是非空列表")
    else:
        cat_ids = []
        cat_names = []
        for i, cat in enumerate(categories):
            if 'id' not in cat:
                errors.append(f"categories[{i}] 缺少 'id' 字段")
            else:
                cat_ids.append(cat['id'])
            
            if 'name' not in cat:
                errors.append(f"categories[{i}] 缺少 'name' 字段")
            else:
                cat_names.append(cat['name'])
        
        # 检查ID重复
        if len(cat_ids) != len(set(cat_ids)):
            duplicates = [x for x in cat_ids if cat_ids.count(x) > 1]
            errors.append(f"类别ID重复: {set(duplicates)}")
        
        # 检查名称重复
        if len(cat_names) != len(set(cat_names)):
            duplicates = [x for x in cat_names if cat_names.count(x) > 1]
            errors.append(f"类别名称重复: {set(duplicates)}")
    
    # 检查images
    images = data['images']
    if not isinstance(images, list):
        errors.append("images必须是列表")
    else:
        img_ids = []
        img_files = []
        for i, img in enumerate(images):
            if 'id' not in img:
                errors.append(f"images[{i}] 缺少 'id' 字段")
            else:
                img_ids.append(img['id'])
            
            if 'file_name' not in img:
                errors.append(f"images[{i}] 缺少 'file_name' 字段")
            else:
                img_files.append(img['file_name'])
            
            if 'width' not in img:
                errors.append(f"images[{i}] 缺少 'width' 字段")
            if 'height' not in img:
                errors.append(f"images[{i}] 缺少 'height' 字段")
        
        # 检查ID重复
        if len(img_ids) != len(set(img_ids)):
            duplicates = [x for x in img_ids if img_ids.count(x) > 1]
            errors.append(f"图片ID重复: {set(duplicates)}")
    
    # 检查annotations
    annotations = data['annotations']
    if not isinstance(annotations, list):
        errors.append("annotations必须是列表")
    else:
        ann_ids = []
        for i, ann in enumerate(annotations):
            if 'id' not in ann:
                errors.append(f"annotations[{i}] 缺少 'id' 字段")
            else:
                ann_ids.append(ann['id'])
            
            if 'image_id' not in ann:
                errors.append(f"annotations[{i}] 缺少 'image_id' 字段")
            elif img_ids and ann['image_id'] not in img_ids:
                errors.append(f"annotations[{i}] 的 image_id {ann['image_id']} 在images中不存在")
            
            if 'category_id' not in ann:
                errors.append(f"annotations[{i}] 缺少 'category_id' 字段")
            elif cat_ids and ann['category_id'] not in cat_ids:
                errors.append(f"annotations[{i}] 的 category_id {ann['category_id']} 在categories中不存在")
            
            if 'bbox' not in ann:
                errors.append(f"annotations[{i}] 缺少 'bbox' 字段")
            else:
                bbox = ann['bbox']
                if not isinstance(bbox, list) or len(bbox) != 4:
                    errors.append(f"annotations[{i}] 的 bbox 必须是长度为4的列表 [x, y, width, height]")
                else:
                    x, y, w, h = bbox
                    if w <= 0 or h <= 0:
                        errors.append(f"annotations[{i}] 的 bbox 宽度或高度必须大于0")
            
            if 'area' not in ann:
                warnings.append(f"annotations[{i}] 缺少 'area' 字段（推荐但不必需）")
            
            if 'iscrowd' not in ann:
                warnings.append(f"annotations[{i}] 缺少 'iscrowd' 字段（推荐但不必需）")
        
        # 检查ID重复
        if len(ann_ids) != len(set(ann_ids)):
            duplicates = [x for x in ann_ids if ann_ids.count(x) > 1]
            errors.append(f"标注ID重复: {set(duplicates)}")
    
    # 统计信息
    info = {
        "categories": len(categories) if categories else 0,
        "images": len(images) if images else 0,
        "annotations": len(annotations) if annotations else 0,
    }
    
    return len(errors) == 0, errors, warnings, info


def main():
    if len(sys.argv) < 2:
        print("用法: python validate_dataset.py <json_file>")
        print("示例: python validate_dataset.py dataset/ovd_coco/annotations/train.json")
        sys.exit(1)
    
    json_file = sys.argv[1]
    
    print("=" * 80)
    print(f"验证数据集文件: {json_file}")
    print("=" * 80)
    
    is_valid, errors, warnings, info = validate_coco_json(json_file)
    
    print(f"\n📊 统计信息:")
    print(f"  类别数量: {info['categories']}")
    print(f"  图片数量: {info['images']}")
    print(f"  标注数量: {info['annotations']}")
    
    if warnings:
        print(f"\n⚠️  警告 ({len(warnings)} 个):")
        for warning in warnings[:10]:  # 只显示前10个警告
            print(f"  - {warning}")
        if len(warnings) > 10:
            print(f"  ... 还有 {len(warnings) - 10} 个警告")
    
    if is_valid:
        print(f"\n✅ 数据集格式正确！")
        sys.exit(0)
    else:
        print(f"\n❌ 数据集格式错误 ({len(errors)} 个错误):")
        for error in errors:
            print(f"  - {error}")
        sys.exit(1)


if __name__ == "__main__":
    main()

