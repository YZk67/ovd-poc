#!/usr/bin/env python3
"""
创建COCO zero-shot的类别列表文件

用法:
    python tools/create_coco_zeroshot_class_lists.py
"""

import json
import os
import sys

# 添加detectron2路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    from detectron2.data.datasets.builtin_meta import (
        COCO_SEEN_CLASSES, 
        COCO_UNSEEN_CLASSES,
        COCO_CATEGORIES
    )
except ImportError:
    print("❌ 无法导入detectron2，请确保已安装detectron2")
    sys.exit(1)


def create_class_lists():
    """创建COCO zero-shot的类别列表文件"""
    
    # 获取所有COCO类别（只包含thing类别）
    all_coco_classes = [cat['name'] for cat in COCO_CATEGORIES if cat['isthing'] == 1]
    
    # 创建seen classes列表（base类）
    seen_classes = list(COCO_SEEN_CLASSES)
    
    # 创建unseen classes列表（novel类）
    unseen_classes = list(COCO_UNSEEN_CLASSES)
    
    # 创建all classes列表（按COCO顺序，包含所有80个类别）
    all_classes = all_coco_classes
    
    # 验证类别数量
    print("=" * 80)
    print("COCO Zero-Shot类别统计")
    print("=" * 80)
    print(f"Seen (Base) Classes: {len(seen_classes)}")
    print(f"Unseen (Novel) Classes: {len(unseen_classes)}")
    print(f"All Classes: {len(all_classes)}")
    print(f"Total (Seen + Unseen): {len(seen_classes) + len(unseen_classes)}")
    
    # 验证seen + unseen是否等于all
    seen_set = set(seen_classes)
    unseen_set = set(unseen_classes)
    all_set = set(all_classes)
    
    if seen_set | unseen_set != all_set:
        print("⚠️  警告: Seen和Unseen类的并集不等于All类")
        print(f"   Seen类中不在All类中的: {seen_set - all_set}")
        print(f"   Unseen类中不在All类中的: {unseen_set - all_set}")
        print(f"   All类中不在Seen或Unseen中的: {all_set - (seen_set | unseen_set)}")
    else:
        print("✅ Seen和Unseen类的并集等于All类")
    
    # 创建输出目录
    output_dir = "dataset/coco"
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存文件
    files_created = []
    
    # 1. Seen classes
    seen_file = f"{output_dir}/coco_seen_classes.json"
    with open(seen_file, 'w') as f:
        json.dump(seen_classes, f, indent=2)
    files_created.append(seen_file)
    print(f"\n✅ 创建: {seen_file} ({len(seen_classes)} 类)")
    
    # 2. Unseen classes
    unseen_file = f"{output_dir}/coco_unseen_classes.json"
    with open(unseen_file, 'w') as f:
        json.dump(unseen_classes, f, indent=2)
    files_created.append(unseen_file)
    print(f"✅ 创建: {unseen_file} ({len(unseen_classes)} 类)")
    
    # 3. All classes
    all_file = f"{output_dir}/coco_all_classes.json"
    with open(all_file, 'w') as f:
        json.dump(all_classes, f, indent=2)
    files_created.append(all_file)
    print(f"✅ 创建: {all_file} ({len(all_classes)} 类)")
    
    # 显示前几个类别作为示例
    print("\n" + "=" * 80)
    print("类别示例")
    print("=" * 80)
    print(f"Seen类（前5个）: {seen_classes[:5]}")
    print(f"Unseen类（前5个）: {unseen_classes[:5]}")
    print(f"All类（前5个）: {all_classes[:5]}")
    
    print("\n" + "=" * 80)
    print("✅ 完成！所有类别列表文件已创建")
    print("=" * 80)
    
    return files_created


if __name__ == "__main__":
    try:
        create_class_lists()
    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

