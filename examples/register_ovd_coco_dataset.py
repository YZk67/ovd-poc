"""
示例：如何注册和使用OVD-COCO数据集

这个脚本展示了如何快速添加一个新的COCO格式数据集。
只需要运行这个脚本注册数据集，然后就可以在配置文件中使用了。
"""

import os
from detectron2.data.datasets import register_coco_instances

# ============================================================================
# 步骤1: 注册数据集
# ============================================================================

# 假设你的数据集结构如下：
# dataset/
#   ovd_coco/
#     images/
#       train/
#       val/
#     annotations/
#       train.json
#       val.json

# 设置数据集根目录（根据实际情况修改）
DATASET_ROOT = "dataset"  # 或 os.getenv("DETECTRON2_DATASETS", "dataset")
OVD_COCO_ROOT = os.path.join(DATASET_ROOT, "ovd_coco")

# 注册训练集
register_coco_instances(
    "ovd_coco_train",
    {},  # 元数据（可以为空，会自动从JSON中读取）
    os.path.join(OVD_COCO_ROOT, "annotations", "train.json"),
    os.path.join(OVD_COCO_ROOT, "images", "train"),
)

# 注册验证集
register_coco_instances(
    "ovd_coco_val",
    {},
    os.path.join(OVD_COCO_ROOT, "annotations", "val.json"),
    os.path.join(OVD_COCO_ROOT, "images", "val"),
)

# 可选：注册测试集
# register_coco_instances(
#     "ovd_coco_test",
#     {},
#     os.path.join(OVD_COCO_ROOT, "annotations", "test.json"),
#     os.path.join(OVD_COCO_ROOT, "images", "test"),
# )

print("✅ 数据集注册成功！")
print("   训练集: ovd_coco_train")
print("   验证集: ovd_coco_val")

# ============================================================================
# 步骤2: 验证数据集（可选）
# ============================================================================

def verify_dataset():
    """验证数据集是否正确注册"""
    from detectron2.data import DatasetCatalog, MetadataCatalog
    
    # 检查数据集是否已注册
    train_name = "ovd_coco_train"
    val_name = "ovd_coco_val"
    
    if train_name in DatasetCatalog.list():
        print(f"✅ {train_name} 已注册")
        dataset_dicts = DatasetCatalog.get(train_name)
        print(f"   训练集包含 {len(dataset_dicts)} 张图片")
        
        # 检查元数据
        metadata = MetadataCatalog.get(train_name)
        if hasattr(metadata, "thing_classes"):
            print(f"   类别数量: {len(metadata.thing_classes)}")
            print(f"   前5个类别: {metadata.thing_classes[:5]}")
    else:
        print(f"❌ {train_name} 未注册")
    
    if val_name in DatasetCatalog.list():
        print(f"✅ {val_name} 已注册")
        dataset_dicts = DatasetCatalog.get(val_name)
        print(f"   验证集包含 {len(dataset_dicts)} 张图片")
    else:
        print(f"❌ {val_name} 未注册")

if __name__ == "__main__":
    verify_dataset()

