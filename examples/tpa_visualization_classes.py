"""
用于TPA可视化的10个代表性类别

这些类别被选择用于CVPR论文中的可视化，展示TPA的作用和效果。
"""

# 选定的10个类别及其LVIS索引
SELECTED_CLASSES = [
    # 动物类（3个）- 展示相似类别的区分
    "dog",                    # LVIS index: 377
    "cat",                    # LVIS index: 224  
    "bird",                   # LVIS index: 98
    
    # 交通工具（3个）- 展示不同交通工具的区分
    "car_(automobile)",       # LVIS index: 206
    "airplane",               # LVIS index: 2
    "bicycle",                # LVIS index: 93
    
    # 家具（2个）- 展示相关但不同类别的区分
    "chair",                  # LVIS index: 231
    "table",                  # LVIS index: 1049
    
    # 其他（2个）
    "person",                 # LVIS index: 792
    "bottle",                 # LVIS index: 132
]

# LVIS索引映射（用于从模型中提取prototypes）
CLASS_INDICES = {
    "dog": 377,
    "cat": 224,
    "bird": 98,
    "car_(automobile)": 206,
    "airplane": 2,
    "bicycle": 93,
    "chair": 231,
    "table": 1049,
    "person": 792,
    "bottle": 132,
}

# 类别分组（用于可视化组织）
CLASS_GROUPS = {
    "animals": ["dog", "cat", "bird"],
    "vehicles": ["car_(automobile)", "airplane", "bicycle"],
    "furniture": ["chair", "table"],
    "others": ["person", "bottle"],
}

# 相似类别对（用于对比可视化）
SIMILAR_PAIRS = [
    ("dog", "cat"),                    # 相似动物
    ("car_(automobile)", "bicycle"),  # 交通工具
    ("chair", "table"),                # 家具
]

# 可视化建议
VISUALIZATION_SUGGESTIONS = {
    # Prototype Attention热力图 - 选择5个代表性类别
    "attention_heatmap": [
        "dog",
        "cat", 
        "car_(automobile)",
        "chair",
        "person"
    ],
    
    # Embedding Space可视化 - 选择相似类别对
    "embedding_space": [
        ("dog", "cat"),                    # 展示如何区分相似动物
        ("car_(automobile)", "bicycle"),  # 展示如何区分交通工具
    ],
    
    # 性能分析 - 使用全部10个类别
    "performance_analysis": SELECTED_CLASSES,
    
    # 典型成功案例 - 选择3-5个类别
    "success_cases": [
        "dog",
        "car_(automobile)",
        "chair",
    ],
}

def get_class_index(class_name: str) -> int:
    """获取类别的LVIS索引"""
    return CLASS_INDICES.get(class_name, -1)

def get_class_group(class_name: str) -> str:
    """获取类别所属的分组"""
    for group, classes in CLASS_GROUPS.items():
        if class_name in classes:
            return group
    return "unknown"

if __name__ == "__main__":
    print("=" * 70)
    print("TPA可视化选定的10个类别")
    print("=" * 70)
    print(f"\n总类别数: {len(SELECTED_CLASSES)}")
    print("\n类别列表：")
    for i, cls in enumerate(SELECTED_CLASSES, 1):
        idx = get_class_index(cls)
        group = get_class_group(cls)
        print(f"  {i:2d}. {cls:25s} (LVIS index: {idx:4d}, 分组: {group})")
    
    print("\n" + "=" * 70)
    print("可视化建议：")
    print("=" * 70)
    print("\n1. Prototype Attention热力图:")
    for cls in VISUALIZATION_SUGGESTIONS["attention_heatmap"]:
        print(f"   - {cls}")
    
    print("\n2. Embedding Space可视化:")
    for pair in VISUALIZATION_SUGGESTIONS["embedding_space"]:
        print(f"   - {pair[0]} vs {pair[1]}")
    
    print("\n3. 性能分析:")
    print(f"   - 使用全部 {len(SELECTED_CLASSES)} 个类别")
    
    print("\n4. 相似类别对（用于对比分析）:")
    for pair in SIMILAR_PAIRS:
        print(f"   - {pair[0]} vs {pair[1]}")

