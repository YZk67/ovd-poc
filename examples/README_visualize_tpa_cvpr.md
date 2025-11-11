# TPA可视化工具使用指南（CVPR论文）

这个工具专门用于生成CVPR论文中需要的TPA可视化图表。

## 功能

1. **Prototype Attention热力图** - 展示TPA如何选择/融合原始prompts
2. **Embedding Space可视化** - 展示prototypes vs 原始prompts的位置（t-SNE和PCA）
3. **Prototypes vs 简单平均对比** - 展示prototypes与简单平均的相似度

## 使用方法

### 基本用法

```bash
python examples/visualize_tpa_for_cvpr.py \
    --checkpoint output/model_final.pth \
    --config lami_dino/configs/dino_convnext_large_4scale_12ep_lvis.py \
    --prompts-json dataset2/metadata/lvis_prompts_claude.json \
    --output-dir tpa_cvpr_visualizations
```

### 参数说明

- `--checkpoint`: 训练好的模型checkpoint路径（必需）
- `--config`: 模型配置文件路径（默认：`lami_dino/configs/dino_convnext_large_4scale_12ep_lvis.py`）
- `--prompts-json`: Prompts JSON文件路径（默认：`dataset2/metadata/lvis_prompts_claude.json`）
- `--output-dir`: 输出目录（默认：`tpa_cvpr_visualizations`）

## 输出文件

运行后会生成以下文件：

1. **attention_heatmap.png** - Prototype Attention热力图
   - 展示5个代表性类别（dog, cat, car_(automobile), chair, person）
   - 显示每个prototype对原始prompts的attention权重

2. **embedding_space_tsne.png** - t-SNE降维可视化
   - 展示相似类别对（dog vs cat, car vs bicycle）
   - 显示prototypes、原始prompts和简单平均的位置

3. **embedding_space_pca.png** - PCA降维可视化
   - 同上，使用PCA方法

4. **prototype_vs_mean.png** - Prototypes vs 简单平均对比
   - 展示所有10个类别
   - 显示每个prototype与简单平均的相似度

## 选定的10个类别

工具会自动使用以下10个代表性类别：

1. dog (LVIS index: 377)
2. cat (LVIS index: 224)
3. bird (LVIS index: 98)
4. car_(automobile) (LVIS index: 206)
5. airplane (LVIS index: 2)
6. bicycle (LVIS index: 93)
7. chair (LVIS index: 231)
8. table (LVIS index: 1049)
9. person (LVIS index: 792)
10. bottle (LVIS index: 132)

这些类别定义在 `examples/tpa_visualization_classes.py` 中。

## 可视化说明

### 1. Attention热力图解读

- **行（Prototype Index）**: 每个prototype
- **列（Original Prompt Index）**: 每个原始prompt
- **颜色深浅**: Attention权重大小（0-1）
- **数值**: 大于0.1的权重会显示数值

**解读**:
- 如果某个prototype对某个prompt的attention很高，说明该prototype主要关注这个prompt
- 如果attention分布均匀，说明prototype融合了多个prompts的信息

### 2. Embedding Space可视化解读

- **红色/蓝色大圆点**: Prototypes
- **浅色小方块**: 原始Prompts
- **深色三角形**: 简单平均

**解读**:
- Prototypes应该位于原始prompts的"中心"或"优化位置"
- 不同类别的prototypes应该分离
- Prototypes与简单平均的位置差异展示了TPA的优化效果

### 3. Prototypes vs Mean对比解读

- **Y轴**: 与简单平均的余弦相似度（0-1）
- **X轴**: 类别
- **不同颜色**: 不同的prototype

**解读**:
- 相似度接近1：prototype与简单平均相似
- 相似度较低：prototype学习了不同的表示
- 多个prototypes的相似度分布展示了多样性

## 常见问题

### Q: 找不到TPA？

确保模型使用了TPA（`use_tpa=True`），并且checkpoint路径正确。

### Q: 类别名称不匹配？

检查 `dataset2/metadata/lvis_prompts_claude.json` 中的类别名称是否与 `examples/tpa_visualization_classes.py` 中定义的一致。

### Q: 内存不足？

如果遇到内存问题，可以修改代码只可视化部分类别，或者使用CPU模式。

## 论文使用建议

1. **主论文Figure**: 
   - 使用 `attention_heatmap.png`（选择3-5个类别）
   - 使用 `embedding_space_tsne.png`（选择1-2个相似类别对）

2. **补充材料**:
   - 所有10个类别的attention热力图
   - 更多embedding space可视化
   - Prototypes vs Mean对比图

## 依赖

确保安装了以下包：

```bash
pip install matplotlib numpy scikit-learn torch
```

