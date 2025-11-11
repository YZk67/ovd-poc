# TPA训练过程可视化指南

本工具用于可视化TPA（Text Prototype Aggregator）在训练过程中的效果和作用。

## 功能特性

### 1. 训练曲线可视化
- **loss_apr**: APR损失（正交性损失 + 多样性损失）
- **loss_orth**: 正交性损失
- **loss_div**: 多样性损失
- **orth_off_mse**: 非对角线正交性MSE
- **diag_mse**: 对角线MSE
- **usage_entropy**: Prototype使用熵
- **lambda_orth / lambda_div**: 正则化系数（带warmup）

### 2. Prototype相似度矩阵
- 可视化每个类别内prototypes之间的相似度
- 帮助理解prototypes是否学习到不同的表示

### 3. Attention权重分布
- 可视化每个prototype的attention权重
- 了解哪些prototypes被更多地使用

### 4. Embedding Space可视化
- t-SNE和PCA降维可视化
- 观察prototypes在embedding space中的分布

### 5. 正交性热力图
- 所有prototypes之间的相似度矩阵
- 评估整体正交性

## 使用方法

### 方法1: 使用示例脚本（推荐）

#### 1.1 可视化训练曲线（仅需CSV日志）

```bash
python examples/visualize_tpa_training.py \
    --mode curves \
    --csv output/training_log.csv \
    --output-dir tpa_visualizations
```

#### 1.2 完整可视化报告（需要checkpoint）

```bash
python examples/visualize_tpa_training.py \
    --mode comprehensive \
    --checkpoint output/model_final.pth \
    --config lami_dino/configs/dino_convnext_large_4scale_12ep_lvis.py \
    --class-names dataset2/lvis/lvis_v1_all_classes.json \
    --csv output/training_log.csv \
    --output-dir tpa_visualizations
```

### 方法2: 直接使用Python API

```python
from lami_dino.visuliazation.tpa_training_visualizer import (
    TPATrainingVisualizer,
    load_model_from_checkpoint
)

# 创建可视化器
visualizer = TPATrainingVisualizer(output_dir="tpa_visualizations")

# 1. 可视化训练曲线
visualizer.visualize_training_curves("output/training_log.csv")

# 2. 加载模型并创建完整报告
model = load_model_from_checkpoint(
    checkpoint_path="output/model_final.pth",
    config_path="lami_dino/configs/dino_convnext_large_4scale_12ep_lvis.py"
)

# 加载类别名称（可选）
import json
with open("dataset2/lvis/lvis_v1_all_classes.json", 'r') as f:
    class_names = json.load(f)

# 创建综合报告
visualizer.create_comprehensive_report(
    model=model,
    csv_path="output/training_log.csv",
    class_names=class_names
)
```

## 输出文件说明

运行完整可视化后，会在输出目录生成以下文件：

1. **tpa_report_curves.png**: 训练曲线图
   - 显示所有TPA相关指标随训练迭代的变化

2. **tpa_report_similarity.png**: Prototype相似度矩阵
   - 每个类别内prototypes的相似度热力图

3. **tpa_report_attention.png**: Attention权重分布
   - 每个prototype的平均attention权重

4. **tpa_report_embedding_tsne.png**: t-SNE可视化
   - Prototypes在embedding space中的2D分布（t-SNE降维）

5. **tpa_report_embedding_pca.png**: PCA可视化
   - Prototypes在embedding space中的2D分布（PCA降维）

6. **tpa_report_orthogonality.png**: 正交性热力图
   - 所有prototypes之间的相似度矩阵

## 如何解读可视化结果

### 训练曲线解读

1. **loss_apr**: 应该逐渐下降，表示正则化损失在减小
2. **loss_orth**: 应该下降，表示prototypes逐渐变得正交
3. **loss_div**: 应该下降，表示prototypes使用更加均匀
4. **orth_off_mse**: 越小越好，表示prototypes之间越正交
5. **usage_entropy**: 接近1表示prototypes使用均匀，接近0表示某些prototypes很少被使用

### Prototype相似度矩阵解读

- **对角线元素**: 应该接近1（每个prototype与自身相似）
- **非对角线元素**: 应该接近0（不同prototypes应该正交）
- 如果非对角线元素较大，说明prototypes没有学习到足够的区分度

### Attention权重分布解读

- **均匀分布**: 所有prototypes被均匀使用（理想情况）
- **集中分布**: 某些prototypes被过度使用，可能表示冗余
- **稀疏分布**: 某些prototypes很少被使用，可能需要调整正则化

### Embedding Space可视化解读

- **聚类**: 同一类别的prototypes应该聚集在一起
- **分离**: 不同类别的prototypes应该分离
- **分布**: 均匀分布表示prototypes覆盖了足够的语义空间

## 常见问题

### Q: 如何生成训练日志CSV？

训练日志CSV由`TrainLogger`自动生成。确保在训练配置中启用了`TrainLogger`：

```python
from lami_dino.utils.logger import TrainLogger

train_logger = TrainLogger(save_dir=cfg.train.output_dir, interval=200)
# 在训练循环中调用
train_logger.log(iteration, model, losses)
```

### Q: 如何获取类别名称？

如果使用LVIS数据集，可以从以下文件加载：

```python
# 方法1: 从LVIS JSON文件
import json
with open("dataset2/lvis/lvis_v1_all_classes.json", 'r') as f:
    class_names = json.load(f)

# 方法2: 从LVIS annotation文件提取
from tools.generate_class_prompts import load_categories
categories = load_categories("dataset2/lvis/lvis_v1_train_norare_cat_info.json")
class_names = [cat["name"] for cat in sorted(categories, key=lambda x: x["id"])]
```

### Q: 可视化时找不到TPA？

确保：
1. 模型使用了TPA（`use_tpa=True`）
2. Checkpoint路径正确
3. 模型结构匹配配置文件

### Q: 如何可视化特定类别？

可以修改代码，在调用可视化函数时指定`class_indices`参数：

```python
visualizer.visualize_prototype_similarity(
    prototypes,
    class_indices=[0, 1, 2, 3, 4],  # 只可视化前5个类别
    class_names=class_names
)
```

## 高级用法

### 对比多个Checkpoint

```python
checkpoints = [
    "output/model_0001000.pth",
    "output/model_0005000.pth",
    "output/model_final.pth"
]

for ckpt in checkpoints:
    model = load_model_from_checkpoint(ckpt, config_path)
    visualizer.create_comprehensive_report(
        model,
        prefix=f"tpa_report_{Path(ckpt).stem}"
    )
```

### 自定义可视化

```python
# 只可视化特定指标
visualizer.visualize_training_curves(
    "output/training_log.csv",
    metrics=['loss_apr', 'orth_off_mse', 'usage_entropy']
)

# 使用PCA而不是t-SNE（更快）
visualizer.visualize_prototype_embedding_space(
    prototypes,
    method='pca',
    save_name="prototypes_pca.png"
)
```

## 依赖项

确保安装了以下Python包：

```bash
pip install matplotlib seaborn pandas scikit-learn numpy torch
```

## 参考

- TPA实现: `lami_dino/models/text_prototype_aggregator.py`
- 训练日志: `lami_dino/utils/logger.py`
- 后处理分析: `analysis/post_eval_analyzer.py`

