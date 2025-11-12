# 训练配置检查报告

## ✅ TPA配置检查

### 主配置文件：`lami_dino/configs/dino_convnext_large_4scale_12ep_lvis.py`

| 配置项 | 值 | 状态 | 说明 |
|--------|-----|------|------|
| `use_tpa` | `True` | ✅ | TPA已启用 |
| `tpa_lambda_orth` | `0.20` | ✅ | 修复后的值（从0.10增加） |
| `tpa_lambda_div` | `0.12` | ✅ | 修复后的值（从0.03增加4倍） |
| `tpa_tau` | `0.10` | ✅ | Softer attention（从0.07增加） |
| `tpa_warmup_steps` | `4615` | ✅ | 5% of max_iter (92300) |
| `tpa_num_prototypes` | `5` | ✅ | 正确 |
| `tpa_hidden_dim` | `256` | ✅ | 正确 |
| `tpa_dropout` | `0.05` | ✅ | 正确 |
| `tpa_log_interval` | `200` | ✅ | 正确 |

### 代码修复检查

| 修复项 | 状态 | 说明 |
|--------|------|------|
| `_step`注册为buffer | ✅ | 已修复，会被保存到checkpoint |
| `tpa_warmup_steps`参数传递 | ✅ | TextClassifier已支持 |
| Diversity loss实现 | ✅ | 已修复，使用相似度矩阵 |

## ✅ 训练配置检查

| 配置项 | 值 | 状态 |
|--------|-----|------|
| `max_iter` | `92300` | ✅ |
| `total_batch_size` | `16` | ✅ |
| `warmup_ratio` | `5%` (4615/92300) | ✅ |

## 📋 训练前检查清单

### 1. 确认代码已更新
```bash
# 检查_step是否已注册为buffer
grep -n "register_buffer.*_step" lami_dino/models/text_prototype_aggregator.py
# 应该看到：self.register_buffer("_step", torch.tensor(0, dtype=torch.long))
```

### 2. 确认配置文件路径
```bash
# 检查配置文件是否存在
ls lami_dino/configs/dino_convnext_large_4scale_12ep_lvis.py
```

### 3. 确认text embedding文件
```bash
# 检查text embedding文件是否存在
ls dataset/metadata/lvis_claude_prompts_convnextl.npy
```

### 4. 确认pretrained model
```bash
# 检查pretrained model是否存在
ls ./pretrained_models/lami_convnext_large_12ep_lvis/model_final.pth
```

## 🚀 开始训练

### 训练命令
```bash
python tools/train_net.py \
    --config-file lami_dino/configs/dino_convnext_large_4scale_12ep_lvis.py
```

### 训练监控

#### 1. 观察训练日志
重点关注以下指标：
- `loss_apr`: APR loss（应该逐渐增加，然后稳定）
- `loss_div`: Diversity loss（应该逐渐下降）
- `loss_orth`: Orthogonality loss（应该逐渐下降）

#### 2. 检查点
- **1.5 epochs (iter 9,230)**: 应该能看到初步的prototype差异化
- **2.2 epochs (iter 13,845)**: 应该能看到明显的prototype差异化

#### 3. 使用诊断工具检查
```bash
# 在1.5 epochs后检查
python examples/diagnose_prototype_diversity.py \
    --checkpoint <checkpoint_path> \
    --config lami_dino/configs/dino_convnext_large_4scale_12ep_lvis.py \
    --text-embed dataset/metadata/lvis_claude_prompts_convnextl.npy
```

**期望结果**：
- Attention Pattern Similarity < 0.6
- Unique Top Prompts Ratio > 0.8
- Prototype Similarity < 0.3

## ⚠️ 注意事项

1. **确保使用修复后的代码**
   - `_step`已注册为buffer
   - Diversity loss实现已修复

2. **训练时间**
   - 需要至少1.5-2.2 epochs才能看到prototype差异化
   - 完整训练需要约14.7 epochs

3. **如果prototypes仍然相似**
   - 检查训练日志中的`loss_div`是否在下降
   - 如果`loss_div`一直很高，可能需要进一步增加`lambda_div`
   - 如果`loss_div`在下降但prototypes仍然相似，可能需要更多训练时间

## 📊 预期训练曲线

### Warmup阶段 (0-0.74 epochs, iter 0-4615)
- `loss_apr`很小（因为effective lambda很小）
- `loss_div`和`loss_orth`作用很弱

### 早期学习阶段 (0.74-1.5 epochs, iter 4615-9230)
- `loss_apr`开始增加
- `loss_div`开始下降
- Prototypes开始学习不同的attention模式

### 明显差异化阶段 (1.5-2.2 epochs, iter 9230-13845)
- `loss_div`持续下降
- Prototypes的差异化更明显
- 应该能看到不同的attention模式

### 稳定学习阶段 (2.2+ epochs, iter 13845+)
- Prototypes的差异化稳定下来
- 不同prototypes关注不同的prompts

## ✅ 总结

**配置检查通过！可以开始训练。**

所有TPA相关配置都已正确设置：
- ✅ Lambda值已更新（lambda_orth=0.20, lambda_div=0.12）
- ✅ Warmup steps已正确设置（4615）
- ✅ 代码已修复（_step保存、diversity loss实现）
- ✅ 所有参数都已正确传递

开始训练后，记得在1.5-2.2 epochs时检查prototype多样性！

