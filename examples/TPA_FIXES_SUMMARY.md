# TPA修复总结

## 修复的问题

### 1. ✅ Diversity Loss设计问题（已修复）

**问题**：
- 原来的diversity loss鼓励所有prototypes的"使用频率"均匀
- 但不鼓励不同prototypes关注不同的prompts
- 导致所有prototypes学到相同的attention模式

**修复**：
- 修改了`_diversity_term`函数
- 新版本直接最小化不同prototypes之间的attention分布相似度
- 鼓励不同prototypes关注不同的prompts

**文件**：`lami_dino/models/text_prototype_aggregator.py` (line 143-174)

### 2. ✅ 初始化改进（已修复）

**问题**：
- prototype_queries使用xavier初始化，可能导致初始状态太相似

**修复**：
- 使用正交初始化（QR分解）
- 让初始queries更不同，鼓励从一开始就关注不同方面

**文件**：`lami_dino/models/text_prototype_aggregator.py` (line 72-100)

### 3. ✅ 超参数调整（已更新）

**修改**：
- `lambda_orth`: 0.10 → 0.20 (2x增加，更强正交性)
- `lambda_div`: 0.03 → 0.12 (4x增加，更强多样性)
- `tau`: 0.07 → 0.10 (更soft的attention，允许更多探索)

**文件**：
- `lami_dino/configs/dino_convnext_large_4scale_12ep_lvis.py` (已更新)
- `lami_dino/configs/models/dino_convnextl.py` (已更新默认值)
- `lami_dino/modeling/text_classifier.py` (已添加参数支持)

### 4. ✅ 参数传递支持（已添加）

**修复**：
- 在`TextClassifier`中添加了`tpa_lambda_orth`和`tpa_lambda_div`参数
- 确保这些参数能正确传递到`TextPrototypeAggregator`

**文件**：`lami_dino/modeling/text_classifier.py`

## 修复后的预期效果

1. **不同prototypes关注不同prompts**
   - Attention热力图应该显示不同的模式
   - 每个prototype应该有不同的top prompts

2. **Prototypes与原始prompts有更大差异**
   - t-SNE图中prototypes应该与prompts有更明显的分离
   - Prototypes应该学到新的语义表示

3. **不同类别的prototypes更好分离**
   - 相似类别（如dog vs cat）的prototypes应该有更好的分离
   - 提高分类性能

## 使用方法

### 使用修复后的配置

```bash
# 使用修复后的配置文件
python tools/train_net.py \
    --config-file lami_dino/configs/dino_convnext_large_4scale_12ep_lvis.py
```

### 验证修复效果

训练后，使用可视化工具验证：

```bash
python examples/visualize_tpa_for_cvpr.py \
    --checkpoint output/model_final.pth \
    --config lami_dino/configs/dino_convnext_large_4scale_12ep_lvis.py \
    --prompts-json dataset2/metadata/lvis_prompts_claude.json
```

**期望看到**：
- Attention热力图中，不同prototypes有不同的attention模式
- t-SNE图中，prototypes与prompts有更明显的分离
- 不同类别的prototypes有更好的分离

## 技术细节

### 新的Diversity Loss

```python
def _diversity_term(self, logits: torch.Tensor) -> torch.Tensor:
    C, K, N = logits.shape
    w = torch.softmax(logits, dim=1)  # [C, K, N]
    
    diversity_loss = 0.0
    for c in range(C):
        attn_c = w[c]  # [K, N]
        attn_norm = F.normalize(attn_c, p=2, dim=1)
        similarity_matrix = torch.mm(attn_norm, attn_norm.t())  # [K, K]
        mask = ~torch.eye(K, dtype=bool, device=similarity_matrix.device)
        off_diag_similarities = similarity_matrix[mask]
        diversity_loss += off_diag_similarities.mean()
    
    return diversity_loss / C
```

**原理**：
- 计算不同prototypes的attention分布之间的余弦相似度
- 最小化相似度 = 最大化多样性
- 直接鼓励不同prototypes关注不同的prompts

### 正交初始化

```python
init_queries = torch.randn(self.num_prototypes, hidden_dim)
Q, R = torch.linalg.qr(init_queries)  # 或 torch.qr
self.prototype_queries.data = Q * 0.1
```

**原理**：
- 使用QR分解生成正交向量
- 确保初始queries不同
- 小尺度（0.1）避免初始状态过大

## 注意事项

1. **需要重新训练**
   - 这些修复需要从头训练才能看到效果
   - 不能直接应用到已训练的模型

2. **超参数可能需要微调**
   - 如果训练不稳定，可以适当降低lambda_div
   - 如果多样性仍然不足，可以进一步增加lambda_div

3. **监控训练过程**
   - 观察`loss_orth`和`loss_div`是否在下降
   - 使用`examples/analyze_tpa_prototypes.py`定期检查多样性

## 相关文件

- `lami_dino/models/text_prototype_aggregator.py` - TPA核心实现（已修复）
- `lami_dino/modeling/text_classifier.py` - TextClassifier（已更新）
- `lami_dino/configs/dino_convnext_large_4scale_12ep_lvis.py` - 配置文件（已更新）
- `examples/visualize_tpa_for_cvpr.py` - 可视化工具
- `examples/analyze_tpa_prototypes.py` - 诊断工具
- `examples/tpa_diversity_loss_fix.md` - 详细修复说明

