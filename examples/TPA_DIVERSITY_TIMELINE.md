# TPA Diversity趋势出现时间线

## 训练配置

- **数据集**: LVIS v1.0 (100,170 images)
- **Batch size**: 16
- **Max iterations**: 92,300
- **Iterations per epoch**: ~6,261
- **Total epochs**: ~14.7
- **Warmup steps**: 4,615 (5% of max_iter)
- **Warmup epochs**: ~0.74

## Diversity Loss生效时间线

### 阶段1: Warmup阶段 (0 - 0.7 epochs)

**特点**：
- `lambda_div` 从 0 逐渐增加到 0.12（cosine warmup）
- Diversity loss作用很弱，prototypes还在探索阶段
- **预期**：看不到明显差异化

**检查点**: Iter 4,615 (Epoch 0.7)

---

### 阶段2: 早期学习阶段 (0.7 - 1.5 epochs)

**特点**：
- `lambda_div` 达到最大值 0.12
- Diversity loss开始强作用
- Prototypes开始学习不同的attention模式
- **预期**：**应该能看到初步的差异化趋势** ⭐

**检查点**: Iter 9,230 (Epoch 1.5)

**验证方法**：
```bash
# 可视化attention热力图
python examples/visualize_tpa_for_cvpr.py \
    --checkpoint output/model_0009230.pth \
    --config lami_dino/configs/dino_convnext_large_4scale_12ep_lvis.py
```

**期望看到**：
- 不同prototypes的attention模式开始不同
- Attention pattern similarity < 0.8（之前可能是0.95+）

---

### 阶段3: 明显差异化阶段 (1.5 - 2.2 epochs)

**特点**：
- Diversity loss持续作用
- Prototypes的差异化更明显
- **预期**：**明显看到不同prototypes关注不同prompts** ⭐⭐

**检查点**: Iter 13,845 (Epoch 2.2)

**期望看到**：
- Attention热力图中，不同prototypes有明显不同的模式
- 每个prototype的top prompts不同
- Attention pattern similarity < 0.6

---

### 阶段4: 稳定学习阶段 (2.2+ epochs)

**特点**：
- Prototypes的差异化稳定下来
- 不同prototypes关注不同的prompts
- **预期**：**稳定的多样性模式** ⭐⭐⭐

**检查点**: 
- Iter 18,460 (Epoch 2.9)
- Iter 46,150 (Epoch 7.4 - 中期)
- Iter 92,300 (Epoch 14.7 - 最终)

**期望看到**：
- Attention pattern similarity < 0.5
- 每个prototype专注于不同的prompts
- t-SNE图中prototypes与prompts有更好的分离

## 快速验证方法

### 方法1: 检查训练日志

观察 `loss_div` 是否在下降：

```bash
# 查看训练日志
grep "loss_div" output/training.log | tail -20
```

**期望**：
- `loss_div` 应该逐渐下降
- 如果一直很高（>0.5），说明diversity loss没有生效

### 方法2: 使用分析工具

```bash
python examples/analyze_tpa_prototypes.py \
    --checkpoint output/model_0009230.pth \
    --config lami_dino/configs/dino_convnext_large_4scale_12ep_lvis.py
```

**关键指标**：
- **Attention pattern similarity**: 应该 < 0.6（越低越好）
- **Unique top prompts**: 应该接近 K（5个prototypes应该有5个不同的top prompts）
- **Prototype similarity**: 应该 < 0.3（越低越好）

### 方法3: 可视化检查

```bash
python examples/visualize_tpa_for_cvpr.py \
    --checkpoint output/model_0009230.pth \
    --config lami_dino/configs/dino_convnext_large_4scale_12ep_lvis.py
```

**检查attention热力图**：
- 不同prototypes（行）应该有不同的attention模式
- 不应该所有prototypes都关注相同的prompts（列）

## 总结

| 阶段 | Epochs | Iterations | 预期效果 |
|------|--------|------------|----------|
| Warmup | 0-0.7 | 0-4,615 | 无差异化 |
| **初步趋势** | **0.7-1.5** | **4,615-9,230** | **⭐ 开始看到差异化** |
| **明显差异化** | **1.5-2.2** | **9,230-13,845** | **⭐⭐ 明显不同** |
| 稳定状态 | 2.2+ | 13,845+ | ⭐⭐⭐ 稳定的多样性 |

## 建议

1. **最早检查**: Epoch 1.5 (Iter 9,230)
   - 如果这时还看不到差异化，可能需要调整超参数

2. **关键检查点**: Epoch 2.2 (Iter 13,845)
   - 这时应该能看到明显的差异化
   - 如果还没有，说明diversity loss可能还不够强

3. **持续监控**: 每0.7-1.0 epochs检查一次
   - 观察趋势是否在改善
   - 如果diversity loss一直很高，考虑增加`lambda_div`

## 如果看不到差异化怎么办？

1. **增加lambda_div**: 0.12 → 0.15-0.20
2. **检查训练日志**: 确认loss_div是否在下降
3. **检查梯度**: 确认TPA参数是否有梯度
4. **降低tau**: 0.10 → 0.08（更sharp的attention）

