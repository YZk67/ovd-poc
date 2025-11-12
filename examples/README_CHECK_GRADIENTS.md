# 如何检查prototype_queries的梯度是否在更新

## 方法1：使用诊断脚本（推荐）

### 使用check_tpa_gradients.py

```bash
python examples/check_tpa_gradients.py \
    --checkpoint <checkpoint_path> \
    --config lami_dino/configs/dino_convnext_large_4scale_12ep_lvis.py
```

### 检查内容

1. **prototype_queries的参数信息**
   - shape, requires_grad, dtype, device
   - 参数值的统计信息（mean, std, min, max）

2. **prototype_queries之间的相似度**
   - 如果相似度>0.99 → 几乎完全相同
   - 如果相似度<0.9 → 有差异

3. **prototype_queries的梯度**
   - 梯度是否存在
   - 梯度的大小（mean, std, max, min）
   - 每个prototype的梯度大小

4. **APR Loss信息**
   - loss_orth, loss_div的值
   - effective lambda的值

### 诊断指标

- ✅ **梯度存在且非零** → 说明在更新
- ❌ **梯度为零或不存在** → 说明没有更新
- ⚠️ **prototypes相似度>0.99** → 说明几乎完全相同

---

## 方法2：在训练代码中添加检查（实时监控）

### 在train_net.py中添加

```python
# 在run_step()中，backward()之后添加
if self.iter % 100 == 0:  # 每100次迭代检查一次
    text_classifier = self.model.transformer.decoder.class_embed[0]
    if hasattr(text_classifier, 'tpa'):
        tpa = text_classifier.tpa
        prototype_queries = tpa.prototype_queries
        
        if prototype_queries.grad is not None:
            grad_norm = prototype_queries.grad.norm().item()
            print(f"[Gradient Check] iter={self.iter}, grad_norm={grad_norm:.6f}")
        else:
            print(f"[Gradient Check] iter={self.iter}, grad=None")
```

---

## 方法3：使用PyTorch的hook（详细检查）

### 添加hook来监控梯度

```python
def check_gradients(model):
    """添加hook来监控prototype_queries的梯度"""
    text_classifier = model.transformer.decoder.class_embed[0]
    if hasattr(text_classifier, 'tpa'):
        tpa = text_classifier.tpa
        prototype_queries = tpa.prototype_queries
        
        def grad_hook(grad):
            print(f"[Gradient Hook] grad_norm={grad.norm().item():.6f}")
            return grad
        
        prototype_queries.register_hook(grad_hook)
```

---

## 常见问题

### Q1: 梯度为零怎么办？

**可能的原因：**
1. `requires_grad=False` → 检查参数是否被冻结
2. 梯度被detach了 → 检查代码中是否有`.detach()`
3. 梯度被裁剪掉了 → 检查梯度裁剪设置
4. diversity loss没有正确计算 → 检查loss计算逻辑

**解决方法：**
1. 确认`prototype_queries.requires_grad = True`
2. 检查是否有`.detach()`调用
3. 增加`lambda_div`的值
4. 检查梯度裁剪的`max_norm`是否太小

### Q2: 梯度存在但prototypes仍然相同？

**可能的原因：**
1. 梯度太小，更新太慢
2. 学习率太小
3. 其他loss的梯度太大，覆盖了diversity loss

**解决方法：**
1. 增加`lambda_div`的值（已增加到0.30）
2. 检查学习率设置
3. 检查是否有梯度裁剪

### Q3: 如何确认prototypes是否在更新？

**检查方法：**
1. 保存两个checkpoint（间隔一定迭代数）
2. 比较`prototype_queries`的值是否变化
3. 如果值没有变化 → 说明没有更新

```python
# 比较两个checkpoint的prototype_queries
checkpoint1 = torch.load("checkpoint1.pth")
checkpoint2 = torch.load("checkpoint2.pth")

tpa1 = checkpoint1["model"]["transformer.decoder.class_embed.0.tpa.prototype_queries"]
tpa2 = checkpoint2["model"]["transformer.decoder.class_embed.0.tpa.prototype_queries"]

diff = (tpa1 - tpa2).abs().mean()
print(f"Prototype queries change: {diff:.6f}")
```

---

## 快速检查清单

在训练过程中，检查：

- [ ] `prototype_queries.requires_grad = True`
- [ ] 梯度存在且非零
- [ ] 梯度大小合理（不是太小）
- [ ] `loss_div`在下降
- [ ] `effective lambda_div > 0`
- [ ] prototypes之间的相似度在下降

如果所有项都正常，但`loss_div`仍然不下降，可能需要：
1. 进一步增加`lambda_div`
2. 检查是否有其他问题（如梯度裁剪）

