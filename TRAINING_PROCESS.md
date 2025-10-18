# LaMI-DETR 训练流程详解

## 📋 目录
1. [训练启动流程](#1-训练启动流程)
2. [数据加载流程](#2-数据加载流程)
3. [模型前向传播](#3-模型前向传播)
4. [损失计算与反向传播](#4-损失计算与反向传播)
5. [训练循环与Hook](#5-训练循环与hook)
6. [评估与checkpoint](#6-评估与checkpoint)

---

## 1. 训练启动流程

### 启动命令
```bash
python tools/train_net.py \
  --config-file lami_dino/configs/dino_convnext_large_4scale_12ep_lvis.py \
  --num-gpus 8 \
  train.init_checkpoint=pretrained_models/lami_convnext_large_12ep_lvis/model_final.pth
```

### 代码执行流程

#### 1.1 `main()` 函数 (train_net.py:213-240)
```python
def main(args):
    # 1. 加载配置文件
    cfg = LazyConfig.load(args.config_file)  # 加载dino_convnext_large_4scale_12ep_lvis.py
    cfg = LazyConfig.apply_overrides(cfg, args.opts)  # 应用命令行参数覆盖
    
    # 2. 处理debug模式
    if args.ddebug:
        cfg.train.max_iter = 8
        cfg.train.eval_period = 8
        ...
    
    # 3. 初始化日志和设备
    default_setup(cfg, args)
    
    # 4. 选择模式
    if args.eval_only:
        # 评估模式
        model = instantiate(cfg.model)
        model.to(cfg.train.device)
        model = create_ddp_model(model)
        DetectionCheckpointer(model).load(cfg.train.init_checkpoint)
        print(do_test(cfg, model))
    else:
        # 训练模式
        do_train(args, cfg)
```

#### 1.2 多GPU启动 (train_net.py:242-251)
```python
if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    launch(
        main,
        args.num_gpus,          # 8 GPUs
        num_machines=1,
        machine_rank=0,
        dist_url="auto",
        args=(args,),
    )
```
- 使用PyTorch的`torch.distributed`启动多进程训练
- 每个GPU运行一个进程

---

## 2. 数据加载流程

### 2.1 训练数据 (configs/common/data/lvis_detr.py)

```python
dataloader.train = build_detection_train_loader(
    # 数据集配置
    dataset=get_detection_dataset_dicts(names="lvis_v1_train_norare"),
    
    # 采样策略 - Repeat Factor Sampling
    sampler="RepeatFactorTrainingSampler",
    repeat_threshold=0.001,  # 重复采样稀有类别
    
    # 数据增强
    mapper=DetrDatasetMapper(
        augmentation=[
            RandomFlip(),
            ResizeShortestEdge(
                short_edge_length=(480, 512, ..., 800),
                max_size=1333,
                sample_style="choice",  # 随机选择
            ),
        ],
        augmentation_with_crop=[...],  # 带裁剪的增强
        is_train=True,
        mask_on=False,
        img_format="RGB",
    ),
    
    total_batch_size=32,  # 总batch size = 8 GPUs × 4 images/GPU
    num_workers=8,
)
```

### 2.2 数据集详情
- **数据集**: LVIS v1.0 (train, no rare categories)
  - 训练图片: 100,170 images
  - 类别数: 1203 classes
  - 去除rare categories的图片

- **Repeat Factor Sampling**:
  - 根据类别频率重复采样稀有类别图片
  - 缓解长尾分布问题

### 2.3 数据增强流程
```
原始图片 (任意尺寸)
    ↓
[RandomFlip] 50%概率水平翻转
    ↓
[ResizeShortestEdge] 随机resize到480-800之间
    ↓
[可选: RandomCrop] 随机裁剪到384×600
    ↓
增强后的图片 + 对应的bbox标注
```

---

## 3. 模型前向传播

### 3.1 模型初始化 (do_train: train_net.py:160-170)

```python
def do_train(args, cfg):
    # 1. 实例化模型
    model = instantiate(cfg.model)  # DINO模型
    model.to(cfg.train.device)      # 移到GPU
    
    # 2. 创建优化器
    cfg.optimizer.params.model = model
    optim = instantiate(cfg.optimizer)  # AdamW
    
    # 3. 创建数据加载器
    train_loader = instantiate(cfg.dataloader.train)
    
    # 4. 包装为DDP模型 (多GPU训练)
    model = create_ddp_model(model, **cfg.train.ddp)
```

### 3.2 DINO模型结构

```
输入图片 (batch_size, 3, H, W)
    ↓
┌─────────────────────────────────────────┐
│ 1. Backbone (ConvNeXt-Large, frozen)   │
│    - 提取多尺度特征                      │
│    - 输出: p1, p2, p3 (3个不同scale)     │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│ 2. Neck (ChannelMapper)                │
│    - 统一channel维度到256               │
│    - 输出: 4个scale的特征                │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│ 3. Position Embedding                  │
│    - 为每个scale添加位置编码             │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│ 4. Transformer                         │
│    ├─ Encoder (6 layers)               │
│    │   - 处理多尺度特征                  │
│    │   - 输出编码后的特征                │
│    │                                    │
│    └─ Decoder (6 layers)               │
│        - Query: Content Query (从文本)  │
│        - 输出: 预测的object embeddings  │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│ 5. Prediction Heads                   │
│    ├─ Class Embed (TextClassifier)     │
│    │   - 使用TPA处理文本prototypes       │
│    │   - 输出: (bs, num_queries, 1203)  │
│    │                                    │
│    └─ BBox Embed (MLP)                 │
│        - 输出: (bs, num_queries, 4)     │
└─────────────────────────────────────────┘
    ↓
输出: pred_logits, pred_boxes
```

### 3.3 关键组件详解

#### A. Content Query生成 (dino.py:304-310)
```python
if self.training:
    # 训练时: 使用Fed Loss筛选的类别
    content_query_embeds = self.content_query_embedding[content_inds]
    content_query_embeds = self.content_layer(content_query_embeds)
    content_query_embeds = F.normalize(content_query_embeds, p=2, dim=1)
else:
    # 推理时: 使用全部1203个类别
    content_query_embeds = self.content_layer(self.eval_content_query_embedding)
    content_query_embeds = F.normalize(content_query_embeds, p=2, dim=1)
```

**Content Query来源**:
- 从 `lvis_tpa_prompts_convnextl.npy` (1203, 5, 768) 加载
- 每个类别有5个text prompt的embeddings
- 通过linear layer投影到transformer维度
- L2归一化

#### B. Text Classifier with TPA (text_classifier.py)
```python
class TextClassifier:
    def __init__(self, ..., use_tpa=True, text_embed_path, ...):
        # 加载text embeddings (1203, 5, 768)
        self.train_text_feats = load_text_embeddings(text_embed_path)
        
        # 初始化Text Prototype Aggregator
        if use_tpa:
            self.tpa = TextPrototypeAggregator(
                dim=768,
                num_prototypes=4,  # 每个类学习4个prototypes
                hidden_dim=256,
            )
    
    def forward(self, x, content_inds):
        # 1. 映射visual feature到text空间
        x = self.linear(x)  # (bs*num_queries, 768)
        x = F.normalize(x, p=2, dim=-1)
        
        # 2. 获取text prototypes
        if self.use_tpa:
            # 从5个prompts学习4个prototypes
            text_feats = self.tpa(self.train_text_feats)  # (1203, 4, 768)
            # 平均为单个prototype
            classifier = text_feats.mean(dim=1)  # (1203, 768)
        else:
            classifier = self.train_text_feats.mean(dim=1)
        
        # 3. 计算余弦相似度
        logits = x @ classifier.T  # (bs*num_queries, 1203)
        logits = logits * self.norm_temperature  # scale by temperature
        
        return logits
```

#### C. Denoising Training (CDN)
```python
# 准备denoising queries (dino.py:314-328)
if self.training:
    gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
    targets = self.prepare_targets(gt_instances)
    
    # Contrastive Denoising (CDN)
    input_query_label, input_query_bbox, attn_mask, dn_meta = \
        self.prepare_for_cdn(
            targets,
            dn_number=100,           # 每个GT生成100个noisy queries
            label_noise_ratio=0.5,   # 50%标签noise
            box_noise_scale=1.0,     # box noise scale
            num_queries=900,         # 总query数
            num_classes=100,         # Fed Loss: 每次采样100个类
            content_query_embeds=content_query_embeds,
        )
```

---

## 4. 损失计算与反向传播

### 4.1 训练一步 (Trainer.run_step: train_net.py:76-121)

```python
def run_step(self):
    # 1. 获取一个batch数据
    data = next(self._data_loader_iter)
    
    # 2. 前向传播
    loss_dict = self.model(data)  # 调用DINO.forward()
    losses = sum(loss_dict.values())
    
    # 3. 反向传播
    self.optimizer.zero_grad()
    
    if self.amp:  # 混合精度训练
        self.grad_scaler.scale(losses).backward()
        if self.clip_grad_params:
            self.grad_scaler.unscale_(self.optimizer)
            self.clip_grads(self.model.parameters())
        self.grad_scaler.step(self.optimizer)
        self.grad_scaler.update()
    else:
        losses.backward()
        if self.clip_grad_params:
            # Gradient Clipping: max_norm=0.1
            self.clip_grads(self.model.parameters())
        self.optimizer.step()
    
    # 4. 记录metrics
    self._write_metrics(loss_dict, data_time)
```

### 4.2 损失函数 (dino.py:393-399)

```python
if self.training:
    # 计算损失
    loss_dict = self.criterion(output, targets, dn_meta)
    
    # 应用权重
    weight_dict = self.criterion.weight_dict
    for k in loss_dict.keys():
        if k in weight_dict:
            loss_dict[k] *= weight_dict[k]
    
    return loss_dict
```

### 4.3 损失类型

**主要损失** (SetCriterion):
1. **Classification Loss** (Focal Loss)
   - 类别预测损失
   - 权重: 2.0
   
2. **L1 Loss** 
   - Box坐标回归损失
   - 权重: 5.0

3. **GIoU Loss**
   - Box IoU损失
   - 权重: 2.0

**辅助损失** (Auxiliary Outputs):
- 每个decoder layer都输出预测
- 共6层 × 3种损失 = 18个辅助损失

**Denoising Loss** (CDN):
- Contrastive denoising queries的损失
- 帮助模型区分正负样本

**总损失**:
```
total_loss = Σ(weight_dict[k] * loss_dict[k])
```

### 4.4 Fed Loss (Federated Loss)

```python
# 动态采样类别 (dino.py:281-282)
if self.use_fed_loss:
    content_inds, batched_inputs = self.filter_content_info(batched_inputs)

def filter_content_info(self, batched_inputs):
    # 1. 收集当前batch中的GT类别
    gt_classes_in_batch = ...
    
    # 2. Cluster-based采样
    if self.cluster_fed_loss:
        # 从128个clusters中采样
        content_inds = get_cluster_fed_loss_inds(
            gt_classes_in_batch,
            self.cluster_label,
            num_sample_cats=100,  # 采样100个类
        )
    else:
        # 基于频率采样
        content_inds = get_fed_loss_inds(
            gt_classes_in_batch,
            num_sample_cats=100,
            freq_weight=self.freq_weight,
        )
    
    return content_inds, batched_inputs
```

**Fed Loss的作用**:
- 每个batch只计算100个类别的损失
- 包含GT类别 + 采样的负类别
- 减少计算量，平衡类别分布

---

## 5. 训练循环与Hook

### 5.1 训练循环 (do_train: train_net.py:186-210)

```python
# 1. 注册Hooks
trainer.register_hooks([
    # 计时器
    hooks.IterationTimer(),
    
    # 学习率调度
    hooks.LRScheduler(scheduler=instantiate(cfg.lr_multiplier)),
    
    # 定期保存checkpoint
    hooks.PeriodicCheckpointer(
        checkpointer, 
        period=3130,  # 每3130 iter (1 epoch)
        max_to_keep=100,
    ),
    
    # 定期评估
    hooks.EvalHook(
        eval_period=3130,  # 每1 epoch评估
        eval_function=lambda: do_test(cfg, model)
    ),
    
    # 日志记录
    hooks.PeriodicWriter(
        default_writers(output_dir, max_iter),
        period=20,  # 每20 iter记录一次
    ),
])

# 2. 加载checkpoint
checkpointer.resume_or_load(cfg.train.init_checkpoint, resume=args.resume)

# 3. 开始训练
if args.resume and checkpointer.has_checkpoint():
    start_iter = trainer.iter + 1
else:
    start_iter = 0

trainer.train(start_iter, cfg.train.max_iter)  # 训练37560 iterations
```

### 5.2 学习率调度 (configs/common/lvis_schedule.py)

```python
lr_multiplier_12ep_warmup = L(WarmupParamScheduler)(
    scheduler=L(MultiStepParamScheduler)(
        values=[1.0, 0.1],
        milestones=[30000],  # 在30000 iter降低lr
    ),
    warmup_length=250 / 37560,  # 前250 iter warmup
    warmup_method="linear",
    warmup_factor=0.001,
)
```

**学习率变化**:
```
Iter 0-250:     线性warmup from 1e-7 to 1e-4
Iter 250-30000: 保持1e-4
Iter 30000+:    降到1e-5
```

---

## 6. 评估与Checkpoint

### 6.1 评估流程 (do_test: train_net.py:132-138)

```python
def do_test(cfg, model):
    # 1. 推理所有测试图片
    results = inference_on_dataset(
        model, 
        instantiate(cfg.dataloader.test),      # LVIS minival
        instantiate(cfg.dataloader.evaluator), # LVISEvaluator
    )
    
    # 2. 打印结果
    print_csv_format(results)
    
    return results
```

### 6.2 评估指标 (LVISEvaluator)

**主要指标**:
- **AP**: Average Precision (overall)
- **APr**: AP for rare categories
- **APc**: AP for common categories  
- **APf**: AP for frequent categories

**按IoU阈值**:
- AP@0.5, AP@0.75

**按物体大小**:
- APs (small), APm (medium), APl (large)

### 6.3 Checkpoint保存

**保存内容**:
```python
checkpoint = {
    'model': model.state_dict(),
    'optimizer': optimizer.state_dict(),
    'scheduler': scheduler.state_dict(),
    'iteration': current_iter,
}
```

**保存位置**:
```
/root/autodl-tmp/lami_convnext_large_12ep_lvis_20251017_093045/
├── model_0003130.pth  # Epoch 1
├── model_0006260.pth  # Epoch 2
├── ...
├── model_0037560.pth  # Epoch 12 (final)
└── model_final.pth    # 链接到最后一个checkpoint
```

---

## 7. 完整训练时间线

```
开始训练
    ↓
Iter 0-250 (Warmup)
    - LR: 1e-7 → 1e-4 (线性增加)
    ↓
Iter 250-3130 (Epoch 1)
    - LR: 1e-4
    - 每20 iter记录loss
    ↓
Iter 3130 (Epoch 1结束)
    - 保存checkpoint
    - 运行评估 (LVIS minival)
    - 记录AP, APr等指标
    ↓
Iter 3131-6260 (Epoch 2)
    - LR: 1e-4
    ↓
...重复...
    ↓
Iter 30000-37560 (Epoch 10-12)
    - LR: 1e-5 (降低10倍)
    ↓
Iter 37560 (训练结束)
    - 保存最终model_final.pth
    - 运行最终评估
    - 输出最终结果
```

---

## 8. 关键配置总结

| 配置项 | 值 | 说明 |
|--------|-----|------|
| **数据** | | |
| Dataset | LVIS v1.0 train (no rare) | 100,170 images |
| Batch Size | 32 (8 GPUs × 4) | 总batch size |
| Num Workers | 8 | 数据加载线程 |
| | | |
| **训练** | | |
| Max Iterations | 37,560 | 12 epochs |
| Eval Period | 3,130 | 每epoch评估 |
| Checkpoint Period | 3,130 | 每epoch保存 |
| Log Period | 20 | 每20 iter记录 |
| | | |
| **优化器** | | |
| Optimizer | AdamW | |
| Base LR | 1e-4 | |
| Backbone LR | 1e-5 (×0.1) | Frozen backbone |
| Weight Decay | 1e-4 | |
| Warmup Iters | 250 | Linear warmup |
| LR Drop | Iter 30000 | ×0.1 |
| | | |
| **模型** | | |
| Backbone | ConvNeXt-Large | Frozen |
| Num Classes | 1203 | LVIS classes |
| Num Queries | 900 | |
| Transformer Layers | 6 encoder + 6 decoder | |
| | | |
| **损失** | | |
| Fed Loss | ✓ | 采样100类/batch |
| Cluster Fed Loss | ✓ | 128 clusters |
| Use TPA | ✓ | 4 prototypes/class |
| DN Number | 100 | Denoising queries |
| | | |
| **其他** | | |
| Gradient Clipping | max_norm=0.1 | |
| AMP | ✗ | 不使用混合精度 |
| Score Ensemble | ✓ | 使用CLIP head |

---

## 9. 预期结果

根据论文，在LVIS v1.0 validation set上:

| Metric | 预期值 | 说明 |
|--------|--------|------|
| AP | ~41.6 | Overall AP |
| APr | ~43.3 | Rare categories AP |
| APc | - | Common categories AP |
| APf | - | Frequent categories AP |

训练大约需要：
- **时间**: ~12-16小时 (8× A100 GPUs)
- **显存**: ~24GB per GPU
- **磁盘**: ~5GB (checkpoints)

---

## 10. 故障排查

### 常见问题

1. **OOM (Out of Memory)**
   ```python
   # 解决方案：减小batch size
   dataloader.train.total_batch_size = 16  # 从32降到16
   ```

2. **NaN Loss**
   ```python
   # 检查：
   # - 学习率是否过大
   # - Gradient clipping是否启用
   # - 数据增强是否正确
   ```

3. **AP不提升**
   ```python
   # 检查：
   # - Text embeddings是否正确加载
   # - Fed Loss是否正常工作
   # - Content queries是否正确
   ```

---

## 总结

LaMI-DETR的训练流程整合了多个先进技术：
1. **DINO**: 高性能的DETR-based检测器
2. **CLIP**: Vision-Language对齐的预训练backbone
3. **TPA**: 学习多个语义prototypes
4. **Fed Loss**: 处理长尾分布
5. **CDN**: Contrastive denoising提升训练效率

整个流程经过精心设计，确保在LVIS这样的长尾open-vocabulary数据集上达到SOTA性能。

