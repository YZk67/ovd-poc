# 8ep 辅助分类梯度：一次配对短程续训

从已完成的 8ep 分类梯度拆分报告出发，只检验一个候选：

| 项目 | A | B |
| --- | --- | --- |
| 起点 | 同一完整 `model_0056799.pth` | 同左 |
| 更新次数 | 500 个 optimizer updates | 同左 |
| 辅助层 `loss_class_0..4` → decoder_core | 保留 | 屏蔽直接梯度 |
| 辅助分类 → 其他参数 | 保留 | 保留 |
| 最终层分类、DN、框损失、APR、RPSA | 原生 | 原生 |
| LR / 全局有效 batch / 推理 | 原生 85200 horizon / 32 / 锁定协议 | 同左 |

`decoder_core` 沿用已审计的参数集合：decoder layers、norm、ref_point_head，
不包括类别头、框头、TPA、encoder 或 query embedding。没有替换旧 decoder 权重，
也没有把辅助分类 loss 的权重设为零。

## 服务器运行

同步提交后，在原来的 `lami` 环境运行。需要原审计目录及其缓存、完整 8ep checkpoint；
输入缺失、代码身份变化、AMP/优化器状态缺失均在训练前拒绝，不自动退化成仅加载权重。
输出目录必须是新的；日志写在它旁边，不能预先创建目录。

```bash
cd ~/LaMI-DETR
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup /root/miniconda3/envs/lami/bin/python -u \
  tools/run_decoder_aux_ablation.py \
  --classification-audit /root/autodl-tmp/no_radius_decoder_classification_sources/report.json \
  --output-dir /root/autodl-tmp/no_radius_aux_decoder_ab_500 \
  > /root/autodl-tmp/no_radius_aux_decoder_ab_500.log 2>&1 &
```

```bash
tail -n 40 -F /root/autodl-tmp/no_radius_aux_decoder_ab_500.log
```

顺序是 **训练 A → 训练 B → 完整 LVIS 评测 A → 完整 LVIS 评测 B**。
每路从 iteration56800 开始，到57299完成，保存各自完整 `model_final.pth`；
最后上传 `no_radius_aux_decoder_ab_500/summary.json`。
不要将这两个短程 checkpoint 写回正式训练目录。

`--prepare-only` 只检查 CPU 输入并生成 manifest，不训练。之后原指令增加
`--execute-prepared` 可执行；此标志也可复用校验通过的已完成训练/评测阶段，
**不能从中断到一半的 A/B 训练继续**，因为那会丢失配对的 RNG/loader 位置。
若训练中途失败，保留目录用于诊断，修复后用新目录重新做配对。
评测失败不会伪装成成功，也不会自动重复训练。
`--updates 1` 或其他小于500的值只供冒烟测试，不能替代正式500步结果。

## 配对和裁剪的含义

沿用原生 `Trainer.run_step`、DDP、AMP、两次微批累积、APR 冲突投影和优化器。
两路都恢复源 checkpoint 的模型、AdamW moments、scheduler、AMP scaler，
逐项计算 digest 核对恢复状态。LR 时间线仍是85200，没有新 warmup，也没有缩短到500。
四卡每卡每微批4图，全局物理 batch16，累积2次，有效 batch32。

两路都额外提取辅助分类的 decoder 梯度，按原微批缩放并跨 rank 平均。
先执行原生 APR routing，随后以各自当前**完整梯度**计算原生 detector/TPA
分别0.5的裁剪系数。A不再改动梯度；B仅减去 `clip_coefficient * auxiliary_gradient`
这一 decoder 分量。不会因为删掉该项而重新放大其他参数的梯度，也不做第二次裁剪。
由于梯度之间可能相互抵消，B删项后的范数可能超过0.5；这是明确记录的对照定义，
不是声称B仍满足原总范数上限。两条轨迹分开后，其完整梯度和裁剪系数本身可以不同。
各步信息保存在 `A|B/updates_rank*.jsonl`。

为了配对增强、DN/dropout及FedLoss，A/B都使用 `num_workers=0`，分别固定每微批
数据读取和前向 RNG，并对图像、框、标签、FedLoss集合、前向后RNG、LR、AMP scale
逐批核验。第一步两路所有 loss 也核对一致，后续不要求 loss 相等。
若 AMP 跳过实际 optimizer.step，或出现非有限梯度，会拒绝认作完成500次更新。

这是一组**新配对的续训轨迹**，不是历史8→10ep的数据顺序精确重放。
CUDA自定义算子可能有浮点非确定性；配对输入/RNG不等于保证最终权重逐位一致。
B也不会擦除源 AdamW moments 中的历史辅助分类作用或移除 weight decay。
其他参数此后会因 decoder 改变而间接适配，所以不能承诺500步后它们仍与A相同。

## 如何解读

只比较这次正式完整LVIS的 **B−A**，同时看 APr、AP、APc/APf；不能拿B直接减去
历史10ep数字，不能用三个事后挑选的类别替代完整指标。
若B没有优势，停止这项干预，不自动扫步数/权重；若有优势，也仅支持这个起点、
短程窗口和种子下的干预，尚不能证明它解释全部旧新差距或长期训练瓶颈。

本地 CPU 测试（包括真实双进程 Gloo 梯度同步；需要回环通信权限）：

```bash
cd tests
PYTHONPATH=.. python -m pytest --rootdir=. --confcutdir=. -q test_decoder_aux_ablation.py
```

CPU测试用原生 Trainer 的未修改方法验证梯度累积/裁剪和恢复机制；不替代服务器
PyTorch1.12/CUDA/detrex/NCCL的实际运行。正式评测仍使用原生 `train_net.py --eval-only`。
