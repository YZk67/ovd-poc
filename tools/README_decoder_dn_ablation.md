# DN 分类梯度到 decoder_core：单变量配对续训

这一步只隔离 `loss_class_dn` 与 `loss_class_dn_0..4` 对 `decoder_core` 的直接梯度。
最终层/辅助层分类、encoder 分类、L1/GIoU、APR、RPSA、LR 和推理协议全部保留。

为避免重复算力，A 直接复用已完成的
`no_radius_aux_decoder_ab_500/A`：它是从同一 8ep 完整 checkpoint 出发的原生
500-update 控制臂。准备阶段会重新验证原 A 的 manifest、checkpoint、正式预测、
四卡 transcript 和代码哈希。新 B 使用相同 seed 公式，并逐微批核对原 A 的数据、
增强、FedLoss、前向 RNG、LR、AMP scale；首步还核对所有 loss。

因此本次只新增一次 500-update 四卡训练和一次四卡完整 LVIS 评测，不会再次训练或
评测 A。输出目录必须不存在，日志放在目录外面：

```bash
cd ~/LaMI-DETR

CUDA_VISIBLE_DEVICES=0,1,2,3 nohup /root/miniconda3/envs/lami/bin/python -u \
  tools/run_decoder_dn_ablation.py \
  --reference-trial /root/autodl-tmp/no_radius_aux_decoder_ab_500 \
  --output-dir /root/autodl-tmp/no_radius_dn_decoder_ab_500 \
  > /root/autodl-tmp/no_radius_dn_decoder_ab_500.log 2>&1 &

echo "PID=$!"
```

实时查看：

```bash
tail -n 50 -F /root/autodl-tmp/no_radius_dn_decoder_ab_500.log
```

完成后上传：

```text
/root/autodl-tmp/no_radius_dn_decoder_ab_500/summary.json
```

## 对照定义

两路的完整梯度都先经过原生 APR routing，并各自用完整 detector 梯度计算原生
0.5 裁剪系数。B 随后只减去：

```text
clip_coefficient × grad(loss_class_dn + loss_class_dn_0..4, decoder_core)
```

不重新裁剪，也不清除 8ep checkpoint 中的 AdamW moments 或 weight decay。
DN 分类对类别头及其他模块的梯度仍然存在；DN box/GIoU 也完全保留。因此正式
`B−A` 只回答“这 500 步中 DN 分类到 decoder_core 的直接梯度是否有净影响”。

若训练或评测中断，脚本不会从不完整 B 静默续跑；保留目录诊断后换一个新输出目录。
可先加 `--prepare-only` 做所有 CPU 输入检查，再用同一命令改为
`--execute-prepared` 开始正式运行。

本地回归：

```bash
cd tests
PYTHONPATH=.. python -m pytest --rootdir=. --confcutdir=. -q \
  test_decoder_dn_ablation.py test_decoder_aux_ablation.py
```
