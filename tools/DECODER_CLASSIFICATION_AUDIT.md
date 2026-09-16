# 8ep decoder 分类梯度细分（只读）

用途：在已完成的 `audit_decoder_loss_sources.py` 审计基础上，重放 **8ep** 的相同
训练数据窗口，进一步区分最终层、辅助层和 DN 分类梯度。不是训练，不执行优化器
step，不生成混合 checkpoint，不运行完整 LVIS AP，不改变 APR、LR 或推理协议。

## 运行

服务器需要保留上次审计整个目录（`report.json`、`old_capture.json`、
`new_capture.json`、`new/validation_*.json`、`old/training_window_*.pt`），以及原 query audit 的输入和缓存。
单独一份上传的报告无法重放原始样本。使用原来的 `lami` Python 环境。

```bash
cd ~/LaMI-DETR
set -o pipefail

CUDA_VISIBLE_DEVICES=0 /root/miniconda3/envs/lami/bin/python -u \
  tools/audit_decoder_classification_sources.py \
  --parent-audit /root/autodl-tmp/no_radius_decoder_loss_sources/report.json \
  --output-dir /root/autodl-tmp/no_radius_decoder_classification_sources \
  2>&1 | tee /root/autodl-tmp/no_radius_decoder_classification_sources.log
```

`--prepare-only` 只检查来源、预算和输出目录，不启动 GPU 前向。
原命令重跑可复用已完成的窗口/图像；未完成的窗口会重算。
`--analyze-only` 只分析已完成的细分采集，不会自动退回 GPU 执行。
日志放在输出目录**外**，不要把源审计目录用作新输出目录。

## 范围与校验

- 预算、种子、控制类别、图像选择继承父报告，不提供重新挑选类别的参数。
  默认父审计对应 4 窗口 × 8 microbatch × 4 图像 = **128 次训练图像输入**，
  以及 13 次验证前向；硬上限是 128 / 16。10ep 不重新前向。
- 单 GPU、串行 microbatch。更多分支反向意味着运行时间不一定是原审计的一半。
  细分梯度缓存约 0.9 GB（当前 decoder 大小和默认预算）。
- 使用 `train_norare` 原生加权损失、相同图像/增强/FedLoss/前向随机种子，
  分类总梯度以及 L1/GIoU/APR/RPSA 与原 8ep 报告作容差内复现校验。
- 分类分组：
  - `class_final`：`loss_class`；
  - `class_aux`：`loss_class_0` 等数字后缀；
  - `class_dn`：`loss_class_dn` 和 `loss_class_dn_0` 等；
  - `class_encoder`：`loss_class_enc`，单独检查、不能计入 decoder 辅助层。
- 保留独立求导的 `classification` 总量，验证各分类分支梯度之和等于总梯度。
  encoder 分类必须没有到 decoder_core 的直接路径，否则停止。
- 原控制区域/标签不变；没有完整 TP/FP 的类别保持 NA，不补选新类别。
  保留 L1/GIoU 对照；所有参数、持久 buffer 和 `.grad` 都必须保持不变。

## 输出解释

上传新目录的 `report.json`。报告包含分支 loss key、梯度范数、复现校验、
TP/FP 局部方向导数、每个类别的逐窗口符号及控制样本覆盖。

负 `TP−FP` 导数表示沿该负梯度方向在当前固定候选、原生 stop-gradient 边界下
局部不利。不是 AdamW 实际更新，不包含框重选、CLIP ROI 或全图 top-300 变化，
不预测 APr，不证明历史因果关系或全局瓶颈。

**原始方向导数可以相加，按各自范数归一化后的方向导数不能直接相加。**
`classification` 与它的子项重叠，不能再次把总量加进子项和。
四个窗口的负号次数不是显著性检验；当前工具不自动选择、删除损失或启动续训。
