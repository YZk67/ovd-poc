# 从完整 no-radius 10ep 审计实际优化器更新

前置结果：scarecrow GT137708 / image218917 在 10ep 仍有正确类别候选进入
top-300，12ep 不再进入。这里只检验 **10ep 状态下局部更新的作用**，不把
新抽取的 batch 当作历史 10→12ep 的精确重放，不直接解释整个 APr 差距。

## 一次运行

服务器拉取包含本工具的提交后，用原来的 lami 环境和四卡拓扑：

```bash
cd ~/LaMI-DETR
set -o pipefail

CUDA_VISIBLE_DEVICES=0,1,2,3 /root/miniconda3/envs/lami/bin/python -u \
  tools/audit_late_optimizer_updates.py \
  --timeline /root/autodl-tmp/no_radius_scarecrow_timeline/report.json \
  --output-dir /root/autodl-tmp/no_radius_10ep_actual_updates \
  --windows 2 \
  2>&1 | tee /root/autodl-tmp/no_radius_10ep_actual_updates.log
```

日志：

```bash
tail -n 60 -F /root/autodl-tmp/no_radius_10ep_actual_updates.log
```

完成后上传 `/root/autodl-tmp/no_radius_10ep_actual_updates/report.json`。
`complete` 必须为 `true`。不能把失败时留下的部分报告当作完整诊断。

## 预算和状态保证

- 默认 2 个**独立**有效 batch：4 卡 × 每卡 4 张 × 累积 2 次，共 64 次训练图像曝光。
  `--windows` 硬上限 4。不是 2 或 4 步顺序续训，每个窗口都重置到完整 iteration70999。
- 捕获四类**已加权**梯度：全部分类（含 final/aux/DN/encoder）、全部 L1/GIoU、APR、RPSA。
  使用原训练 forward、AMP scaling/unscale、FedLoss 和同步有效 batch。
- 默认 12 次真实但可恢复的内存中 AdamW 单步；不写训练 checkpoint，不动 `last_checkpoint`。
  独立分支为 native、minus_classification、minus_box、minus_apr、minus_rpsa、history_only。
  后者把本次梯度置零，但仍保留 AdamW 历史动量和权重衰减；不等同于不更新。
- 默认最多 15 次单图评测 forward：1 次未经更新的原生复现，2 次 buffer-only 对照，12 次更新后评测。
  梯度捕获完成、四卡进程退出后，评测只使用可见 GPU0，不做完整验证集推理/正式 AP。
- 完整读取模型、AdamW moments、各参数组 LR、原始 85,200 scheduler horizon、累积次数和 AMP scaler。
  只接受 iteration70999 / scheduler last_epoch71000，拒绝 weights-only、LR 跳变或错误端点。
- 每个分支重新加载原始 optimizer 和新的同状态 scaler；按原生 Trainer 顺序 unscale、
  APR 冲突投影、分别 0.5 L2 clipping、scaler.step/update。原生单步的 CPU toy 对照测试
  与实际 `Trainer.run_step` 的参数、moment 和 scaler 结果相同。
- 移除源梯度覆盖所有可训练 optimizer 参数，重算**全部**裁剪系数和路由。因此变化包括这些
  中介效应，不应当作可相加的 loss 百分比贡献。保留原来的 None/zero 梯度语义。
- 单图评测重新计算全部 900 query、框、CLIP ROI 和 1203 类分数及 top-300。query ID 不跨分支硬配对。
- 原始完整 10ep forward 必须复现缓存的全部 logits/scores、900 个框及 300 个输出后才允许临时更新。
  训练 forward 产生的 buffer 改变对所有分支保持一致，并有只改变 buffer、不更新参数的独立对照。
- 正常完成或评测异常都恢复内存中的参数、持久/非持久 buffer、optimizer；完成后重新验证源 checkpoint 哈希。
  历史样本次序/RNG 未保存，手工同步也不能承诺与原 DDP bucket 浮点舍入逐位一致。

## 分阶段与失败恢复

同一命令加 `--prepare-only`：只查 CPU 文件身份、完整状态和配置资产，不做 forward。

同一命令加 `--phase capture`：只捕获梯度，不执行 optimizer.step。
已完成的 `window_XX.pt` 与对应带哈希的 JSON 可以复用；文件不完整或身份不符时拒绝混用。
梯度缓存可能占数 GiB，脚本事前检查空间，不自动删除任何文件。

捕获已完成、评测失败时，将相同命令改成 `--phase evaluate` 即可只重做评测阶段，
可设 `CUDA_VISIBLE_DEVICES=0`。不要将 `--num-gpus` 改为 1，它记录的是原始**捕获**拓扑。
评测重试会重新执行该阶段的临时单步；预算按单次成功调用计算，不声称异常重试零成本。
输出目录的 manifest 锁定输入和代码；若修复代码改变 manifest，需新目录，不能跳过校验。

## 如何解释

主指标预先固定为 IoU≥0.5 合格框中真类最高分的 `log(score / image_top300_cutoff)`；
同时保留 IoU 0.5/0.75/0.9 的框覆盖、真类排名、CLIP 分数及 top-300。

1. 先看 native 相对 buffer-only 是否真的降低该指标（数值容差 1e-4）。
2. native 没有复现退化，就不能把某个反事实分支更高解释成找到了历史退化原因。
3. 仅当至少两个窗口都复现退化，且移除同一个源都改善 native，才记作 `LOCAL_CANDIDATE_ONLY`。
   这不是历史归因、不是全局 AP 证据，也不会自动建议或执行下一次训练。
4. 不能仅看相对 native 回升：还需看是否回到更新前水平、框/阈值是否变化，以及是否主要来自
   裁剪系数或继承的 optimizer history。报告保留所有分支，不筛掉不支持猜想的窗口。
5. 两个新 batch、一次单图探针不足以证明整个 71000–85199 区间由某个 loss、APR 或 LR 导致。
   若返回 `NO_CONSISTENT_LOCAL_SOURCE`，本次假设不获支持，不自动加训练预算。

本地测试不能替代服务器 PyTorch1.12 / CUDA / detrex / 四卡 NCCL 集成验证；
首次运行仍须关注 native 复现、AMP 梯度重构和完整状态校验。
