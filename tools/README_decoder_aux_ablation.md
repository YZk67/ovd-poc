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

## 已完成 A/B 的 CPU 全验证集复盘

本次 APr：A=`43.7873`，B=`43.0763`，B−A=`−0.7110`。以下脚本只读此次
`summary.json` 对应的已保存预测，不加载 checkpoint，不重训、不运行模型或 GPU。
不把前期三个局部类别直接当成整体结论。

```bash
cd ~/LaMI-DETR
set -o pipefail
/root/miniconda3/envs/lami/bin/python -u tools/review_decoder_aux_ablation.py \
  --summary /root/autodl-tmp/no_radius_aux_decoder_ab_500/summary.json \
  --output-dir /root/autodl-tmp/no_radius_aux_decoder_ab_500_review \
  2>&1 | tee /root/autodl-tmp/no_radius_aux_decoder_ab_500_review.log
```

输出目录必须不存在。脚本从原 manifest/分类审计记录自动找到原验证集标注，
核对 A/B 预测的 SHA256 和官方 APr，使用 LVIS 的 ignore/一对一匹配规则。
**全类别候选先执行原始每图 top-300，再只统计 rare 类**；不根据 rare GT 筛图。

两路各进行一次 CPU 官方评估，同时保存所有 rare 类的全部10个 IoU 阈值 PR，
不会为不同类别或 IoU 再次评估。相较于只保留 .50/.75 曲线，会增加 CPU 后处理时间
和报告体积；原来约1GB的预测文件不复制。首次官方统计可能较慢，不能根据短期
没有新日志就判定卡住。总日志以及输出目录的 `A.log` / `B.log` 可查看阶段。

最终上传 **`no_radius_aux_decoder_ab_500_review/report.json`**。包含：

- 全部有效 rare 类逐类 AP/AP50/AP75 差值及宏平均贡献，含获益和受损两侧。
- GT数量1、2–4、5–9、10–19、20+分层；主结果始终按全部有效类等权。
- 原先关注的 `bass_horn`、`keg`、`lasagna` 各自完整AP，以及三类与其余类的贡献。
- 全部10个 IoU 上的召回变化、同等召回处 FP-before-TP 变化及完整AP分解；
  `lost_recall_support + gained_recall_support + shared_recall_precision` 核对到正式 ΔAPr。
- 每个GT分层分别选择获益、受损、不变的示例类别；这些是解释用的事后示例，
  不作为独立验证集或调参集合。全部类别都已经分析，示例不会改变统计结果。

这里的“共同召回区域 precision 变化”不是纯FP因果贡献：它可以来自 FP 提前或
TP后移/替换；同序号TP不等于同一GT。保存的 top-300 召回下降也不能直接解释成
没有合格原始 proposal。报告不自动推荐新 loss、屏蔽比例或新的训练。

`--prepare-only` 仅预检输入；随后增加 `--resume` 执行。`--resume` 也可复用哈希校验
通过的已完成CPU报告。中途失败的未认证报告不覆盖；不回退到GPU重导预测。

## 完成后的零 GPU 全 rare 类分解

以下是已有的 GT 转移/梯度日志扩展入口。本次复盘只运行上面的
`review_decoder_aux_ablation.py` 即可，不必两个入口都跑；上面的入口补齐全部10个
IoU 的 APr 分解和原先关注类别与其余类别的对照，不需要梯度日志。

A/B 的两次完整评测完成后，可用保存的预测和每步梯度日志解释结果。该步骤会重新做
两次 CPU LVIS 累积及 rare-GT 图像的官方一对一匹配，通常需要一段时间，但不加载
checkpoint、不训练，也不启动 GPU。输出目录同样必须不存在：

```bash
cd ~/LaMI-DETR
CUDA_VISIBLE_DEVICES='' nohup /root/miniconda3/envs/lami/bin/python -u \
  tools/analyze_decoder_aux_outcome.py \
  --trial-dir /root/autodl-tmp/no_radius_aux_decoder_ab_500 \
  --output-dir /root/autodl-tmp/no_radius_aux_decoder_ab_500_analysis \
  > /root/autodl-tmp/no_radius_aux_decoder_ab_500_analysis.log 2>&1 &
```

```bash
tail -n 60 -F /root/autodl-tmp/no_radius_aux_decoder_ab_500_analysis.log
```

结果保存在 `no_radius_aux_decoder_ab_500_analysis/report.json`。它包含全部有效 rare 类
的 `ΔAP/ΔAP50/ΔAP75`、GT数分层、IoU=.50/.75 的 recall/ranking AP 分解、A/B
正式GT命中转移以及全局梯度范数分布。GT漏失原因只在最终top-300内分为：有正确类
候选但未匹配；存在重叠框但真类候选缺席；最终top-300连重叠框也没有。

最终预测没有保存被top-300丢弃的全query，所以后两项不能继续解释成“proposal不存在”
或“正确pair只是在top-300以下”。每步梯度日志也没有类别维度，脚本不会伪造逐类
梯度与ΔAP的相关性。若要回答那两个问题，必须另有训练前锁定的全query缓存。
