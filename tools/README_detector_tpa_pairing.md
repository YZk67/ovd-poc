# Detector–TPA 末端分类配对诊断

只做诊断，不训练、不修改 checkpoint、不写回模型配置。目的：区分末端
prototype 表示与投影后的 detector query 特征是否匹配；不直接把差异归因于
APR、梯度投影、初始化或训练长度。

## 对照与范围

| 名称 | query 特征、预测框、上游 query fusion、CLIP 分数 | 末端 prototype |
|---|---|---|
| `old_old` | 旧 checkpoint | 旧 checkpoint |
| `old_new` | 旧 checkpoint | 新 checkpoint |
| `new_new` | 新 checkpoint | 新 checkpoint |
| `new_old` | 新 checkpoint | 旧 checkpoint |
| `new_mean` | 新 checkpoint | 新 prototype 的逐类原始均值，再归一化 |

`new_mean` **不等于** `model.tpa_eval_mode_scale=0`：不重新运行 query fusion，
也不改变框。特征是最后分类层 `linear` 的输出，不是 CLIP ROI 特征。
分类 bias 保留特征来源模型的值，不随 prototype 交换；实际值记入报告。

默认使用当前推理协议：`alpha=0, beta=0.3, novel_scale=3`，TPA 温度
`0.004375`，calibrated Eq.2 温度 `0.07`，无 `log(5)` 偏置，category top-k=3，
最终全图 top-300，无 per-query 类别上限。旧模型缺少的结构 buffer 初始化为
0，防止当前配置凭空给旧模型增加模式；新模型的结构 buffer 由 checkpoint 恢复。

保留验证集中所有含 rare GT 的图（当前数据为 322 张）；另外固定 seed=42，
从“没有 rare GT 且至少一个 rare 类被明确标注为不存在”的图片中均匀抽取
最多 128 张负对照。选择不使用预测或类别 AP；不会只挑最差类别。
一个负对照**不代表所有 rare 类均已确认不存在**，匹配交给官方 LVIS 忽略规则。

对每个 GT / IoU 阈值分别报告：

- 合格框：两边都有、仅旧有、仅新有、都没有；另存逐类分解，未匹配类别不消失。
- 原生最高真类融合分数 query：在同一模型内固定，替换 prototype 时不重选。
- 最高 IoU query：独立的、按几何选取的辅助对照，避免只看分数选出的 query。
- 每个 variant 自己最高真类融合分数的 oracle query：单列，不冒充可部署召回。
- 真类 rank、真类 logit、真类与最高错误类 logit 差、CLIP rank/概率、全图
  top-300 阈值比、正确类别候选是否保留。

跨模型按 annotation ID 配对，**不按 query 编号配对**。主要分类宏平均仅在
两边都有合格框的共同 GT 集合内按类别等权计算，必须同时看 `paired_classes`
与定位排除统计；它可能少于验证集全部 178 个有效 rare 类。

FP/PR：五组均先从完整词表中选全图 top-300，再按原生后处理去掉空框、不补位，
最后使用官方 LVIS 对所选图片、rare 类、IoU 0.50/0.75 做一对一匹配和 ignore。
包含无正 GT 类的有效 FP，输出每类原始分数排序的 TP/FP/PR 曲线。
报告是 **panel 诊断，不是完整验证集 APr**；负对照仅抽样，不能由净 TP 或
分类 margin 改善直接推断正式 AP 一定提高。两组交叉组合都变差仅说明共同适配。

## 服务器运行

先把新增代码提交并推送到服务器使用的分支，再在服务器拉取。使用显式的
`lami` Python，避免 `nohup` 使用 base 环境。只需要一张 GPU，旧、新模型顺序
运行，每个模型对每张图只做一次 native forward。

```bash
cd ~/LaMI-DETR
mkdir -p /root/autodl-tmp/detector_tpa_pairing

CUDA_VISIBLE_DEVICES=0 /root/miniconda3/envs/lami/bin/python -u \
  tools/diagnose_detector_tpa_pairing.py \
  --old-checkpoint /root/autodl-tmp/model_final_ovd_lvis_kang.pth \
  --new-checkpoint /root/autodl-tmp/instructdet_k5_effective_bs32_4ep_seed42/model_final.pth \
  --output /root/autodl-tmp/detector_tpa_pairing/report.json \
  2>&1 | tee /root/autodl-tmp/detector_tpa_pairing/run.log
```

单独窗口实时查看：

```bash
tail -F /root/autodl-tmp/detector_tpa_pairing/run.log
```

启动即检查 LVIS 依赖。载入 checkpoint 前检查参数完整性与形状；每张图在缓存前
验证末端原生分类和融合重算（logit 绝对误差不超过 `1e-4`，分数不超过 `5e-6`）。
top-300 浮点边界差异单列记录。失败会中止，不继续给出诊断结论。

缓存为 float32 query / ROI 特征及预测框，不存全部 1203 类 logits。
按 450 图、900 queries、768 维、两个模型估算约 4.6 GiB，建议预留至少 6 GiB；
首张图后脚本会打印实际剩余缓存估算。速度取决于 GPU、CPU 与磁盘，未作服务器计时。

原命令再次运行会跳过已缓存的图，适合中断恢复。manifest 锁定 checkpoint、
文本库及 CLIP head 文件的 SHA256、代码与推理协议；不匹配时拒绝混用，需新输出
目录。断点续跑时重新生成的 bank 也必须与旧缓存完全一致。
已完成缓存后可加 `--phase analyze --device cpu` 仅 CPU 重算（无需 Detectron2，
但仍需 PyTorch、NumPy、LVIS 和原始文件来核验 SHA256）；CPU 不一定比单卡重算快。
不要在需要复用缓存时更改 checkpoint、文本库、协议或相关代码。

完成后 `report.json` 的 `complete` 为 true。重点输出：

- `Paired classification`：共同 GT 上按类别平均的分类对照和定位分区。
- `LVIS panel`：五组官方匹配的诊断 TP/FP。
- `Final-only swap deltas`：`new_old-new_new`、`old_new-old_old`、`new_mean-new_new`。
- JSON 的 `rows` / `paired.*.variants.*.per_class`：逐实例与逐类证据。
- `panel_pr.*.per_class.*.iou_curves`：PR 曲线；不要只看末端 TP/FP 总数。

## 测试

```bash
python -m pytest -q --rootdir=tests --confcutdir=tests \
  tests/test_pairing_diagnostic_ops.py \
  tests/test_pairing_lvis_support.py \
  tests/test_detector_tpa_pairing.py
```

未安装 LVIS 时相关集成测试会跳过，其余张量测试可在没有 Detectron2/GPU 的环境运行。
完整 checkpoint 前向仍须在服务器的 Detectron2/Detrex 环境验证。

## 已保存结果：官方 GT 命中转移分解（纯 CPU）

`analyze_rare_gt_transitions.py` 不加载 checkpoint、特征缓存、PyTorch 或
Detectron2；只读取上述已完成的 `report.json`、`old_old_predictions.json`、
`new_new_predictions.json` 和 LVIS 标注。不会再次运行模型或改动原缓存。

官方 LVIS 的全类别 top-300、ignore 和逐图逐类一对一匹配决定每个 GT 的 TP/FN。
对“旧 TP → 新 FN”分解新模型的失败侧，对“旧 FN → 新 TP”也分解旧模型的
失败侧，两者相减必须与原报告的净 TP 变化闭合。正常三种原因是：

- `no_eligible_query`：全部原始 query 中没有满足指定 IoU 的框。
- `correct_pair_below_topk`：有合格框，但没有正确类别的合格候选进入全图 top-k。
- `matching_competition`：正确候选已进入保存结果，但被官方匹配分配给另一个 GT。

缓存覆盖与实际保存候选不一致等情况单列，不强行塞进这三种原因。几何覆盖不是
一对一召回；query 编号不能跨模型对应，脚本按原始 annotation ID 对应。
各原因还报告类别等权的 **macro recall 百分点贡献**，避免 GT 多的类别支配结论。
这不是 APr 归因：逐类正式 `ΔAP` 从完整验证集报告独立关联，不按漏检数量摊分。
只能定位输出侧的失败环节，不能单凭此结果归罪于 APR、训练初始化等因素。

默认检查原 `pairing_cache/manifest.json` 的标注 SHA256 与报告 fingerprint；若
manifest 不在，会明确警告无法验证原始标注哈希。无论如何，两边重新匹配的每类
GT/TP、FP、ignore 和完整分数排序 PR 曲线必须与原报告一致，否则拒绝输出完成报告。
修改这里的独立脚本不会改变原模型缓存的代码指纹。

服务器同步新增代码后运行（不需要分配 GPU）：

```bash
cd ~/LaMI-DETR
set -o pipefail

/root/miniconda3/envs/lami/bin/python -u tools/analyze_rare_gt_transitions.py \
  --source-json /root/autodl-tmp/detector_tpa_pairing/report.json \
  --old-ap-report /root/autodl-tmp/k5_12ep_rare_pr/kang_current_protocol_report.json \
  --new-ap-report /root/autodl-tmp/k5_12ep_rare_pr/report.json \
  --output /root/autodl-tmp/detector_tpa_pairing/gt_transitions.json \
  2>&1 | tee /root/autodl-tmp/detector_tpa_pairing/gt_transitions.log
```

原生预测 JSON 默认从 `--source-json` 同目录读取；搬动过文件可显式传入
`--old-predictions`、`--new-predictions`。只需分析召回原因时可同时省略两个
`--*-ap-report`，此时不会输出正式逐类 AP 关联。运行时间取决于 CPU/磁盘，
本地未对服务器数据计时；开始、旧模型匹配、新模型匹配、保存各阶段即时打印。

```bash
python -m pytest -q --rootdir=tests --confcutdir=tests \
  tests/test_lvis_gt_matching.py \
  tests/test_rare_transition_ops.py \
  tests/test_rare_gt_transitions_cli.py
```

## 完整验证集 AP50/AP75 与 TP 前的 FP（纯 CPU）

`compare_rare_pr_reports.py` 先比较两份完整验证集报告的逐类 AP、AP50、AP75。
默认关注之前下降最大的十类（包括 koala、cocoa_(beverage)、joystick、
roller_skate）；可用 `--focus` 显式指定其他类别。选择用于诊断，不是按验证集
类别结果调整模型或逐类融合参数。

原报告的 `per_class` 保存全部 rare 类的 AP 指标，但 `focus` 只保存当时
指定类别的原始 PR。缺失曲线会明确标记 `MISSING_CURVE`，不视作零 FP。
仅报告模式若缺曲线，仍保存 AP 比较，`complete=false`，CLI 退出码为 2。

加 `--fill-missing-curves` 后，若缺曲线，读取报告中记录的完整预测 JSON：
每边只为缺失的目标类别执行一次官方 CPU LVIS 评估。全验证集图片（含负样本）
都参与，完整类别预测先执行每图 top-300，再限制分析类别。**不使用 450 图
panel 代替完整验证集、不运行模型、不读 checkpoint、不用 GPU、不改原报告。**
补算的逐类 AP/AP50/AP75/AR 必须与原报告一致，否则拒绝混用数据。实际预测和
标注哈希写入输出；旧报告没有原始哈希时，补算一致不能追溯证明整个旧报告的
文件身份，也不会把目标类别均值冒充完整 APr。

```bash
cd ~/LaMI-DETR
set -o pipefail

/root/miniconda3/envs/lami/bin/python -u tools/compare_rare_pr_reports.py \
  --old-report /root/autodl-tmp/k5_12ep_rare_pr/kang_current_protocol_report.json \
  --new-report /root/autodl-tmp/k5_12ep_rare_pr/report.json \
  --expected-old-apr 45.2037 \
  --expected-new-apr 41.5932 \
  --fill-missing-curves \
  --output /root/autodl-tmp/k5_12ep_rare_pr/ranking_comparison.json \
  2>&1 | tee /root/autodl-tmp/k5_12ep_rare_pr/ranking_comparison.log
```

如果预测文件搬了位置，加 `--old-predictions` / `--new-predictions` 指定新路径。
原生预测应是各自完整验证集的全类别结果，不是 `old_old_predictions.json` 等
抽样 panel 文件。已保存曲线齐全时即使带补算开关，也不会重评测。未对服务器
实测耗时；载入大 JSON、匹配、累积各阶段会即时打印进度。

输出含每个 IoU/类别的 TP、FP、最大召回、首个 TP 的 1-based 排名、首个 TP 前
的 FP 数，以及每个 TP 前的累计 FP。排名是**同一类别在全部验证图上的分数
顺序**，不是单图 top-300 的位置；忽略项不计排名，分数并列保留原评估顺序。
第 k 个 TP 只表示分数顺序，不能跨 checkpoint 当成同一 GT。没有 TP 时返回
`NO_TP`/null，而不是 `FP before first TP=0`。

这些统计用于区分 PR 排序和不同 IoU 下的召回变化，不直接证明训练原因；不要
仅凭首个 TP 的排名或相同的终点 TP 数推断完整 AP。

## 查出排在 rare TP 前面的具体误检（纯 CPU）

上一节生成 `complete=True` 的 `ranking_comparison.json` 后，读取同一份完整验证集
的旧/新预测及 LVIS 标注，复现官方按类别排序，并逐项核对上次报告的 TP/FP 数、
每个 TP 的排名和分数。输入预测与标注还须通过上次补算记录的 SHA256；路径从
对比文件读取，搬迁过文件时可用 `--old-predictions`、`--new-predictions`、
`--annotations` 指定新路径。两边每图 300 个候选的截断发生在按类别筛选之前。

```bash
cd ~/LaMI-DETR
set -o pipefail

/root/miniconda3/envs/lami/bin/python -u tools/inspect_rare_pre_tp_fps.py \
  --comparison /root/autodl-tmp/k5_12ep_rare_pr/ranking_comparison.json \
  --output /root/autodl-tmp/k5_12ep_rare_pr/pre_tp_fp_details.json \
  2>&1 | tee /root/autodl-tmp/k5_12ep_rare_pr/pre_tp_fp_details.log
```

默认分析对比文件中的十个类别，在终端列出旧/新首个和最后一个 TP 前的 FP 数、
新模型的几何重叠类型，并打印 `koala`、`roller_skate`、`joystick` 在 IoU 0.5
下排在 TP 前的具体预测。JSON 保留两边在 IoU 0.5/0.75 下所有排在最后一个
TP 前的 FP，包括类内排名、检测 ID、图像 ID/文件名、框和分数、最近同类 GT
及其他已标注类别 GT 的 IoU。没有 TP 时，对应“TP 前 FP 数”是 `null`。

“与其他已标注类别重叠”只描述框与标注的几何关系，不能单凭它断言物体被错认。
LVIS 未完整标注所有物体；未标注区域也不能直接称作背景。检测 ID 是这次官方
重播的局部编号，跨 checkpoint 不表示同一个 query。

```bash
PYTHONPATH=. python -m pytest -q --rootdir=tests --confcutdir=tests \
  tests/test_inspect_rare_pre_tp_fps.py
```

```bash
python -m pytest -q --rootdir=tests --confcutdir=tests \
  tests/test_rare_pr_comparison_ops.py \
  tests/test_rare_pr_curve_replay.py \
  tests/test_compare_rare_pr_reports_cli.py \
  tests/test_lvis_rare_pr.py
```
