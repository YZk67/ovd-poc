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

## 对具体 FP 区域比较两路分数和框重叠

`diagnose_rare_fp_regions.py` 读取 `pre_tp_fp_details.json`，默认选 `koala`、
`roller_skate`、`joystick` 的新模型前置 FP，并加入这三类旧/新模型的 TP 作为参照。
打印实际选中的图像 ID；默认最多 20 张。已有 `pairing_cache` 保持只读。

缓存中可能没有全部负样本图。带 `--fill-missing` 时，只对缺少的图片和 checkpoint
各补一次原生前向，写入新输出目录下的 `region_cache`；完整缓存不启动模型。
补算前检查 checkpoint、模型代码、配置和文本库的原始哈希，补算的 prototype
bank 必须与原缓存一致（浮点最大误差不超过 `1e-6`）。每张图断点续跑可复用。

```bash
cd ~/LaMI-DETR
mkdir -p /root/autodl-tmp/k5_12ep_fp_regions
set -o pipefail

CUDA_VISIBLE_DEVICES=0 /root/miniconda3/envs/lami/bin/python -u \
  tools/diagnose_rare_fp_regions.py \
  --fp-details /root/autodl-tmp/k5_12ep_rare_pr/pre_tp_fp_details.json \
  --pairing-cache /root/autodl-tmp/detector_tpa_pairing/pairing_cache \
  --output /root/autodl-tmp/k5_12ep_fp_regions/report.json \
  --fill-missing --dump-device cuda:0 --device cpu \
  2>&1 | tee /root/autodl-tmp/k5_12ep_fp_regions/run.log
```

Checkpoint 路径默认读取原 manifest；搬迁后用 `--old-checkpoint` / `--new-checkpoint`
覆盖路径，文件内容哈希仍须相同。去掉 `--fill-missing` 可先纯 CPU 检查缓存；若
缺失则保存图片清单和 `complete=false`，退出码 2。模型前向需要服务器的 lami
环境；本地测试验证了缓存重算和缺失图片调度，没有执行真实 checkpoint 前向。

每个来源 FP/TP 先通过原像素框和融合分数找回来源 query（容差 `0.05` 像素、
`5e-5` 分数）；找不到会中止，多解会输出全部候选并标记 `ambiguous`。
另一 checkpoint 按几何对应，分别输出最高 IoU 框，以及 IoU≥0.5 的框中该类别
融合分数最高者，同时记录两边区域内进入全图 top-300 的同类候选数。

输出包括 detector sigmoid 概率、CLIP 全词表 softmax 概率、目标类别在各分支全词表中的排名、
融合分数、框坐标、图像 top-300 状态及阈值。与其他标注类别重叠时，也报告该
标注类别的两路分数（例如 roller_skate 区域的 skateboard 分数）。新模型同图
同类前置 FP 的完整两两 IoU 矩阵可核实是否集中在同一物体上。

另一模型的 detector query 与来源框一般并不完全重合；这里是两个原生框的
区域对照，**不是把同一 ROI 强行送入两个 detector 分类器的因果实验**。
区域内最高分的选择由分数决定，几何最高 IoU 的对照独立列出。加权 log 分数
变化只做代数分解，不据此直接归因于某项训练修改，也不估算 APr 增益。

```bash
python -m pytest -q --rootdir=tests --confcutdir=tests tests/test_rare_region_pairing.py
```

## 固定 query 的中心 / 多原型 / 偏置分解

`analyze_rare_fp_logits.py` 读取上一节已完成的 `report.json`，直接复用它记录的
原缓存和补充 `region_cache`。**只用 CPU，不加载 detector/checkpoint、不读取图片，
也不会补算缺失缓存。** 不修改原报告、缓存、训练配置或推理协议。

```bash
cd ~/LaMI-DETR
set -o pipefail

/root/miniconda3/envs/lami/bin/python -u tools/analyze_rare_fp_logits.py \
  --source-json /root/autodl-tmp/k5_12ep_fp_regions/report.json \
  --output /root/autodl-tmp/k5_12ep_fp_regions/logit_decomposition.json \
  2>&1 | tee /root/autodl-tmp/k5_12ep_fp_regions/logit_decomposition.log
```

对于已固定的 post-linear query 特征 `x` 和目标类别原始 prototypes `p_k`：

```text
u_k = dot(normalize(x), normalize(p_k))
center = scale * dot(normalize(x), normalize(mean(p_k)))
native_logit = scale * tau * log(mean(exp(u_k / tau))) + bias
mode_delta = native_logit - bias - center

mean_slot = scale * mean(u_k)
center_to_mean_correction = mean_slot - center
dispersion_uplift = native_logit - bias - mean_slot

native_logit = center + mode_delta + bias
mode_delta = center_to_mean_correction + dispersion_uplift
```

`mode_delta` 是相对**先平均原始 prototype、再归一化**的单中心分类器的净变化，
可以正也可以负。只有 `dispersion_uplift`（相对平均 cosine 的 LME 增量）保证非负；
不能忽略中心归一化和平均方式造成的校正项，把它直接当成多原型净收益。
若中心向量为零/接近零，JSON 会标记方向未定义，并保留与现有归一化实现一致的数值。

脚本逐条核验框、原生 detector/CLIP 分数和融合分数，检查加性分解闭合，再输出：

- 来源 FP/TP、另一模型的最近框和区域内最高分框的三项 logit，以及两个子项。
- 以新模型 FP/TP 区域为组的旧新差值均值；旧模型 TP 参照不重复纳入主汇总。
- 保持 query、框、CLIP 不变，只把末端分类器变为单中心时，选中 FP/TP 的分数顺序。
- JSON 额外保存每个 slot 的 cosine、后验权重、prototype 范数、偏置及重算误差。

这是 **logit** 的加性分解；sigmoid 概率和最终融合分数不能这样相加。
固定 query 的单中心分类器也**不等于全模型 `mode_scale=0`**：后者还可能改变前面的
query fusion 和预测框。旧新对应框仍各用自己的特征，不是共享特征的跨模型因果对照。

主汇总中的旧端点只继承新来源区域的分组，不代表它已被正式匹配为旧模型 FP/TP。
重叠 FP 和被多次选中的旧 query 不独立，脚本报告 unique query 数；样本为事后选中。
分数顺序检查不重新选择 top-300、不重新匹配 TP/FP，也不计算或预测正式 AP。
不自动给出“修改哪项损失能提高 APr”的结论。

```bash
python -m pytest -q --rootdir=tests --confcutdir=tests tests/test_rare_logit_decomposition.py
```

## 第一步：同一 checkpoint 的固定半径前向几何审计

`audit_tpa_geometry.py` 默认只审计 **new checkpoint**。加载一次可信的本地 checkpoint，
仅在 CPU 上重建文本侧 TPA，复用上一节区域报告中的 query/box/CLIP 缓存。
**不运行 detector 或图像前向，不做梯度审计，不训练，不重新评测 AP。**
它不会修改 checkpoint、缓存、训练配置或模型代码。

```bash
cd ~/LaMI-DETR
set -o pipefail

/root/miniconda3/envs/lami/bin/python -u tools/audit_tpa_geometry.py \
  --source-json /root/autodl-tmp/k5_12ep_fp_regions/report.json \
  --side new \
  --prompt-bank dataset/metadata/lvis_claude_prompts_convnextl.npy \
  --output /root/autodl-tmp/k5_12ep_fp_regions/geometry_audit.json \
  2>&1 | tee /root/autodl-tmp/k5_12ep_fp_regions/geometry_audit.log
```

checkpoint 路径从区域报告读取；如仅移动了文件，可用 `--checkpoint` 指定新路径，
但 SHA256 必须一致。脚本同时核验 prompt 张量身份、类别顺序、共享 TPA 权重别名、
缓存来源、原生 prototype 重建误差和所选 query 的原生分数。
CPU/GPU 浮点容差明确保存在报告中；不匹配直接报错，不自动生成新缓存。
`--side old` 可单独用于旧 checkpoint 的零强度对照，但不自动追加执行。

严格区分以下三个均值：

```text
c = mean(value_proj(prompts))                  # 投影后的 prompt 均值
a = mean(attention-weighted prototypes)        # 固定半径变换前的 slot 均值
b = mean(fixed-radius prototypes)              # 固定半径变换后的 slot 均值

u_k = (p_k - c) / clamp(norm(p_k - c), min=1e-6)
r = mode_strength * clamp(norm(c), min=1e-6)
q_k = c + r * u_k                              # strength > 0 时
b = c + r * mean(u_k)                          # 不保证 b=c，也不保证 b=a
```

变换前后的权重、slot prior、attention 和 query 完全相同，只在局部分类重放中
绕过/应用现有固定半径变换。`before` 不是“从未使用防坍塌训练”的模型，也不是
重新初始化的 TPA，更不等于把变换后的 prototypes 压成单中心的 `mode_scale=0`。

输出分为三部分：

- 全词表及 rare/common/frequent 分组：中心位移、中心方向夹角、逐 slot 范数、
  单位 slot 均值的长度和方向、变换前后的 rank/cosine。JSON 保留全部类别，
  不只查看事后挑出的失败类别。
- 固定原生 query/box/CLIP：FP 与 TP 的 `after-before` logit 变化，拆分为中心响应、
  center-to-mean 校正和 LME 增量；也保留投影后 prompt 中心 `c` 的固定响应。
- 所选 FP/TP 的融合分数顺序，不重新筛选全图 top-300、不重新匹配 FP/TP。
  不确定 query 身份的区域仅明细列出，不进入汇总；相同 query 去重。

零中心/接近零残差会单独标记；被 clamp 的残差实际半径不必等于目标半径。
几何摘要是类别等权的描述统计，不是 LVIS APr（其中还包括没有验证 GT 的类别）。

这一步能隔离**固定半径前向变换在当前已训练权重下的直接影响**。
若它同时抬高 FP 和 TP，或中心偏移并未解释误排序，不能简单删掉该变换。
即使观察到 FP 受益，也不能据此判定是哪项训练修改造成了最终权重差异，或承诺
回退后提高 APr；APR、梯度投影、初始化的训练归因仍需后续独立对照。

```bash
python -m pytest -q --rootdir=tests --confcutdir=tests tests/test_tpa_geometry_audit.py
```

## 第二步：检测任务 / APR / 冲突投影的局部梯度审计

`audit_tpa_gradients.py` 读取第一步完整 `geometry_audit.json`，只审计同一新 checkpoint。
**需要短暂使用一张 GPU 求真实训练损失的梯度，但不创建 optimizer/scheduler，
不执行参数更新，不保存新模型 checkpoint，也不训练任何 epoch。**
缓存特征没有检测损失计算图，因此不能用 CPU 分类代理损失冒充真实任务梯度。

```bash
cd ~/LaMI-DETR
set -o pipefail

CUDA_VISIBLE_DEVICES=0 /root/miniconda3/envs/lami/bin/python -u tools/audit_tpa_gradients.py \
  --geometry-json /root/autodl-tmp/k5_12ep_fp_regions/geometry_audit.json \
  --config-file lami_dino/configs/dino_convnext_large_4scale_12ep_lvis.py \
  --windows 2 --microbatches 8 --batch-size 4 \
  --output /root/autodl-tmp/k5_12ep_fp_regions/gradient_audit.json \
  2>&1 | tee /root/autodl-tmp/k5_12ep_fp_regions/gradient_audit.log
```

默认预算是 **2 个独立窗口 × 8 个 micro-batch × 4 张图 = 64 次训练图像曝光**，
每个窗口平均 32 张图的梯度，但所有窗口始终停留在同一 checkpoint，不连续更新。
重复采样可能重复图像，不能声称是 64 张不同图片。脚本硬限制最多 128 次曝光。
如显存不足，可改为 `--batch-size 2 --microbatches 16`，同时使用新的输出路径；
这会改变 FedLoss 子集和局部 batch 构成，不是严格等价的梯度。

先检查已有 query 缓存，再加载完整 checkpoint，使用配置原有的
`lvis_v1_train_norare`、训练增强、RepeatFactor sampler、dropout、denoising、匹配和
加权损失。固定随机种子，worker 数为零；iteration 和 APR 内部计数保持 checkpoint
位置，不重启 warmup。只请求 TPA 参数的偏导，其他参数作为常量，检测前向保持
training 模式。每个窗口检查所有模型参数与持久 buffer 未被写入，`.grad` 未填充。
不支持含训练态 BatchNorm 的其他模型，避免混入不同统计口径。

每个 micro-batch 在**同一个前向图**上分别求：

```text
g_detector = ∇(全部加权检测损失，包括分类、框、GIoU、辅助层、encoder、denoising)
g_rpsa     = ∇(已加权、已应用 schedule 的 RPSA)
g_apr      = ∇(原模型返回的完整加权 APR)

各项先跨 micro-batch 平均：
g_task   = g_detector + g_rpsa
g_total  = g_task + g_apr
g_routed = 原训练器使用的 route_conflicting_task_gradient(g_total, g_apr)
```

**不能先投影每个 micro-batch 再平均。** JSON 保留图像 ID、增强后尺寸、FedLoss
类别子集、各损失、未连接的参数项，以及梯度范数、余弦和各参数块范数。
单卡 FedLoss 只从本地 batch 收集 GT；虽有相同的图像曝光数，仍不是原四卡 batch
的精确重放，也未恢复当时的数据/随机数状态。

GPU 求导结束先保存 `gradient_audit.gradients.pt`，随后 CPU 计算文本侧 JVP。
如果分析中断，或只需再次打印结果，给**同一命令**加 `--reuse-gradients`，
会核验缓存身份并跳过所有图像/GPU 工作；不要删除梯度缓存重跑。
缺失或不匹配的源文件直接报错，不补算图像，不自动覆盖已有梯度缓存。

### 如何读方向审计

针对 detector、RPSA、task、APR、未投影总梯度、投影后 task、最终 routed 和
投影新增项，分别计算 `J[-g/||g||]`：它是沿**单位参数空间下降方向**的局部导数。
这不是实际更新，也不是把 APR 梯度当作 prototype 的直接梯度；它经过完整文本侧
key/value/query 参数化和固定半径变换的 Jacobian。

- `d_shift`：固定半径变换前后中心偏移比的变化；正值表示局部增大。
- `d_spread`：变换前残差相对 prompt 中心的分散程度变化。
- `d_unit_mean`：变换后单位原型平均向量长度变化。
- `d_cos`：变换后 pairwise cosine 变化；不能把更低 cosine 直接解释为更高 AP。
- `value-center angular-speed`：投影后 prompt 中心方向的转动速率，**无正负**，
  不代表向正确或错误语义转动。零中心方向标记为未定义，不纳入该均值。
- 固定验证 query 的分类 logit 导数：中心、mean-correction、LME 增量、总 logit。
  这些验证标签只对诊断结果分 FP/TP，绝不用于训练损失。

单位方向归一化去掉了梯度量级，**不能仅比较两行单位导数就说谁主导了训练**。
报告同时给原始梯度范数，以及统一采用最终 routed 梯度裁剪系数后的
`common-clip logit` 导数。后者保留各分量量级，可作加性分解，但仍没有乘 LR，
也没有 AdamW 动量、预条件和 weight decay；它不是 optimizer 实际位移。

全词表/频次分组统计为描述性均值，包括没有验证 GT 的类；选定 FP/TP 分数导数
只改变 TPA，固定 decoder query、框和 CLIP，不重新运行 query fusion 或选 top-300。
梯度来自训练态前向（包含 dropout），几何/分数 JVP 则读取无 dropout 的推理态
文本原型，衡量这些训练梯度对推理表示的局部影响。
结果只能说明**当前训练位置、这些 batch 上的局部趋势**，不能证明整个训练历史的
因果关系，也不能承诺改动后恢复旧模型 APr。

```bash
python -m pytest -q --rootdir=tests --confcutdir=tests tests/test_tpa_gradient_audit.py
```
