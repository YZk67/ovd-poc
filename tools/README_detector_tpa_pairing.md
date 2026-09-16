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

## 第三步：只移除固定半径参数化的 4ep 训练消融

配置：`lami_dino/configs/dino_convnext_large_4scale_4ep_lvis_no_radius.py`。
这次是**新的训练实验，需要 GPU，耗时接近此前 effective-bs32 的 4ep run**；
不是继续做 CPU 重放，也不自动启动第二组训练或延长到 12ep。

唯一训练变量是 `model.classifier.tpa_prototype_mode_strength: 1.5 -> 0.0`。
已有实现中，零强度直接保留 attention 聚合的五个原型，绕过固定半径变换。
不增加新方法、不改初始化函数、不改 APR/投影。与原生配置使用同一种子时，
所有初始可训练张量和随机数消耗相同，只有这个持久 buffer 不同。
**初始输出原型不保证相同**：变换在第一次 forward 就不同，这正是消融变量。
这也不等于 `model.tpa_eval_mode_scale=0`，后者把原型收缩到均值；本实验保持该值为 1。

继承 `4ep_lvis_screen` 的其余设置，不缩短 LR 时间线：

| 项目 | 固定设置 |
| --- | --- |
| 初始化 | CLIP backbone-only；detector/TPA 从头初始化；seed=42 |
| batch | 4 卡，physical global batch=16，累积 2 次，effective batch=32 |
| 时长 | 28,400 次 optimizer updates（项目的 4ep 标签）；LR horizon=85,200 |
| LR | 基础 detector=1e-4，TPA=1e-3；继承原 LR warmup，不额外冻结或压缩时间线 |
| TPA | K=5，slot prior=0.2，identity value init，dropout=0.1 |
| APR/投影 | barrier 和 balance 权重 0.10/0.03；冲突投影开启；独立裁剪 0.5 |
| RPSA | 原版 RPSA，weight=0.05，20,000 开始、8,000 ramp；不是 teacher RPSA |
| 推理 | alpha=0、beta=0.3、novel_scale=3；Eq.1 tau=0.004375，Eq.2 tau=0.07，top-3 |
| 保存/评测 | checkpoint 每 14,200 次更新；正式评测只在 28,400 结束时 |

参照已有**同阶段、同 batch/protocol**的原生 K=5 结果：
`instructdet_k5_effective_bs32_4ep_seed42/model_0028399.pth`（如果仍保留），
AP=33.8616、APr=37.7073、rank=4.7566。
该目录后续已续训：**现在的 model_final.pth、预测 JSON 和最后一条 log 不是 4ep 对照**。
不要与旧 bs16 的 4ep 或现在的 12ep 数值作匹配因果比较。已有参照避免默认多跑一组；
但历史参照未在本提交重新训练，多种子波动、当时代码/资产身份仍需单独说明。

### 运行

拉取包含该配置的提交后，先运行纯 CPU 回归测试和资产检查：

```bash
cd ~/LaMI-DETR
/root/miniconda3/envs/lami/bin/python -m pytest -q --rootdir=tests --confcutdir=tests \
  tests/test_tpa_radius_ablation.py tests/test_text_prototype_aggregator.py \
  tests/test_prototype_ops.py tests/test_checkpoint_init.py \
  tests/test_lr_scheduler_horizon.py tests/test_gradient_accumulation.py
/root/miniconda3/envs/lami/bin/python tools/preflight_instructdet.py --hash
```

使用全新目录、**不加 `--resume`、不加载现有 detector checkpoint**。下面的目录检查
用于避免混入旧 run 或覆盖日志；检查失败时先确认目录内容，不要直接删除重跑。

```bash
cd ~/LaMI-DETR
if [ -e /root/autodl-tmp/instructdet_k5_no_radius_bs32_4ep_seed42 ]; then
  echo "Output already exists; inspect it before starting or resuming."
else
  CUDA_VISIBLE_DEVICES=0,1,2,3 /root/miniconda3/envs/lami/bin/python tools/train_net.py \
    --config-file lami_dino/configs/dino_convnext_large_4scale_4ep_lvis_no_radius.py \
    --num-gpus 4 \
    train.output_dir=/root/autodl-tmp/instructdet_k5_no_radius_bs32_4ep_seed42 \
    dataloader.evaluator.output_dir=/root/autodl-tmp/instructdet_k5_no_radius_bs32_4ep_seed42
fi
```

只有本组因中断需要恢复时，使用**同一 no_radius 配置、同一新目录**加 `--resume`。
强度是 checkpoint buffer：从原生模型 resume 会把 1.5 重新加载回来，
仅在命令行写 0 不足以抵消它，且这种续训也不是从头训练消融。

启动后核对日志为 iteration=0、stop=28400、LR horizon=85200、effective batch=32。
`metrics.json` 新增的 `tpa/prototype_mode_strength` 和 `tpa/fixed_radius_enabled`
应始终为 0；`tpa/task_gradient_scale` 应为 1。同时观察已有的 APR、冲突投影、
梯度和 rank/cosine 指标；不要因 rank 下降临时改 LR、APR 或 strength。

### 结束后提取结果

```bash
cd ~/LaMI-DETR
printf '%s\n' 'AP,AP50,AP75,APs,APm,APl,APr,APc,APf'
grep -E 'copypaste: [0-9]' \
  /root/autodl-tmp/instructdet_k5_no_radius_bs32_4ep_seed42/log.txt \
  | tail -n 1 | sed 's/.*copypaste: //'
/root/miniconda3/envs/lami/bin/python tools/check_tpa_rank.py \
  /root/autodl-tmp/instructdet_k5_no_radius_bs32_4ep_seed42/model_final.pth \
  dataset/metadata/lvis_claude_prompts_convnextl.npy
```

应显示 `Kp=5, slot_prior=0.2, mode_strength=0`。训练退出不等于评测成功；没有
`copypaste` 数字时不能认定已有 AP。保存 config、日志、metrics、2ep/4ep checkpoint
和正式评测 JSON，不能仅记录最终 rank。

解读边界：

- APr 上升且仍不坍塌：支持固定半径在这组训练条件下有净性能代价；仍非全局保证。
- APr 上升但坍塌：是“移除半径”的整体收益，**不是保持不坍塌的性能修复**。
- APr 下降：不能把前述少量固定 query 的局部收益外推到完整训练。
- 不论结果如何，这项对照包含半径对初始化输出、训练路径和最终前向的全部影响，
  不会单独证明“中心旋转”或“均值校正”就是全部 APr 差距的原因。

本组不自动满足 12ep 启动条件；仍按约定用正式 4ep APr（优先 40 以上）与 rank
一起决策，不因 loss 下降、rank 变高或几个事后选中类别改善而自动延长训练。

## 第四步：已完成的 no-radius 4ep，只续到 8ep 验证

此节保留原 8ep 停止方案。**当前已改为文末“当前启动方案：一次恢复至 12ep”**，
8ep 照常评测，但不主动结束进程，避免多一次 sampler/RNG 重建。

当前 no-radius 4ep：AP=33.1235、APr=39.7325；全词表 rank=4.7135、
pairwise cosine=0.19964、mode strength=0。相比原生同阶段 APr +2.0252、
AP -0.7381。这是继续验证的依据，不代表已达到 12ep 的目标 APr。

新配置 `dino_convnext_large_4scale_8ep_lvis_no_radius.py` **只改变停止点**：

| 项目 | 4→8ep 续训 |
| --- | --- |
| 恢复 / 停止 | 从 iteration=28400 开始，到 56799 完成；新增 28,400 次 optimizer updates |
| LR 时间线 | 仍为 85,200，继承原 scheduler；不重启 warmup、不把衰减提前到 8ep |
| batch / 结构 / loss | physical batch=16、累积 2 次；radius=0；APR、投影、RPSA 不变 |
| 推理 | alpha=0、beta=0.3、novel_scale=3；其余完全继承 no-radius 4ep |
| 保存 / 评测 | 每 14,200 updates 保存；6ep=42599；8ep=56799 并正式评测，然后停止 |

**不要改用原生 12ep 配置，也不要新建空输出目录后直接写 `--resume`。**
原训练器找不到 `last_checkpoint` 会回退到从头初始化。下面的 CPU 预检先检查
真实指针、iteration、完整 optimizer/LRScheduler 状态、85,200 horizon、累积次数，
以及 checkpoint 内每个 TPA 的 K=5、slot prior=0.2、radius=0；不加载 detector/GPU。
检查失败就停下查原因，不跳过检查改成 weights-only 初始化。

### 首次从 4ep 续训

先同步本提交。确认该目录没有训练/评测进程在写入，然后运行：

```bash
cd ~/LaMI-DETR
/root/miniconda3/envs/lami/bin/python -m pytest -q --rootdir=tests --confcutdir=tests \
  tests/test_no_radius_resume.py tests/test_tpa_radius_ablation.py \
  tests/test_lr_scheduler_horizon.py tests/test_gradient_accumulation.py
```

下面用 `&&` 保证预检/快照成功后才会启动训练：

```bash
cd ~/LaMI-DETR
/root/miniconda3/envs/lami/bin/python -u tools/prepare_no_radius_8ep_resume.py \
  --output-dir /root/autodl-tmp/instructdet_k5_no_radius_bs32_4ep_seed42 \
  --snapshot && \
CUDA_VISIBLE_DEVICES=0,1,2,3 /root/miniconda3/envs/lami/bin/python -u tools/train_net.py \
  --config-file lami_dino/configs/dino_convnext_large_4scale_8ep_lvis_no_radius.py \
  --num-gpus 4 --resume \
  train.output_dir=/root/autodl-tmp/instructdet_k5_no_radius_bs32_4ep_seed42 \
  dataloader.evaluator.output_dir=/root/autodl-tmp/instructdet_k5_no_radius_bs32_4ep_seed42
```

`--snapshot` 将指针指向的 4ep checkpoint、`last_checkpoint`、`config.yaml`、
`metrics.json`、`log.txt`、预测 JSON **独立复制**到本目录的 `four_ep_snapshot/`，
保存大小与 SHA256。需要这些文件总大小的额外磁盘空间；不会使用会随覆盖而损坏
的硬链接。训练期间当前 `model_final.pth`、配置和预测结果可能被覆盖；原始 4ep
证据在快照中保留。不改任何 checkpoint 内容，不移动或删除源文件。
源数据未变时可重复执行准备；已有快照不一致则拒绝覆盖。复制失败时保留临时目录，
报告其路径，不自动清理用户数据。不带 `--snapshot` 时只校验、不写文件。

该准备步骤仅适用于 **iteration=28399 的首次阶段转换**。如 4→8ep 途中中断，
不要用更晚 checkpoint 重建“4ep 快照”；直接重跑上面的 `train_net.py` 命令部分，
同一 8ep no-radius 配置、同一输出目录和 `--resume`。保留正确的 `last_checkpoint`。
恢复完整优化器和 scheduler，但不是 bitwise 连续重放的数据顺序/RNG 保证。

启动后应看到 `Resuming training from iteration 28400`（首次续训）；
LR horizon=85200、effective batch=32。该阶段 detector LR=1e-4、TPA LR=1e-3，
RPSA scale=1、`tpa/prototype_mode_strength=0`。若看到 `Starting ... iteration 0`、
累积变 1、radius=1.5 或 LR 时间线变化，先停止检查，不继续消耗算力。

### 8ep 结束后

```bash
cd ~/LaMI-DETR
/root/miniconda3/envs/lami/bin/python - <<'PY'
import json
from pathlib import Path
root = Path('/root/autodl-tmp/instructdet_k5_no_radius_bs32_4ep_seed42')
rows = [json.loads(line) for line in (root / 'metrics.json').read_text().splitlines() if line.strip()]
# Final EvalHook may write at 56800, after optimizer iteration 56799 finished.
evaluations = [row for row in rows if row.get('iteration') in (56799, 56800) and 'bbox/APr' in row]
if not evaluations:
    raise SystemExit('No 8ep bbox/APr at iteration 56799/56800: check eval/log before treating the last result as 8ep.')
row = evaluations[-1]
print({key: row.get(key) for key in ('iteration', 'bbox/AP', 'bbox/APr', 'bbox/APc', 'bbox/APf')})
print('Native-radius 8ep reference: AP=40.6867, APr=40.4466')
print('delta_APr:', row['bbox/APr'] - 40.4466)
PY
/root/miniconda3/envs/lami/bin/python tools/check_tpa_rank.py \
  /root/autodl-tmp/instructdet_k5_no_radius_bs32_4ep_seed42/model_final.pth \
  dataset/metadata/lvis_claude_prompts_convnextl.npy
```

原生参照是 **8ep / iteration=56799**，即使旧评测目录名曾误写为 `5ep`，也不要
按目录名混用 5ep 的 AP=37.3383、APr=40.3582。比较正式 AP/APr 与全词表 rank，
检查 4ep 的 rare 收益是否延续及整体 AP 代价；之后再决定是否续至 12ep。
8ep 尚未到原 LR 衰减点，单个阶段结果不是最终 12ep 收益的保证或否定。
本配置到 8ep 即停止，不自动启动任何新消融或 12ep 续训。

## 当前启动方案：一次恢复至 12ep，8ep 评测后继续

已确认当前 4ep 后尚未启动续训。本次改为**只恢复一次、剩余 8ep 连续执行**。
使用 `dino_convnext_large_4scale_12ep_lvis_no_radius.py`，不是原生有半径的
`dino_convnext_large_4scale_12ep_lvis.py`。新配置从 no-radius 4ep 继承一切，
只把 `train.max_iter` 改成 85,200；不引入新训练方法或改变 APR/投影/RPSA。

| 位置 | 行为 |
| --- | --- |
| 28,400 | 从原目录的 no-radius 4ep checkpoint 恢复模型、optimizer 和 scheduler |
| 56,799（8ep） | 保存 `model_0056799.pth`，正式评测，结束后同一进程继续训练 |
| 78,100 | 按原 12ep 时间线衰减 LR；不是在 8ep 提前衰减 |
| 85,199（12ep） | 保存最终 checkpoint；最终评测后退出 |

physical batch=16、累积=2、effective batch=32、seed=42、radius=0、slot prior=0.2、
K=5、推理 alpha=0/beta=0.3/novel_scale=3 均不变。继承 checkpoint period=14,200
（6/8/10/12ep 保存）和 eval period=28,400（恢复后在 8/12ep 评测）。

边界：这是**减少一次计划中的进程重启**，不是从 iteration=0 到 12ep 的严格
等价重放。已有 checkpoint 没有保存 RNG、sampler 游标或 worker/预取状态，
首次 resume 仍会重新创建这些状态；这次不尝试事后恢复它们，不承诺 APr 无波动。
现有预检验证的是优化器/LR/模型状态，不是完整随机训练轨迹的等价性。

### 服务器启动命令（替代上面的 8ep 命令）

先同步包含新配置的提交。保持原输出目录，确认没有其他训练/评测进程向它写入。
CPU 预检沿用工具名 `prepare_no_radius_8ep_resume.py`，但显式指定 `--target-epochs 12`。
如果原 8ep 准备步骤已生成 4ep 快照，校验源文件和 SHA256 一致后直接复用；
不会修改该快照的旧 manifest，也不会重新复制大文件。

```bash
cd ~/LaMI-DETR
PYTHONPATH=. /root/miniconda3/envs/lami/bin/python -m pytest -q \
  --rootdir=tests --confcutdir=tests \
  tests/test_no_radius_resume.py tests/test_tpa_radius_ablation.py \
  tests/test_lr_scheduler_horizon.py tests/test_gradient_accumulation.py && \
/root/miniconda3/envs/lami/bin/python -u tools/prepare_no_radius_8ep_resume.py \
  --output-dir /root/autodl-tmp/instructdet_k5_no_radius_bs32_4ep_seed42 \
  --target-epochs 12 --snapshot && \
CUDA_VISIBLE_DEVICES=0,1,2,3 /root/miniconda3/envs/lami/bin/python -u tools/train_net.py \
  --config-file lami_dino/configs/dino_convnext_large_4scale_12ep_lvis_no_radius.py \
  --num-gpus 4 --resume \
  train.output_dir=/root/autodl-tmp/instructdet_k5_no_radius_bs32_4ep_seed42 \
  dataloader.evaluator.output_dir=/root/autodl-tmp/instructdet_k5_no_radius_bs32_4ep_seed42
```

必须核对日志：`Resuming training from iteration 28400`、stop=85200、LR horizon=85200、
effective batch=32；radius monitor=0。当前目录的名字仍含 `4ep` 是为沿用恢复指针，
不代表现在只训练 4ep。预检或测试失败不要绕过 `&&` 直接开始训练。

8ep 评测期间日志暂时没有训练 loss 是正常的；评测后应出现 iteration≥56800 的
训练记录。无需杀进程或另起 eval-only。用原生 8ep AP=40.6867/APr=40.4466 作参照，
如果决定继续，让当前进程运行即可，不再执行一次 `--resume`。

8ep 的 AP 记录在日志/metrics，checkpoint 是 `model_0056799.pth`，**不要用
当时仍可能属于 4ep 的 `model_final.pth` 检查 8ep rank**。结束 12ep 后，
`model_final.pth` 才会被新的最终权重覆盖。预测 JSON 在下一次评测会被覆盖，
如果要留 8ep 的逐框结果，在下一次评测之前独立复制 `lvis_instances_results.json`；
这次不改变 evaluator 的输出行为。4ep 原始证据继续由 `four_ep_snapshot/` 保存。

如意外中断，只重跑上面的训练命令部分（仍是 12ep no-radius 配置、同一目录、
`--resume`）；不要让更晚 checkpoint 再通过仅针对 4ep 的准备步骤或覆盖 4ep 快照。

## no-radius 8ep→10ep：定位 −0.5812 APr（CPU，不再训练）

本次仅比较同一 no-radius run 的两个阶段，不能使用前面原生 radius / Kang 报告替代：

| 阶段 | checkpoint | AP | APr |
| --- | --- | --- | --- |
| 8ep | `model_0056799.pth` | 40.4804 | 42.8843 |
| 10ep | `model_0070999.pth` | 41.7624 | 42.3031 |
| 12ep（不是本次输入） | `model_final.pth` | 44.6979 | 42.4229 |

`run_rare_stage_comparison.py` 顺序执行两次 **CPU 官方 LVIS 匹配**，在同一次匹配中保存
所有 rare 类的 IoU 0.50/0.75 PR 曲线，然后比较报告，不加载模型、checkpoint 或 GPU。
两个预测输入和标注必须都存在才开始；APr 不符立即中止；拒绝覆盖已有报告。
默认自动选择这对结果中 AP 下降最大的 20 类，不复用以前挑选的类别。

**先确认 8ep 的预测 JSON 是否独立保存。** 训练目录中的同名 JSON 在 12ep 后已被覆盖，
日志里的 AP 数字无法恢复逐框结果。以下 8ep 路径是独立评测目录的约定，不代表它已存在。
若另存到其他位置，只替换 `--old-predictions`。不要为了通过文件存在检查改成 12ep 的 JSON。

```bash
cd ~/LaMI-DETR
mkdir -p /root/autodl-tmp/no_radius_8ep_vs_10ep_pr
set -o pipefail
/root/miniconda3/envs/lami/bin/python -u tools/run_rare_stage_comparison.py \
  --old-predictions /root/autodl-tmp/eval_k5_no_radius_bs32_8ep/lvis_instances_results.json \
  --new-predictions /root/autodl-tmp/eval_k5_no_radius_bs32_10ep/lvis_instances_results.json \
  --expected-old-apr 42.8843 --expected-new-apr 42.3031 \
  --top-declines 20 \
  --output-dir /root/autodl-tmp/no_radius_8ep_vs_10ep_pr \
  2>&1 | tee -a /root/autodl-tmp/no_radius_8ep_vs_10ep_pr/comparison.log
```

如果 8ep JSON 确实没有保存，需要用户决定是否补 **一次 8ep eval-only**。这不是 CPU
诊断的一部分，脚本绝不会自动启动它，也不需要重训；原有 10ep JSON 可直接复用。
以下命令只在尚未存在独立 8ep 目录时执行，保留现有结果，不覆盖训练目录：

```bash
cd ~/LaMI-DETR
if [ -e /root/autodl-tmp/eval_k5_no_radius_bs32_8ep ]; then
  echo '8ep directory already exists: inspect it or choose a new output path; no evaluation started.'
else
  CUDA_VISIBLE_DEVICES=0,1,2,3 /root/miniconda3/envs/lami/bin/python tools/train_net.py \
    --config-file lami_dino/configs/dino_convnext_large_4scale_12ep_lvis_no_radius.py \
    --num-gpus 4 --eval-only \
    train.init_checkpoint=/root/autodl-tmp/instructdet_k5_no_radius_bs32_4ep_seed42/model_0056799.pth \
    train.output_dir=/root/autodl-tmp/eval_k5_no_radius_bs32_8ep \
    dataloader.evaluator.output_dir=/root/autodl-tmp/eval_k5_no_radius_bs32_8ep \
    model.alpha=0.0 model.beta=0.3 model.novel_scale=3.0 model.tpa_eval_mode_scale=1.0
fi
```

输出：`old_report.json`、`new_report.json`、`comparison.json`。已有两份报告时可以直接：

```bash
/root/miniconda3/envs/lami/bin/python -u tools/compare_rare_pr_reports.py \
  --old-report /root/autodl-tmp/no_radius_8ep_vs_10ep_pr/old_report.json \
  --new-report /root/autodl-tmp/no_radius_8ep_vs_10ep_pr/new_report.json \
  --expected-old-apr 42.8843 --expected-new-apr 42.3031 \
  --top-declines 20 \
  --output /root/autodl-tmp/no_radius_8ep_vs_10ep_pr/comparison.json
```

读结果时关注：

- `All-valid-class APr attribution`：全部有效 rare 类（本数据通常 178，不是 taxonomy 的
  337）贡献闭合到官方 ΔAPr；同时列出提升、下降和 GT 数分层。不能只看净下降而忽略抵消。
- `Recall support vs shared-recall precision`：分别在 IoU .50/.75 的官方 101 点 recall
  网格，将该 IoU 的 ΔAP 拆为丢失召回区间 `lost-R`、新增召回区间 `gained-R`、共同召回
  区间精度变化 `shared-PR`；三项精确闭合到 **该类别该 IoU** 的 ΔAP。
- `mean dFP-before`：在两侧都存在的相同 TP 序号（相同 attained recall）处，前置 FP
  数变化。TP 序号不是同一 GT 身份；相同分数保持官方稳定顺序，“前置”不等于严格高分。
- 共同召回精度变差可能是 FP 排名上升，也可能是 TP 排名下降或匹配改变，不能直接称作
  “FP 新增的因果贡献”。top-300 内召回减少也不证明 detector 没有合格 proposal。
- .50/.75 的 PR 拆分不是十个 IoU 平均 AP 的完整机制解释；单 GT 类很敏感，所有分析
  只用于诊断，不据此按验证集类别调整阈值、重训或声称统计显著。

本地回归测试（无需模型或 GPU；真实 LVIS replay 测试在缺少依赖时 skip）：

```bash
PYTHONPATH=. python -m pytest -q --rootdir=tests --confcutdir=tests \
  tests/test_rare_pr_comparison_ops.py tests/test_compare_rare_pr_reports_cli.py \
  tests/test_lvis_rare_pr.py tests/test_rare_pr_curve_replay.py \
  tests/test_run_rare_stage_comparison.py
```

## no-radius 8ep→10ep：排序退化的更新来源审计（有界、单卡、不更新参数）

`audit_rare_stage_updates.py` 接上面的完整 `comparison.json`。它不是新的训练实验，
也不声称能用两个 checkpoint 还原中间的 AdamW 历史。默认从报告的 focus 类别中选择
**AP、AP50 都下降且 IoU=.50 TP 数保持不变**的前三类；本次结果应是
`lasagna`、`keg`、`bass_horn`。不把仅 AP75 下降的 `chocolate_mousse` 当作 AP50 排序退化。

执行顺序与硬预算：

1. 核对 8ep/10ep APr、checkpoint iteration=56799/70999、K=5、radius=0、slot prior=.2；
   锁定所有输入的 SHA256。用已有全类预测在 CPU 重放这三类的官方 LVIS 匹配，取得新模型
   TP 和前置 FP 的真实身份。不是重跑全验证集 GPU inference。
2. 仅对相关图像分别执行两端原生 forward，**最多 32 张不同验证图/端**，超预算先报错。
   按框位置和分数恢复来源 query；另一端按 IoU≥.5 配对，不假定 query index 一致。
   无合格对应框/来源不唯一的区域列入 excluded，不硬凑配对。
3. 固定缓存的 query，交叉使用 8ep/10ep 的末端 TPA bank，分解融合 log-score 的变化：
   `terminal_tpa_bank`、`query_and_bias_path`、`clip_roi_path`，检查逐区域加法闭合。
   使用两种替换顺序的平均来分配交互项，不称作独立训练因果贡献。
4. 分别在两端用 `train_norare` 做 2 个窗口×8 microbatches×4 张的原生损失梯度探针，
   **两端合计最多 128 次训练图像使用**。只求 TPA 参数的检测损失、RPSA、APR 梯度，
   先平均再做原有冲突投影/TPA clipping。不创建 optimizer/scheduler，不做参数更新，
   不修改原 checkpoint；验证图像标签只标记诊断 TP/FP，绝不作为训练损失输入。
5. CPU 方向导数报告各方向如何改变这些固定区域的 `mean(TP log-score)−mean(FP log-score)`。
   同时报告 `cos(-g, TPA_10ep−TPA_8ep)`，仅作为局部方向与总漂移的一致性证据。

同步代码后在有 `lvis`、已编译 detrex 的 **lami 环境**运行：

```bash
cd ~/LaMI-DETR
mkdir -p /root/autodl-tmp/no_radius_8ep_vs_10ep_updates
set -o pipefail
CUDA_VISIBLE_DEVICES=0 /root/miniconda3/envs/lami/bin/python -u tools/audit_rare_stage_updates.py \
  --comparison /root/autodl-tmp/no_radius_8ep_vs_10ep_pr/comparison.json \
  --old-checkpoint /root/autodl-tmp/instructdet_k5_no_radius_bs32_4ep_seed42/model_0056799.pth \
  --new-checkpoint /root/autodl-tmp/instructdet_k5_no_radius_bs32_4ep_seed42/model_0070999.pth \
  --old-predictions /root/autodl-tmp/eval_k5_no_radius_bs32_8ep/lvis_instances_results.json \
  --new-predictions /root/autodl-tmp/eval_k5_no_radius_bs32_10ep/lvis_instances_results.json \
  --output-dir /root/autodl-tmp/no_radius_8ep_vs_10ep_updates \
  2>&1 | tee -a /root/autodl-tmp/no_radius_8ep_vs_10ep_updates/audit.log
```

如只想先知道图像数量，在该命令加 `--prepare-only`：只做 CPU 预检/匹配，之后去掉此参数
原命令重跑。完整执行会缓存原生区域和两端梯度；同一输入/配置重跑时校验后复用，不重复
已完成的 GPU 探针。输入、模型代码、采样设置不一致则拒绝混用。`--classes 1` 可进一步
减少目标类别，但要使用新的输出目录；不根据这三类结果调每类阈值。

输出 `report.json`（完成时 `complete=true`），以及可复查的 `fp_details.json`、
`regions.json`、两端 geometry/gradient 文件、原生 query 缓存。日志包含逐文件 hash、
逐图 dump、逐 microbatch 和逐 JVP 进度；耗时受读取两份预测 JSON、模型加载和硬件影响。

结果解释：

- `terminal_tpa_bank` 的 margin 变化若明显为负，支持检查末端分类 bank 漂移；
  若主要在 `query_and_bias_path`，不能据此称“TPA 无关”，因为它还包含上游 TPA query
  fusion、detector 表示及配对框变化。`clip_roi_path` 包含 ROI 位置变化。
- `dMargin<0` 表示该**局部假想梯度下降**会缩小选中 TP/FP 分数差；正值相反。
  各分量使用 routed-total 的同一个 clip 系数，检查加法闭合；未乘 LR，未应用 AdamW
  动量、二阶矩或 weight decay，不能当作真实训练步的分数变化。
- 相同 seed 不等于相同 probe batch。报告逐 microbatch 核对 image ID、增强后像素/GT
  SHA256、FedLoss 子集；`all_verified_equal=false` 时不得把两端梯度差当成严格配对变化。
  即便为 true，也未证明内部 dropout/denoising 随机数一致，更不是历史四卡采样重放。
- TP/FP 标签来自 **10ep 的已选区域**，携带到 8ep 的几何对应区域；并非两端重新匹配后
  同一个 TP/FP。报告不是整条 PR/AP，不是全部 −0.5812 APr 的归因，也不能证明某项
  训练修改导致了它。只有一致的证据才值得考虑下一项受控训练消融。

本地测试覆盖数值差分、加法闭合、checkpoint/预算/缓存保护、几何 query 配对、输入
哈希核验和无 optimizer 调用；真实 GPU/D2 损失捕获仍须在服务器环境验证：

```bash
PYTHONPATH=. python -m pytest -q --rootdir=tests --confcutdir=tests \
  tests/test_rare_stage_updates.py tests/test_tpa_gradient_audit.py \
  tests/test_tpa_geometry_audit.py tests/test_rare_region_pairing.py
```

### CPU JVP 中断后恢复（PyTorch 1.12.1 / no-radius）

若 GPU 梯度已保存，却在 `[CPU JVP] old` 报 `Nonfinite text-side JVP`，先保留整个
输出目录。no-radius 下中心位移恒为零；旧审计仍对它执行 `norm(0)` 的双反向 JVP。
[PyTorch 1.12.1 的 norm backward](https://github.com/pytorch/pytorch/blob/v1.12.1/torch/csrc/autograd/FunctionsManual.cpp#L193-L214)
先除以 norm 再 mask，在这种双反向路径可能产生 NaN。它是审计数值问题，不能直接
判定 checkpoint/训练梯度非有限。本地测试按该版本的 norm backward 复现旧故障，
再检查修复后 JVP 与中心差分一致。

修复仅在诊断代码中：radius=0 时用精确恒等分支，恒零位移报告精确零导数；区域排序
审计仅请求 logits JVP，不为无关几何量求导。没有 epsilon 扰动、`nan_to_num`、训练
参数/损失修改；真正非有限的数值仍会中止，并指出是哪块输出或导数。

同步修复提交后直接执行，不必再传 checkpoint 或预测路径：

```bash
cd ~/LaMI-DETR
set -o pipefail
CUDA_VISIBLE_DEVICES="" /root/miniconda3/envs/lami/bin/python -u tools/audit_rare_stage_updates.py \
  --resume-cpu \
  --output-dir /root/autodl-tmp/no_radius_8ep_vs_10ep_updates \
  2>&1 | tee -a /root/autodl-tmp/no_radius_8ep_vs_10ep_updates/jvp_resume.log
```

`--resume-cpu` 从已锁定 manifest 恢复输入/探针设置，校验 checkpoint、prompt、
区域报告、梯度及训练标注身份，仅在 CPU 重算分析。不会重新跑 LVIS 匹配、原生图像
forward、训练梯度捕获或 geometry 报告；不改捕获指纹/训练代码。缺缓存、输入变化或
指纹不符时直接报错，**绝不自动回退到 GPU**。完成后上传同目录 `report.json`
（`complete=true`）；若仍失败，上传 `jvp_resume.log`，不要删除或强行重写缓存。

### Query 路径实际参数变化：8ep/10ep 双向模块替换

`audit_query_path_updates.py` 接在**完整**的 `audit_rare_stage_updates.py` 输出后运行。
它补查 TPA-only 梯度探针没有覆盖的 query 生成路径，**不做训练、不计算梯度、不创建
optimizer，不修改 checkpoint 文件**。这里的“实际更新”指两端 checkpoint 中真实的
参数差，而不是重放 AdamW 中间步骤，更不能直接归因于某项 loss 或训练修改。

当前报告原选 6 张图，排除无法配对的区域后只用 **5 张图、10 个区域**：需要一次单卡小面板前向，不能仅靠已保存的最终 query 特征
在 CPU 重建 encoder/decoder 改变后的输出。默认硬上限为 8 张图、两端合计 192 次
单图 forward；8 个模块均有变化时，本次 5 张图是 100 次（含两端原生和联合对照）。
**不重跑 19,809 张全验证集，不做任何 train_norare 探针**。耗时取决于模型加载和单图
前向速度；预检会打印预算，每张图会打印进度，可中断后原命令恢复。

模块分组：

| 输出名称 | 替换的参数 |
|---|---|
| `visual_adapter` | backbone 输出 norm、neck、位置编码（CLIP trunk 必须完全一致） |
| `encoder_memory` | encoder、level embeddings、encoder output projection/norm |
| `proposal_head` | encoder 用的分类 projection/bias 与 box head |
| `query_content` | `content_layer`：原型到 query 初始化空间的投影 |
| `decoder_core` | decoder attention/FFN、norm、reference-point position head |
| `decoder_box` | decoder 各层 box refinement，可能改变后续注意力采样位置 |
| `final_projection` | 最后一个 decoder classifier 的 feature projection |
| `upstream_tpa` | 共享 TPA 权重在 encoder 类别选择/query fusion 路径中的作用 |

`final_bias_only` 单独在 CPU 计算，无需额外 forward；`all_query` 联合替换以上八组，
用于对照非线性交互，**各模块效果不能相加解释整体变化**。训练专用、辅助分类头、
未改变的张量也列在 inventory 中，不把它们悄悄混入其他模块。未知张量、CLIP trunk
或协议 buffer 若改变则预检失败。共享 TPA/classifier/bbox 的所有 alias 同步替换并
恢复，原型 eval cache 每次清空，防止旧缓存让替换虚假“无效”。

每组双向执行：8ep 模型中放入 10ep 模块；10ep 模型中换回 8ep 模块。测量时：

- 最终分类 bank、最终 scalar bias 和 CLIP 分数固定在接收端；因此 `upstream_tpa`
  的最终分类 bank 变化不混入测量。偏置效果单列。
- 原生 forward 必须复现原缓存的 features/boxes/prototypes，否则立即停止。
- 使用原报告的 TP/FP 区域；新产生的候选按框 IoU 做**不看分数**的一对一匹配，不假定
  query index 恒定，不根据新模型分数挑选更有利的候选。重复来源 query 可复用同一
  匹配；不同 query 不共享对应候选。几何并列不强行配对。
- 输出每个区域的匹配 IoU、特征 cosine、融合 log-score 变化，以及类别 TP−FP 间隔
  变化。只在全部区域匹配成功时输出 `full_panel_margin_change`；缺失/歧义框显示
  `NA`，不能算成“FP 被修好了”。子集统计保留供检查，不能和完整面板直接比较。
- 这不是锁死网络内部框的位置：box refinement 可以改变特征和候选位置，最终通过
  几何对应区域进行条件评分，且不重算 CLIP 项。它不是新推理协议或可部署 gate。

服务器同步本次提交后运行（固定 lami Python，避免 base 环境 detrex 错误）：

```bash
cd ~/LaMI-DETR
mkdir -p /root/autodl-tmp/no_radius_8ep_vs_10ep_query_updates
set -o pipefail
CUDA_VISIBLE_DEVICES=0 /root/miniconda3/envs/lami/bin/python -u tools/audit_query_path_updates.py \
  --stage-dir /root/autodl-tmp/no_radius_8ep_vs_10ep_updates \
  --output-dir /root/autodl-tmp/no_radius_8ep_vs_10ep_query_updates \
  2>&1 | tee -a /root/autodl-tmp/no_radius_8ep_vs_10ep_query_updates/audit.log
```

`--prepare-only` 仅做 CPU 输入/参数分组/预算预检；`--analyze-only` 仅分析新工具已经
完成的 forward cache，缺失时拒绝运行，绝不回退到 GPU。恢复时保留原 `--device`
字符串（它是缓存身份的一部分）；可设置 `CUDA_VISIBLE_DEVICES=""`，证明没有 GPU
调用。普通重跑同样自动复用已完成的逐图缓存。身份锁包含源 report、checkpoint、
模型/配置/诊断代码、资源、PyTorch 版本和匹配协议；不覆盖旧 stage 的任何文件。

完成后上传新目录的 `report.json`。优先看 `bidirectional_check` 和区域覆盖率：
“8ep 加入新模块使间隔变小，10ep 换回旧模块使间隔变大”支持该模块的**端点局部负作用**；
只在一个方向改善、或两端混合都变差，可能是模块协同适配/混合模型分布外行为，不能
马上冻结或删掉该模块。面板只有事后选定的几类/几个区域，不代表全验证集 APr，也不
证明是 APR、RPSA 或某一种历史优化更新造成的。不自动启动消融训练或全量评测。

本地 CPU 回归：

```bash
PYTHONPATH=. python -m pytest -q --rootdir=tests --confcutdir=tests \
  tests/test_query_path_updates.py tests/test_rare_stage_updates.py \
  tests/test_tpa_gradient_audit.py tests/test_rare_region_pairing.py
```
