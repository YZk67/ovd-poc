# Scarecrow 8→12ep：只定位退化发生的采样区间

输入是已完成的 **`trace_rare_gt_queries.py` 输出**：
`/root/autodl-tmp/no_radius_scarecrow_queries_v2/report.json`。
不是后续 `analyze_rare_query_readout.py` 的交叉 bank/bias 报告。

范围固定：`image_id=218917`、`scarecrow GT=137708`、同一次 no-radius K5 训练。
8ep/12ep 必须复用源报告记录的原始缓存；不会再次前向这两个端点。

| 阶段 | iteration（零基） | 权重 | 行为 |
| --- | ---: | --- | --- |
| 8ep | 56799 | 源报告已认证的端点 | CPU 复现缓存并验证原生 top-300 |
| 9ep | 63899 | `model_0063899.pth` | 有权重才检查；优先缓存，否则一次单图前向 |
| 10ep | 70999 | `model_0070999.pth` | 同上 |
| 11ep | 78099 | `model_0078099.pth` | 同上 |
| 12ep | 85199 | 源报告已认证的端点 | CPU 复现缓存并验证原生 top-300 |

这里沿用该实验的 7100 optimizer updates/阶段标记，不重新推断全局 batch 或训练曝光量。
checkpoint 目录默认取源报告中旧/新权重所在的共同目录。

## 服务器一次运行

在正确的 `lami` 环境执行。只需要一张 GPU。不要预先创建输出目录；日志写在目录旁边。

```bash
cd ~/LaMI-DETR
set -o pipefail
CUDA_VISIBLE_DEVICES=0 /root/miniconda3/envs/lami/bin/python -u \
  tools/trace_rare_gt_timeline.py \
  --source-json /root/autodl-tmp/no_radius_scarecrow_queries_v2/report.json \
  --cache-search-root /root/autodl-tmp \
  --output-dir /root/autodl-tmp/no_radius_scarecrow_timeline \
  2>&1 | tee /root/autodl-tmp/no_radius_scarecrow_timeline.log
```

在另一个终端看日志：

```bash
tail -n 60 -F /root/autodl-tmp/no_radius_scarecrow_timeline.log
```

最后上传 **`/root/autodl-tmp/no_radius_scarecrow_timeline/report.json`**。

最多 **3 次单图前向，0 次训练更新，0 次全验证集推理/评估**。初始化模型、加载约1GB
权重和验证集索引可能占主要时间，因此不能把“3次前向”解释为只需几秒。
不会创建新的训练 checkpoint，也不修改任何源权重、预测或缓存。

## 缓存、缺失和恢复

- 先验证两端源缓存的 SHA256、完整900×1203分数、768D特征/原型、协议和几何设置，
  再 CPU 复现全部300个端点预测。端点缓存缺失或不一致直接停止，绝不自动重跑。
- 中间权重必须通过真实 `iteration`、K5、显式 no-radius buffer、`slot_prior=.2` 和
  共享TPA参数校验。文件名相同但内容不同不能混用；这仍不能从权重反推出完整历史训练配置。
- 搜索已存在的匹配缓存时要求 checkpoint SHA256、模型代码、外部文本/模型资产、
  标注、推理参数全部相同。`--cache-search-root` 仅扫描 `*/pairing_cache/manifest.json`
  和 `*/cache/manifest.json`，不递归扫描大量图像分片。也可用 `--reuse-cache` 指定目录。
- 只查表中三个中间权重文件。缺失会打印 `[missing]` 并在结果中记下；不拿邻近或另一轮
  权重替代，不重训补点。如果都缺失，仅返回原8→12ep区间并标注 `endpoints_only_not_narrowed`。
- 中间点是原生模型前向加 dense replay 检查，没有独立的完整验证集预测JSON时，不声称
  复现了该阶段完整验证集结果。所有900个query与1203类参与原生top-300，之后才分析此GT。
- `--prepare-only` 做身份/缓存/权重/预算预检，保存 `plan.json`，不前向。
  去掉该参数再次运行即可。`--cache-only` 拒绝任何缺失缓存的前向。
- 相同输入和代码下直接重试原指令即可复用已成功生成的中间缓存；如模型资产、代码或
  可用权重集合改变，会拒绝混用旧报告，需新输出目录。之前的缓存仍可显式指定复用。
- 若训练目录整体搬迁，可用 `--checkpoint-dir`；搬迁后的8ep与12ep权重哈希也必须匹配
  原报告。若仅标注搬迁，用 `--annotations` 指定字节相同的JSON。源缓存路径必须仍可访问。

## 结果如何解释

在 IoU=0.50/0.75/0.90 分别报告：

- 每个阶段合格最终query数量、正确类别进入top-300的数量。
- 合格query中**真类融合分数最高者**的框IoU、detector/CLIP分数与类内排名、全图pair排名、
  原生top-300门槛和分数/门槛比值；同时保存完整900query分析，避免只挑best-IoU框。
- 按真实迭代顺序列出全部“保留→丢失”和“丢失→恢复”，不假设性能单调，不二分搜索。
- 首次观测到的丢失区间，以及最终漏检之前最后一次观测到的丢失区间。
  两者可能不同。例如8保留、9丢失、10恢复、11丢失、12丢失，会分别给出8→9和10→11，
  而不会把8→12标成持续退化。
- 中间缺点会扩大区间。例如10保留、11缺失、12丢失，只能定位到10→12。
  分数距门槛≤1e-5时标为 `boundary_sensitive`，不能把数值边界变动当成稳健退化证据。

这些区间只表示**采样端点间的候选覆盖状态不同**，不证明区间内从未恢复，更不能定位
某个精确optimizer update或损失来源。query编号属于各checkpoint，不跨端点强行配对。
`no_eligible_box`只涉及最终pre-top300 query，不代表encoder没有proposal。
保留正确候选不是正式LVIS一对一TP或AP；这一张事后选中的图不能代表整体rare类别。
脚本不会自动启动梯度审计、屏蔽loss、修改LR/APR或继续训练。

本地CPU测试：

```bash
cd tests
PYTHONPATH=.. python -m pytest --rootdir=. --confcutdir=. -q \
  test_rare_gt_timeline.py test_rare_gt_queries.py test_rare_query_readout.py
```

CPU测试包含真实768D缓存复现，以及模拟的3次单图预算/失败恢复/缺失权重/非单调时间线。
不替代服务器PyTorch1.12/CUDA/Detrex的实际前向验证。
