# 融合常数扫描：8ep / 12ep / kang，零训练

目的：同一份权重上离线扫 `beta`（novel 类的 CLIP 权重）× `novel_scale`，回答
"现在固定的推理常数 0.3 / 3 是否随训练漂移"。scarecrow 的算术已经指向这一点：
12ep 那条检测在 beta=0.3 下融合分 0.373 低于门槛 0.387，同一候选在 beta=0.5 下约 0.68。
这一步只定位差距来源，不改训练、不改 checkpoint。

## 网格

- `alpha` 固定 0（当前协议和 EXPERIMENT_LOCK 都是 0）。
- `beta ∈ {0.3, 0.4, 0.5, 0.6}`，`novel_scale ∈ {3, 5}`，共 8 格，profile 名
  `power_beta{beta}_scale{scale}`。其中 `power_beta0.3_scale3` 等于当前协议，
  `power_beta0.4_scale5` 等于 EXPERIMENT_LOCK 写的协议。
- 网格写进 dump 的 `manifest.json`，与命令行的 `model.beta` / `model.novel_scale`
  无关；`current_power` 仍按命令行覆盖的值，用来对照官方评测。
- 想换网格用 `--sweep-betas 0.3,0.4 --sweep-scales 3,4,5`（逗号分隔，不要用空格，
  否则会吞掉后面的 `model.xxx=` 覆盖）。

dump 只保存每个 profile 的 top-300 候选并集，所以每一格的离线结果都是精确的，
不是近似。三份权重各 dump 一次，之后所有评测都是 CPU。

## 服务器命令

先同步提交。目录不要预先创建 dump 子目录；日志放在外面。

```bash
cd ~/LaMI-DETR
mkdir -p /root/autodl-tmp/fusion_sweep
set -o pipefail
LAMI=/root/miniconda3/envs/lami/bin/python
PROTO="model.alpha=0.0 model.beta=0.3 model.novel_scale=3.0 model.tpa_eval_mode_scale=1.0"

# 12ep（no-radius，iteration 85199）
CUDA_VISIBLE_DEVICES=0,1,2,3 $LAMI -u tools/dump_ovd_raw_scores.py \
  --config-file lami_dino/configs/dino_convnext_large_4scale_12ep_lvis_no_radius.py \
  --checkpoint /root/autodl-tmp/instructdet_k5_no_radius_bs32_4ep_seed42/model_final.pth \
  --output-dir /root/autodl-tmp/fusion_sweep/no_radius_12ep \
  --num-gpus 4 $PROTO \
  2>&1 | tee /root/autodl-tmp/fusion_sweep/no_radius_12ep_dump.log

# 8ep（no-radius，iteration 56799）
CUDA_VISIBLE_DEVICES=0,1,2,3 $LAMI -u tools/dump_ovd_raw_scores.py \
  --config-file lami_dino/configs/dino_convnext_large_4scale_12ep_lvis_no_radius.py \
  --checkpoint /root/autodl-tmp/instructdet_k5_no_radius_bs32_4ep_seed42/model_0056799.pth \
  --output-dir /root/autodl-tmp/fusion_sweep/no_radius_8ep \
  --num-gpus 4 $PROTO \
  2>&1 | tee /root/autodl-tmp/fusion_sweep/no_radius_8ep_dump.log

# kang（released 权重，用它自己的 eval 配置，融合常数覆盖成当前协议）
CUDA_VISIBLE_DEVICES=0,1,2,3 $LAMI -u tools/dump_ovd_raw_scores.py \
  --config-file lami_dino/configs/dino_convnext_large_4scale_12ep_lvis_kang_eq2_eval.py \
  --checkpoint /root/autodl-tmp/model_final_ovd_lvis_kang.pth \
  --output-dir /root/autodl-tmp/fusion_sweep/kang \
  --num-gpus 4 $PROTO \
  2>&1 | tee /root/autodl-tmp/fusion_sweep/kang_dump.log
```

中断后加 `--resume` 重跑同一条命令即可续上。每份 dump 是全部 19,809 张验证图，
磁盘按几百 MB 到 1 到 2 GB 预留。

评测：第一个 profile 固定是 `current_power`，它必须复现官方 APr，否则脚本在保存
报告后报错退出，其余格子的数字不能用。

```bash
cd ~/LaMI-DETR
set -o pipefail
LAMI=/root/miniconda3/envs/lami/bin/python

$LAMI -u tools/evaluate_ovd_fusion.py \
  --dump-dir /root/autodl-tmp/fusion_sweep/no_radius_12ep \
  --profiles current_power --profile-prefix power_beta \
  --expected-current-apr 42.4229 \
  --output /root/autodl-tmp/fusion_sweep/no_radius_12ep_metrics.json --resume \
  2>&1 | tee /root/autodl-tmp/fusion_sweep/no_radius_12ep_eval.log

$LAMI -u tools/evaluate_ovd_fusion.py \
  --dump-dir /root/autodl-tmp/fusion_sweep/no_radius_8ep \
  --profiles current_power --profile-prefix power_beta \
  --expected-current-apr 42.8843 \
  --output /root/autodl-tmp/fusion_sweep/no_radius_8ep_metrics.json --resume \
  2>&1 | tee /root/autodl-tmp/fusion_sweep/no_radius_8ep_eval.log

$LAMI -u tools/evaluate_ovd_fusion.py \
  --dump-dir /root/autodl-tmp/fusion_sweep/kang \
  --profiles current_power --profile-prefix power_beta \
  --expected-current-apr 45.2037 \
  --output /root/autodl-tmp/fusion_sweep/kang_metrics.json --resume \
  2>&1 | tee /root/autodl-tmp/fusion_sweep/kang_eval.log
```

每个 profile 一次官方 LVIS 评测，纯 CPU，每次几分钟；三份各 9 个 profile。
`--resume` 让中断后已算完的 profile 不重复。结束时打印 beta × scale 网格
（每格 AP / APr）和网格内 APr 最高的一格相对 `current_power` 的差。

kang 的 `current_power` 若复现不了 45.2037，说明历史报告用的协议不同；这时不要
放宽 `--apr-tolerance`，把 kang 这份当作自身基线，只在它自己的网格内比较。

## 读法，事先定好

- 对每份权重看网格内 APr 最高的一格在哪里，以及那一格的 AP 代价。
- 12ep 在 `power_beta0.4_scale5` 或更高 beta 下 APr 比 `current_power` 高 1.0 以上、
  AP 掉不到 0.3：校准漂移成立，后半段训练的 rare 损失有相当一部分是融合常数没跟上。
- 8ep 与 12ep 的最优格相同：不是漂移，差距在表示本身。
- kang 的 APr 在每一格都比 12ep 高 2 以上：与 kang 的差距不是校准，要从训练配方找。
  kang 的优势在高 beta 格缩小到 1 以内：差距主要是校准。
- 这些格子都是在 val 上扫的，只用来定位。能进论文的只有事先锁定的
  `power_beta0.4_scale5` 那一格，它是 EXPERIMENT_LOCK 写明的协议，不算按 val 调参。

上传三份 `*_metrics.json` 即可，dump 目录不用传。

## 本地回归

```bash
PYTHONPATH=. python -m pytest -q --rootdir=tests --confcutdir=tests \
  tests/test_fusion_sweep_profiles.py tests/test_diagnostic_ops.py
```

CPU 测试覆盖网格构造、候选并集对每一格精确、profile 筛选、官方 APr 复现检查和
网格打印；真实 dump / LVIS 评测仍要在服务器验证。
