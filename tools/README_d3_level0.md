# D3 Level-0 接入

这个接入把 D3 的 `sent_id` 当成固定检测类别，先验证现有 LaMI-DETR checkpoint 能否在 D3 上走完整的 `box + sent_id -> COCO-style AP` 流水线。

## 1. 准备 D3 数据

官方下载后至少需要：

- `d3_images.zip` 解压后的图片目录
- `d3_json.zip` 的 COCO-style annotation，或 `d3_pkl.zip` 的 `sentences.pkl / annotations.pkl / images.pkl / groups.pkl`

默认注册路径是：

```text
dataset/d3/images
dataset/d3/annotations/d3_inter_full.json
dataset/d3/annotations/d3_inter_pres.json
dataset/d3/annotations/d3_inter_abs.json
dataset/d3/annotations/d3_intra_full.json
dataset/d3/annotations/d3_intra_pres.json
dataset/d3/annotations/d3_intra_abs.json
```

如果你的文件不放在这些位置，可以用环境变量覆盖：

```bash
export D3_IMAGE_ROOT=/path/to/d3_images
export D3_INTRA_FULL_JSON=/path/to/d3_intra_full.json
```

## 2. 导出 phrase 列表

如果你已经有官方 COCO-style annotation：

```bash
python tools/prepare_d3_metadata.py \
  --coco-json dataset/d3/annotations/d3_intra_full.json \
  --metadata-output dataset/metadata/d3_sentences.json \
  --phrases-output dataset/metadata/d3_phrases.json
```

如果你只有 `d3_pkl`，可以直接转成默认的 intra-group FULL COCO JSON：

```bash
python tools/prepare_d3_metadata.py \
  --pkl-root /path/to/d3_pkl \
  --split full \
  --setting intra \
  --coco-output dataset/d3/annotations/d3_intra_full.json \
  --metadata-output dataset/metadata/d3_sentences.json \
  --phrases-output dataset/metadata/d3_phrases.json
```

主表常用的 PRES / ABS 同样用 intra setting 生成：

```bash
python tools/prepare_d3_metadata.py --pkl-root /path/to/d3_pkl --split pres --setting intra --coco-output dataset/d3/annotations/d3_intra_pres.json
python tools/prepare_d3_metadata.py --pkl-root /path/to/d3_pkl --split abs --setting intra --coco-output dataset/d3/annotations/d3_intra_abs.json
```

## 3. 生成 D3 text embedding bank

```bash
python tools/generate_text_embeddings.py \
  --prompt-json dataset/metadata/d3_phrases.json \
  --clip-model pretrained_models/clip_convnext_large_head.pth \
  --output dataset/metadata/d3_clip_convnextl_sentences.npy \
  --aggregate mean \
  --normalize
```

期望输出 shape：

```text
(422, 768)
```

## 4. 跑 Level-0 eval-only

```bash
python tools/train_net.py \
  --config-file lami_dino/configs/dino_convnext_large_4scale_12ep_lvis_zeroshot_d3.py \
  --eval-only
```

默认评测 split 是 `d3_intra_full`，对应论文主表常用的 D3/DOD Full AP。输出目录是：

```text
output/lami_convnext_large_12ep_lvis_zeroshot_d3_full
```

如果要跑更难的 inter-scenario 压力测试，使用：

```bash
python tools/train_net.py \
  --config-file lami_dino/configs/dino_convnext_large_4scale_12ep_lvis_zeroshot_d3_inter_full.py \
  --eval-only
```

它会评测 `d3_inter_full`，输出到：

```text
output/lami_convnext_large_12ep_lvis_zeroshot_d3_inter_full
```

这个阶段只验证最小闭环，不包含 D3 finetune、phrase-aware DN 或 alias aggregation。

## 5. 跑 description-bank multi-prototype eval

先从 D3 phrase 列表生成每类 3 条 anchored description prompt。这个版本保留原始 phrase，并只加少量语义等价或弱锚定改写：

```bash
python tools/prepare_d3_description_prompts.py \
  --phrases-json dataset/metadata/d3_phrases.json \
  --output dataset/metadata/d3_description_anchor_prompts.json \
  --preset anchored \
  --prompt-count 3
```

再用同一个 ConvNeXt-L OpenCLIP text encoder 生成 3D bank：

```bash
python tools/generate_text_embeddings.py \
  --prompt-json dataset/metadata/d3_description_anchor_prompts.json \
  --output dataset/metadata/d3_description_anchor_bank_convnextl.npy \
  --aggregate none \
  --batch-size 128 \
  --normalize
```

期望输出 shape：

```text
(422, 3, 768)
```

然后跑 classifier-only multi-prototype mean 聚合版本。query init 和 VLM score ensemble 仍使用原始单句 phrase bank：

```bash
python tools/train_net.py \
  --config-file lami_dino/configs/dino_convnext_large_4scale_12ep_lvis_zeroshot_d3_desc_anchor_mean_cls_only.py \
  --eval-only
```

如果要复现旧的泛模板 max 聚合版本，先用 `--preset generic6` 生成：

```bash
python tools/prepare_d3_description_prompts.py \
  --phrases-json dataset/metadata/d3_phrases.json \
  --output dataset/metadata/d3_description_prompts.json \
  --preset generic6
```

再生成 `dataset/metadata/d3_description_bank_convnextl.npy` 并运行：

```bash
python tools/train_net.py \
  --config-file lami_dino/configs/dino_convnext_large_4scale_12ep_lvis_zeroshot_d3_desc_bank_max.py \
  --eval-only
```

这个配置默认仍评测 `d3_intra_full`，输出到：

```text
output/lami_convnext_large_12ep_lvis_zeroshot_d3_desc_bank_max
```
