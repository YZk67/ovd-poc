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
export D3_INTER_FULL_JSON=/path/to/d3_inter_full.json
```

## 2. 导出 phrase 列表

如果你已经有官方 COCO-style annotation：

```bash
python tools/prepare_d3_metadata.py \
  --coco-json dataset/d3/annotations/d3_inter_full.json \
  --metadata-output dataset/metadata/d3_sentences.json \
  --phrases-output dataset/metadata/d3_phrases.json
```

如果你只有 `d3_pkl`，可以直接转成 inter-group FULL COCO JSON：

```bash
python tools/prepare_d3_metadata.py \
  --pkl-root /path/to/d3_pkl \
  --split full \
  --setting inter \
  --coco-output dataset/d3/annotations/d3_inter_full.json \
  --metadata-output dataset/metadata/d3_sentences.json \
  --phrases-output dataset/metadata/d3_phrases.json
```

同理可生成 PRES / ABS：

```bash
python tools/prepare_d3_metadata.py --pkl-root /path/to/d3_pkl --split pres --setting inter --coco-output dataset/d3/annotations/d3_inter_pres.json
python tools/prepare_d3_metadata.py --pkl-root /path/to/d3_pkl --split abs --setting inter --coco-output dataset/d3/annotations/d3_inter_abs.json
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

默认评测 split 是 `d3_inter_full`，输出目录是：

```text
output/lami_convnext_large_12ep_lvis_zeroshot_d3
```

这个阶段只验证最小闭环，不包含 D3 finetune、phrase-aware DN 或 alias aggregation。
