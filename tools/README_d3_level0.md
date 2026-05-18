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

紧邻 ablation 1：只保留原始 phrase + 第一条 anchored 改写，检查第 3 条 fallback prompt 是否真的有贡献：

```bash
python tools/prepare_d3_description_prompts.py \
  --phrases-json dataset/metadata/d3_phrases.json \
  --output dataset/metadata/d3_description_anchor2_prompts.json \
  --preset anchored \
  --prompt-count 2

python tools/generate_text_embeddings.py \
  --prompt-json dataset/metadata/d3_description_anchor2_prompts.json \
  --output dataset/metadata/d3_description_anchor2_bank_convnextl.npy \
  --aggregate none \
  --batch-size 128 \
  --normalize

python tools/train_net.py \
  --config-file lami_dino/configs/dino_convnext_large_4scale_12ep_lvis_zeroshot_d3_desc_anchor2_mean_cls_only.py \
  --eval-only
```

紧邻 ablation 2：使用同一个 3-prompt anchored bank，把 classifier 聚合从 `mean` 换成 `logsumexp`：

```bash
python tools/train_net.py \
  --config-file lami_dino/configs/dino_convnext_large_4scale_12ep_lvis_zeroshot_d3_desc_anchor_logsumexp_cls_only.py \
  --eval-only
```

紧邻 ablation 3：只保留原始 phrase + fallback-style anchor prompt，不使用 heuristic paraphrase：

```bash
python tools/prepare_d3_description_prompts.py \
  --phrases-json dataset/metadata/d3_phrases.json \
  --output dataset/metadata/d3_description_anchor_fallback_prompts.json \
  --preset fallback \
  --prompt-count 3

python tools/generate_text_embeddings.py \
  --prompt-json dataset/metadata/d3_description_anchor_fallback_prompts.json \
  --output dataset/metadata/d3_description_anchor_fallback_bank_convnextl.npy \
  --aggregate none \
  --batch-size 128 \
  --normalize

python tools/train_net.py \
  --config-file lami_dino/configs/dino_convnext_large_4scale_12ep_lvis_zeroshot_d3_desc_anchor_fallback_mean_cls_only.py \
  --eval-only
```

紧邻 ablation 4：把两个 fallback prompt 拆开，分别测试 `the described target is ...` 和 `a visible region matching ...`：

```bash
python tools/prepare_d3_description_prompts.py \
  --phrases-json dataset/metadata/d3_phrases.json \
  --output dataset/metadata/d3_description_anchor_target_prompts.json \
  --preset fallback_target \
  --prompt-count 2

python tools/generate_text_embeddings.py \
  --prompt-json dataset/metadata/d3_description_anchor_target_prompts.json \
  --output dataset/metadata/d3_description_anchor_target_bank_convnextl.npy \
  --aggregate none \
  --batch-size 128 \
  --normalize

python tools/train_net.py \
  --config-file lami_dino/configs/dino_convnext_large_4scale_12ep_lvis_zeroshot_d3_desc_anchor_target_mean_cls_only.py \
  --eval-only

python tools/prepare_d3_description_prompts.py \
  --phrases-json dataset/metadata/d3_phrases.json \
  --output dataset/metadata/d3_description_anchor_region_prompts.json \
  --preset fallback_region \
  --prompt-count 2

python tools/generate_text_embeddings.py \
  --prompt-json dataset/metadata/d3_description_anchor_region_prompts.json \
  --output dataset/metadata/d3_description_anchor_region_bank_convnextl.npy \
  --aggregate none \
  --batch-size 128 \
  --normalize

python tools/train_net.py \
  --config-file lami_dino/configs/dino_convnext_large_4scale_12ep_lvis_zeroshot_d3_desc_anchor_region_mean_cls_only.py \
  --eval-only
```

紧邻 ablation 5：固定 `phrase` 和 `the described target is {phrase}` 两个 embedding，直接合成单 prototype 并扫描 target prompt 权重：

```bash
python tools/mix_text_prototype_weights.py \
  --input dataset/metadata/d3_description_anchor_target_bank_convnextl.npy \
  --weights 0.25 0.5 0.75 1.0
```

这会生成：

```text
dataset/metadata/d3_description_anchor_target_w025_bank_convnextl.npy
dataset/metadata/d3_description_anchor_target_w050_bank_convnextl.npy
dataset/metadata/d3_description_anchor_target_w075_bank_convnextl.npy
dataset/metadata/d3_description_anchor_target_w100_bank_convnextl.npy
```

每个输出都是单 prototype bank，shape 为 `(422, 768)`。评测时用同一个 weighted config，并通过 override 指定不同 bank 和 output dir：

```bash
python tools/train_net.py \
  --config-file lami_dino/configs/dino_convnext_large_4scale_12ep_lvis_zeroshot_d3_desc_anchor_target_weighted_cls_only.py \
  --eval-only \
  model.classifier.zs_weight_path=dataset/metadata/d3_description_anchor_target_w025_bank_convnextl.npy \
  model.classifier.eval_zs_weight_path=dataset/metadata/d3_description_anchor_target_w025_bank_convnextl.npy \
  model.classifier.text_embed_path=dataset/metadata/d3_description_anchor_target_w025_bank_convnextl.npy \
  model.classifier.eval_text_embed_path=dataset/metadata/d3_description_anchor_target_w025_bank_convnextl.npy \
  train.output_dir=./output/lami_convnext_large_12ep_lvis_zeroshot_d3_desc_anchor_target_w025_cls_only \
  dataloader.evaluator.output_dir=./output/lami_convnext_large_12ep_lvis_zeroshot_d3_desc_anchor_target_w025_cls_only
```

## 6. Crop-level description rerank pilot

如果 `w075` 是当前最优 first-stage detector，可以直接复用它保存的 COCO predictions 做离线 crop rerank，不需要重新跑 detector：

```bash
python tools/rerank_d3_predictions_with_clip_crops.py \
  --predictions output/lami_convnext_large_12ep_lvis_zeroshot_d3_desc_anchor_target_w075_cls_only/coco_instances_results.json \
  --annotation dataset/d3/annotations/d3_intra_full.json \
  --image-root dataset/d3/images \
  --phrases-json dataset/metadata/d3_phrases.json \
  --output output/d3_crop_rerank_w075_top50_l025_m025.json \
  --rerank-topk-per-image 50 \
  --keep-topk-per-image 100 \
  --crop-margin 0.25 \
  --fusion logit_add \
  --fusion-weight 0.25 \
  --clip-scale 10.0 \
  --clip-center 0.25 \
  --eval
```

更严谨的检查是只在 verifier held-out val images 上比较 detector-only 和 verifier rerank，避免 verifier 训练图像泄漏影响结论：

```bash
python tools/rerank_d3_predictions_with_clip_crops.py \
  --predictions output/lami_convnext_large_12ep_lvis_zeroshot_d3_desc_anchor_target_w075_cls_only/coco_instances_results.json \
  --annotation dataset/d3/annotations/d3_intra_full.json \
  --image-root dataset/d3/images \
  --phrases-json dataset/metadata/d3_phrases.json \
  --image-id-jsonl dataset/d3/verifier_pairs_w075/val.jsonl \
  --output output/d3_verifier_val_detector_only.json \
  --keep-topk-per-image 100 \
  --skip-rerank \
  --eval

python tools/rerank_d3_predictions_with_clip_crops.py \
  --predictions output/lami_convnext_large_12ep_lvis_zeroshot_d3_desc_anchor_target_w075_cls_only/coco_instances_results.json \
  --annotation dataset/d3/annotations/d3_intra_full.json \
  --image-root dataset/d3/images \
  --phrases-json dataset/metadata/d3_phrases.json \
  --image-id-jsonl dataset/d3/verifier_pairs_w075/val.jsonl \
  --output output/d3_verifier_val_rerank_top20_vw025.json \
  --rerank-topk-per-image 20 \
  --keep-topk-per-image 100 \
  --crop-margin 0.1 \
  --verifier-checkpoint output/d3_crop_verifier_w075_balanced/verifier_best.pt \
  --verifier-fusion logit_add \
  --verifier-fusion-weight 0.25 \
  --eval
```

最后做 verifier feature ablation，证明收益不是只来自 detector score calibration 或 box-quality 判断。这里复用 full verifier 已经编码好的 feature cache，不重新跑 OpenCLIP：

```bash
python tools/train_d3_crop_verifier.py \
  --train-cache output/d3_crop_verifier_w075_balanced/cache/train_features.pt \
  --val-cache output/d3_crop_verifier_w075_balanced/cache/val_features.pt \
  --output-dir output/d3_crop_verifier_w075_no_text \
  --feature-mode no_text \
  --epochs 5 \
  --batch-size 512 \
  --lr 1e-3

python tools/train_d3_crop_verifier.py \
  --train-cache output/d3_crop_verifier_w075_balanced/cache/train_features.pt \
  --val-cache output/d3_crop_verifier_w075_balanced/cache/val_features.pt \
  --output-dir output/d3_crop_verifier_w075_no_score \
  --feature-mode no_detector_score \
  --epochs 5 \
  --batch-size 512 \
  --lr 1e-3
```

预期 `no_text` 在 `wrong_phrase_same_region` 上应接近随机，因为同一个 crop 配正确 phrase 和错误 phrase 时，去掉 text 后特征几乎相同；`no_detector_score` 如果还能保持较高 AUC/AP，说明 verifier 不是只靠 detector score。

建议先用 `--max-images 100` 做 smoke test，确认图片路径和 OpenCLIP 环境正常：

```bash
python tools/rerank_d3_predictions_with_clip_crops.py \
  --predictions output/lami_convnext_large_12ep_lvis_zeroshot_d3_desc_anchor_target_w075_cls_only/coco_instances_results.json \
  --annotation dataset/d3/annotations/d3_intra_full.json \
  --image-root dataset/d3/images \
  --output /tmp/d3_crop_rerank_smoke.json \
  --rerank-topk-per-image 20 \
  --keep-topk-per-image 100 \
  --max-images 100
```

小范围参数趋势检查先只跑 500 张图。`--max-images` 会让 eval 自动限制在输出中的 image ids；先用 `--skip-rerank` 得到同一 500 张图上的 detector-only baseline：

```bash
python tools/rerank_d3_predictions_with_clip_crops.py \
  --predictions output/lami_convnext_large_12ep_lvis_zeroshot_d3_desc_anchor_target_w075_cls_only/coco_instances_results.json \
  --annotation dataset/d3/annotations/d3_intra_full.json \
  --image-root dataset/d3/images \
  --phrases-json dataset/metadata/d3_phrases.json \
  --output output/d3_crop_rerank_subset500_detector_only.json \
  --keep-topk-per-image 100 \
  --max-images 500 \
  --skip-rerank \
  --eval
```

然后只扫四个紧邻组合：

```bash
for margin in 0.1 0.5; do
  for weight in 0.1 0.5; do
    tag="m${margin/./}_l${weight/./}"
    python tools/rerank_d3_predictions_with_clip_crops.py \
      --predictions output/lami_convnext_large_12ep_lvis_zeroshot_d3_desc_anchor_target_w075_cls_only/coco_instances_results.json \
      --annotation dataset/d3/annotations/d3_intra_full.json \
      --image-root dataset/d3/images \
      --phrases-json dataset/metadata/d3_phrases.json \
      --output output/d3_crop_rerank_subset500_top20_${tag}.json \
      --rerank-topk-per-image 20 \
      --keep-topk-per-image 100 \
      --crop-margin "$margin" \
      --fusion logit_add \
      --fusion-weight "$weight" \
      --clip-scale 10.0 \
      --clip-center 0.25 \
      --max-images 500 \
      --eval
  done
done
```

这个阶段的核心观察不是最终 SOTA，而是 region-level text verification 是否能在 `w075 AP 9.16` 之上继续涨。若默认参数涨，下一步再扫：

```text
fusion-weight: 0.1, 0.25, 0.5
crop-margin:   0.1, 0.25, 0.5
rerank top-k:  20, 50, 100
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

## 7. 构建 description verifier 训练样本

如果要从 text prototype calibration 进入真正的 region-description verification，先把 first-stage detector predictions 和 D3 GT 匹配成二分类 pair：

```bash
python tools/build_d3_verifier_pairs.py \
  --predictions output/lami_convnext_large_12ep_lvis_zeroshot_d3_desc_anchor_target_w075_cls_only/coco_instances_results.json \
  --annotation dataset/d3/annotations/d3_intra_full.json \
  --phrases-json dataset/metadata/d3_phrases.json \
  --output-dir dataset/d3/verifier_pairs_w075 \
  --pred-topk-per-image 100 \
  --pos-iou-thresh 0.5 \
  --neg-iou-thresh 0.3 \
  --wrong-phrase-neg-per-pos 2
```

输出：

```text
dataset/d3/verifier_pairs_w075/train.jsonl
dataset/d3/verifier_pairs_w075/val.jsonl
dataset/d3/verifier_pairs_w075/all.jsonl
dataset/d3/verifier_pairs_w075/summary.json
```

样本语义：

```text
label=1: predicted box 和同 phrase GT IoU >= 0.5
label=0 same_phrase_bad_box: 同 phrase 但 box 没对上
label=0 wrong_phrase_same_region: 对上的 box 配另一个不匹配 phrase
```

这是后续训练 `Description-Conditioned Target Verifier` 的输入；先固定这个样本集，再训练 MLP 或 cross-attention verifier。

## 8. 训练 crop-description verifier

先不要接回 detector。先用 frozen OpenCLIP crop/text feature 训练一个小 MLP verifier，并看验证集上是否能区分：

```text
positive: region 和 phrase 匹配
same_phrase_bad_box: phrase 对，但 box 没对上
wrong_phrase_same_region: box 是真实目标，但 phrase 换成了不匹配描述
```

建议先跑一个小 smoke，确认缓存和训练链路正常：

```bash
python tools/train_d3_crop_verifier.py \
  --train-jsonl dataset/d3/verifier_pairs_w075/train.jsonl \
  --val-jsonl dataset/d3/verifier_pairs_w075/val.jsonl \
  --image-root dataset/d3/images \
  --output-dir output/d3_crop_verifier_w075_smoke \
  --cache-dir output/d3_crop_verifier_w075_smoke/cache \
  --same-phrase-neg-per-pos 1 \
  --wrong-phrase-neg-per-pos 2 \
  --max-train-samples 2000 \
  --max-val-samples 1000 \
  --crop-margin 0.1 \
  --epochs 1 \
  --batch-size 256
```

再跑完整 balanced 版本：

```bash
python tools/train_d3_crop_verifier.py \
  --train-jsonl dataset/d3/verifier_pairs_w075/train.jsonl \
  --val-jsonl dataset/d3/verifier_pairs_w075/val.jsonl \
  --image-root dataset/d3/images \
  --output-dir output/d3_crop_verifier_w075_balanced \
  --cache-dir output/d3_crop_verifier_w075_balanced/cache \
  --same-phrase-neg-per-pos 1 \
  --wrong-phrase-neg-per-pos 2 \
  --crop-margin 0.1 \
  --epochs 5 \
  --batch-size 512 \
  --image-batch-size 64 \
  --lr 1e-3
```

输出：

```text
output/d3_crop_verifier_w075_balanced/cache/train_features.pt
output/d3_crop_verifier_w075_balanced/cache/val_features.pt
output/d3_crop_verifier_w075_balanced/verifier_best.pt
output/d3_crop_verifier_w075_balanced/metrics.json
```

先重点看 `metrics.json` 里 `val.wrong_phrase_same_region.ap/auc`。如果这个子集明显高于随机，说明 verifier 确实学到了“同一个 region 是否满足目标描述”，再把 `verifier_best.pt` 接回 rerank。

接回离线 rerank 时，先还是只跑 500 张图扫融合权重。`--verifier-checkpoint` 会让脚本使用 verifier logit，而不是裸 CLIP cosine：

```bash
for weight in 0.25 0.5 1.0; do
  tag="vw${weight/./}"
  python tools/rerank_d3_predictions_with_clip_crops.py \
    --predictions output/lami_convnext_large_12ep_lvis_zeroshot_d3_desc_anchor_target_w075_cls_only/coco_instances_results.json \
    --annotation dataset/d3/annotations/d3_intra_full.json \
    --image-root dataset/d3/images \
    --phrases-json dataset/metadata/d3_phrases.json \
    --output output/d3_verifier_rerank_subset500_top20_${tag}.json \
    --rerank-topk-per-image 20 \
    --keep-topk-per-image 100 \
    --crop-margin 0.1 \
    --verifier-checkpoint output/d3_crop_verifier_w075_balanced/verifier_best.pt \
    --verifier-fusion logit_add \
    --verifier-fusion-weight "$weight" \
    --max-images 500 \
    --eval
done
```

如果 subset 上涨，再跑全量：

```bash
python tools/rerank_d3_predictions_with_clip_crops.py \
  --predictions output/lami_convnext_large_12ep_lvis_zeroshot_d3_desc_anchor_target_w075_cls_only/coco_instances_results.json \
  --annotation dataset/d3/annotations/d3_intra_full.json \
  --image-root dataset/d3/images \
  --phrases-json dataset/metadata/d3_phrases.json \
  --output output/d3_verifier_rerank_w075_top20_vw05.json \
  --rerank-topk-per-image 20 \
  --keep-topk-per-image 100 \
  --crop-margin 0.1 \
  --verifier-checkpoint output/d3_crop_verifier_w075_balanced/verifier_best.pt \
  --verifier-fusion logit_add \
  --verifier-fusion-weight 0.5 \
  --eval
```
