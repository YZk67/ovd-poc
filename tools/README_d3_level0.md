# D3 Level-0 接入

这个接入把 D3 的 `sent_id` 当成固定检测类别，先验证现有 LaMI-DETR checkpoint 能否在 D3 上走完整的 `box + sent_id -> COCO-style AP` 流水线。

## 1. 准备 D3 数据

官方下载后至少需要：

- `d3_images.zip` 解压后的图片目录
- `d3_json.zip` 的 COCO-style annotation，或 `d3_pkl.zip` 的 `sentences.pkl / annotations.pkl / images.pkl / groups.pkl`

当前默认实验至少使用这些文件：

```text
dataset/d3/images
dataset/d3/annotations/d3_inter_full.json
dataset/d3/annotations/d3_intra_full.json
dataset/d3/d3_json/d3_full_annotations.json
dataset/d3/d3_json/d3_pres_annotations.json
dataset/d3/d3_json/d3_abs_annotations.json
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

`inter_pres` / `inter_abs` 不是当前默认主结果需要的 split。只有在明确要做
inter-setting 的 PRES / ABS 压力测试时，才从 `d3_pkl` 额外派生：

```bash
python tools/prepare_d3_metadata.py --pkl-root /path/to/d3_pkl --split pres --setting inter --coco-output dataset/d3/annotations/d3_inter_pres.json
python tools/prepare_d3_metadata.py --pkl-root /path/to/d3_pkl --split abs --setting inter --coco-output dataset/d3/annotations/d3_inter_abs.json
```

如果继续用 422 类 fixed-bank detector 评估这些派生 split，也要把
`categories` 补回 full 422 类，例如：

```bash
python tools/prepare_d3_subset_full_categories.py \
  --pres-json dataset/d3/annotations/d3_inter_pres.json \
  --abs-json dataset/d3/annotations/d3_inter_abs.json \
  --pres-output dataset/d3/annotations/d3_inter_pres_fullcats.json \
  --abs-output dataset/d3/annotations/d3_inter_abs_fullcats.json
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

## 9. Detector-internal no-score verifier

离线 rerank 验证有效后，可以把 `no_detector_score` verifier 接进 `DINO.forward()` 推理内部。这个版本不再裁图跑外部 OpenCLIP；它复用 detector 预测框上的 ConvNeXt/CLIP-head region feature，再和 target-prompt text feature 进入 verifier MLP。

先确认这些文件存在：

```bash
ls -lh output/d3_crop_verifier_w075_no_score/verifier_best.pt
ls -lh dataset/metadata/d3_description_anchor_target_w075_bank_convnextl.npy
ls -lh dataset/metadata/d3_description_anchor_target_w100_bank_convnextl.npy
ls -lh dataset/metadata/d3_seen_empty.json
```

如果 `d3_seen_empty.json` 不存在，创建一个空 seen-class 列表：

```bash
python - <<'PY'
import json
json.dump([], open('dataset/metadata/d3_seen_empty.json', 'w'))
PY
```

然后直接 eval：

```bash
CUDA_VISIBLE_DEVICES=0 python tools/train_net.py \
  --config-file lami_dino/configs/dino_convnext_large_4scale_12ep_lvis_zeroshot_d3_internal_verifier_w075.py \
  --num-gpus 1 \
  --eval-only \
  train.init_checkpoint=/root/autodl-tmp/model_final_ovd_lvis_kang.pth
```

这个内部版默认设置：

```text
first-stage classifier: d3_description_anchor_target_w075_bank_convnextl.npy
score ensemble: enabled, beta=0.1
verifier checkpoint: output/d3_crop_verifier_w075_no_score/verifier_best.pt
verifier text: d3_description_anchor_target_w100_bank_convnextl.npy
verifier fusion: logit_add, weight=0.25
verifier top-k: 20 flat query-class pairs per image
```

如果这个内部版接近离线 no-score rerank，就可以把它作为下一阶段主实现；如果低很多，说明内部 ROI feature 和外部 crop OpenCLIP feature 分布不一致，下一步就要训练一个真正基于 detector ROI feature 的 verifier。

## 10. 训练真正的 detector-ROI verifier

如果 crop-trained verifier 接进 `DINO.forward()` 后明显低于离线 rerank，说明训练特征和推理特征分布不一致。下一步改成用 detector 自己的 ROI feature 训练 verifier：

```text
detector predicted box -> extract_region_feature -> ROI feature
```

先用同一个 `w075` first-stage detector 保存每张图的 query box 和 `roi_features_ori`：

```bash
CUDA_VISIBLE_DEVICES=0 python tools/train_net.py \
  --config-file lami_dino/configs/dino_convnext_large_4scale_12ep_lvis_zeroshot_d3_desc_anchor_target_w075_save_roi_features.py \
  --num-gpus 1 \
  --eval-only \
  train.init_checkpoint=/root/autodl-tmp/model_final_ovd_lvis_kang.pth
```

输出目录：

```text
output/d3_roi_features_w075/pth
```

这个 config 只保存 verifier 需要的轻量内容，并把 ROI feature 以 fp16 落盘，避免全量 `pred_logits` 把磁盘打满。每个 `.pth` 包含：

```text
pred_boxes
roi_features_ori
```

如果之前保存中途因为磁盘满或 `torch.save` 写失败中断，先清掉旧的半截目录再重跑：

```bash
rm -rf output/d3_roi_features_w075/pth
```

然后把固定的 `verifier_pairs_w075` 匹配回这些 query，导出 trainer 可直接读取的 ROI-feature cache。建议先 smoke 2000/1000 样本，确认 bbox 到 query 的匹配 IoU 基本接近 1：

```bash
python tools/export_d3_roi_verifier_features.py \
  --train-jsonl dataset/d3/verifier_pairs_w075/train.jsonl \
  --val-jsonl dataset/d3/verifier_pairs_w075/val.jsonl \
  --saved-output-dir output/d3_roi_features_w075/pth \
  --text-embedding dataset/metadata/d3_description_anchor_target_w100_bank_convnextl.npy \
  --output-dir output/d3_roi_verifier_features_w075_smoke/cache \
  --same-phrase-neg-per-pos 1 \
  --wrong-phrase-neg-per-pos 2 \
  --max-train-samples 2000 \
  --max-val-samples 1000
```

完整导出：

```bash
python tools/export_d3_roi_verifier_features.py \
  --train-jsonl dataset/d3/verifier_pairs_w075/train.jsonl \
  --val-jsonl dataset/d3/verifier_pairs_w075/val.jsonl \
  --saved-output-dir output/d3_roi_features_w075/pth \
  --text-embedding dataset/metadata/d3_description_anchor_target_w100_bank_convnextl.npy \
  --output-dir output/d3_roi_verifier_features_w075/cache \
  --same-phrase-neg-per-pos 1 \
  --wrong-phrase-neg-per-pos 2
```

再复用同一个 MLP trainer，训练 no-score ROI verifier。这里的 `crop_feats` 实际上已经是 detector ROI feature：

```bash
python tools/train_d3_crop_verifier.py \
  --train-cache output/d3_roi_verifier_features_w075/cache/train_features.pt \
  --val-cache output/d3_roi_verifier_features_w075/cache/val_features.pt \
  --output-dir output/d3_roi_verifier_w075_no_score \
  --feature-mode no_detector_score \
  --epochs 5 \
  --batch-size 512 \
  --lr 1e-3
```

最后接回 `DINO.forward()` eval：

```bash
CUDA_VISIBLE_DEVICES=0 python tools/train_net.py \
  --config-file lami_dino/configs/dino_convnext_large_4scale_12ep_lvis_zeroshot_d3_internal_roi_verifier_w075.py \
  --num-gpus 1 \
  --eval-only \
  train.init_checkpoint=/root/autodl-tmp/model_final_ovd_lvis_kang.pth
```

这一步才是 “真正内部 verifier” 的关键检查：如果比 crop-trained internal verifier 高，说明之前主要问题是 feature distribution mismatch；如果还不高，再考虑把 verifier 从 MLP 升级成 query/text cross-attention 或直接训练进 detector。

## 11. D3 ROI verifier 实验记录

以下结果用于后续整理实验表。评测设置：

```text
benchmark: D3 intra full
detector: LaMI-DETR ConvNeXt-L
checkpoint: /root/autodl-tmp/model_final_ovd_lvis_kang.pth
first-stage classifier: d3_description_anchor_target_w075_bank_convnextl.npy
score ensemble: enabled, beta=0.1
verifier text: d3_description_anchor_target_w100_bank_convnextl.npy
verifier fusion: logit_add
verifier top-k: 20 flat query-class pairs per image
```

主结果：

| Method | AP | AP50 | AP75 | APs | APm | APl |
|:--|--:|--:|--:|--:|--:|--:|
| w075 target-framed text prototype | 9.1563 | 11.4391 | 9.3068 | 5.9630 | 10.8197 | 10.7859 |
| crop-trained internal verifier | 9.4168 | 11.6221 | 9.5625 | 6.2668 | 11.3023 | 11.3114 |
| ROI-trained internal verifier, weight=0.10 | 9.8187 | 12.2622 | 9.9849 | 6.5359 | 11.6386 | 11.5203 |
| ROI-trained internal verifier, weight=0.25 | **10.3111** | 12.8845 | **10.4725** | 7.0019 | 12.2422 | 12.0509 |
| ROI-trained internal verifier, weight=0.50 | 10.2652 | **12.8994** | 10.4105 | **7.0393** | **12.7739** | **12.2676** |
| offline verifier rerank, weight=0.25 | 10.50 | 13.10 | 10.70 | 6.60 | 12.30 | 12.00 |

Verifier validation：

| Feature setting | overall AP | overall AUC | same_phrase AP | same_phrase AUC | wrong_phrase AP | wrong_phrase AUC |
|:--|--:|--:|--:|--:|--:|--:|
| crop no-score verifier | 0.8720 | 0.9573 | 0.9344 | 0.9470 | 0.9218 | 0.9624 |
| ROI no-score verifier | 0.8718 | 0.9565 | 0.9279 | 0.9395 | 0.9272 | 0.9650 |
| ROI no-text verifier | 0.3188 | 0.6362 | 0.8825 | 0.9087 | 0.3333 | 0.5000 |

关键结论：

```text
Best internal result: ROI-trained verifier, fusion weight=0.25, AP 10.3111.
Gain over w075 text prototype baseline: +1.1548 AP.
Gain over crop-trained internal verifier: +0.8943 AP.
No-text verifier collapses on wrong_phrase_same_region to AP 0.3333 / AUC 0.5.
This supports that the gain comes from region-description matching, not detector score or box calibration.
```

### Frozen verifier follow-up runs

After the train-time BCE ablation, keep the ROI verifier frozen and evaluate it
as the main plug-in method. The runner below uses:

```text
config: dino_convnext_large_4scale_12ep_lvis_zeroshot_d3_internal_roi_verifier_w075.py
detector checkpoint: /root/autodl-tmp/model_final_ovd_lvis_kang.pth
verifier checkpoint: $TMP_OUT/d3_roi_verifier_w075_no_score/verifier_best.pt
train-time BCE: disabled
```

First run the main split fusion/top-k ablation:

```bash
export TMP_OUT=/root/autodl-tmp/LaMI-DETR-output
bash tools/run_d3_frozen_roi_verifier_ablation.sh ablation
```

This runs:

```text
d3_intra_full detector_only
d3_intra_full verifier weight=0.10 topk=20
d3_intra_full verifier weight=0.25 topk=20
d3_intra_full verifier weight=0.50 topk=20
d3_intra_full verifier weight=0.25 topk=10
d3_intra_full verifier weight=0.25 topk=50
```

Official D3 `pres/abs` JSONs contain only category subsets (316 present
categories and 106 absent categories). Since this detector predicts the full 422
D3 phrase bank, first create full-category compatible versions:

```bash
python tools/prepare_d3_subset_full_categories.py
```

This creates:

```text
dataset/d3/annotations/d3_pres_fullcats.json
dataset/d3/annotations/d3_abs_fullcats.json
```

Then run the available official D3 splits with detector-only and the current
best verifier setting:

```bash
D3_INTRA_PRES_JSON=d3/annotations/d3_pres_fullcats.json \
D3_INTRA_ABS_JSON=d3/annotations/d3_abs_fullcats.json \
bash tools/run_d3_frozen_roi_verifier_ablation.sh splits
```

To rerun the same splits with the best full-split top-k setting:

```bash
SPLIT_VERIFIER_TOPK=50 \
D3_INTRA_PRES_JSON=d3/annotations/d3_pres_fullcats.json \
D3_INTRA_ABS_JSON=d3/annotations/d3_abs_fullcats.json \
bash tools/run_d3_frozen_roi_verifier_ablation.sh splits
```

The split runner evaluates the non-redundant main bbox splits by default:

```text
d3_intra_full
d3_intra_pres, using d3_pres_fullcats.json
d3_intra_abs, using d3_abs_fullcats.json
```

`d3_inter_full` can still be run explicitly with `D3_SPLITS=d3_inter_full`,
but the released `d3_inter_full.json` has the same effective
`(image_id, category_id, bbox)` set as `d3_intra_full.json` for bbox AP in this
setup. Treat it as a duplicate sanity check, not an independent main split.

Outputs are written under:

```text
$TMP_OUT/d3_frozen_roi_verifier_ablation/{split}/{setting}
```

Summarize all finished logs:

```bash
python tools/summarize_d3_ablation_results.py \
  --root "$TMP_OUT/d3_frozen_roi_verifier_ablation" \
  --output "$TMP_OUT/d3_frozen_roi_verifier_ablation/summary.csv"
```

Frozen verifier summary:

| Split | Setting | AP | AP50 | AP75 | APs | APm | APl | Gain |
|:--|:--|--:|--:|--:|--:|--:|--:|--:|
| d3_intra_full | detector_only | 9.1563 | 11.4391 | 9.3068 | 5.9630 | 10.8197 | 10.7859 | - |
| d3_intra_full | verifier w=0.25 topk=20 | 10.3111 | 12.8845 | 10.4725 | 7.0019 | 12.2422 | 12.0509 | +1.1548 |
| d3_intra_full | verifier w=0.25 topk=50 | **10.4748** | **13.1072** | **10.6299** | **7.0552** | **12.9048** | 12.1969 | **+1.3185** |
| d3_intra_pres | detector_only | 9.7479 | 12.2098 | 9.8280 | 5.5991 | 11.5583 | 11.6346 | - |
| d3_intra_pres | verifier w=0.25 topk=20 | 11.0696 | 13.8643 | 11.1748 | 6.6718 | 13.0225 | 13.1646 | +1.3217 |
| d3_intra_pres | verifier w=0.25 topk=50 | **11.2590** | **14.1168** | **11.3535** | **6.6955** | **13.7282** | **13.3405** | **+1.5111** |
| d3_intra_abs | detector_only | 7.4037 | 9.1556 | 7.7627 | 6.9056 | 8.7863 | 8.2977 | - |
| d3_intra_abs | verifier w=0.25 topk=20 | 8.0640 | 9.9817 | 8.3918 | 7.8568 | 10.0944 | 8.7853 | +0.6603 |
| d3_intra_abs | verifier w=0.25 topk=50 | **8.1516** | **10.1163** | **8.4861** | **7.9867** | **10.6383** | **8.8441** | **+0.7479** |

Top-k conclusion: `topk=50` beats `topk=20` on all three non-redundant D3 bbox
splits, so the current main frozen-verifier setting is `w=0.25, topk=50`.

`d3_inter_full` loads a different JSON file from `d3_intra_full`, but direct
COCOeval checks against the same predictions give identical bbox AP because the
effective `(image_id, category_id, bbox)` annotation set is the same:
`intra only=0`, `inter only=0`, `shared=20278`, with all annotations
`iscrowd=0`. Do not count it as an independent main result.

The runner skips existing `coco_instances_results.json` by default. Set
`SKIP_EXISTING=0` to force reruns, or `DRY_RUN=1` to print commands without
running them. It also skips missing D3 split annotation files by default; set
`SKIP_MISSING_SPLITS=0` if missing split files should be treated as a hard error.

## 12. 训练时内部 ROI verifier loss

前面的 ROI verifier 是离线训练再接回推理。下一步把 verifier loss 放进 `DINO.forward()` 训练分支：

```text
matched query ROI feature + correct phrase -> label 1
matched query ROI feature + wrong phrase   -> label 0
bad query ROI feature + same phrase        -> label 0
loss_region_verifier = BCE(verifier(region_feature, phrase_feature), label)
```

对应 config：

```text
lami_dino/configs/dino_convnext_large_4scale_12ep_lvis_zeroshot_d3_train_roi_verifier_w075.py
```

默认设置：

```text
region_verifier_train_enabled=True
region_verifier_enabled=True
region_verifier_train_feature_mode=no_detector_score
same_phrase_neg_per_pos=1
wrong_phrase_neg_per_pos=2
max_pairs=256
loss_region_verifier weight=0.1
region features detached=True
fusion weight=0.25
```

先跑一个短 smoke，确认 `loss_region_verifier` 正常下降且不会 OOM：

```bash
TMP_OUT=/root/autodl-tmp/LaMI-DETR-output

CUDA_VISIBLE_DEVICES=0 python tools/train_net.py \
  --config-file lami_dino/configs/dino_convnext_large_4scale_12ep_lvis_zeroshot_d3_train_roi_verifier_w075.py \
  --num-gpus 1 \
  train.init_checkpoint=/root/autodl-tmp/model_final_ovd_lvis_kang.pth \
  train.max_iter=200 \
  train.eval_period=200 \
  train.checkpointer.period=200 \
  train.output_dir="$TMP_OUT/d3_train_roi_verifier_w075_smoke" \
  dataloader.evaluator.output_dir="$TMP_OUT/d3_train_roi_verifier_w075_smoke"
```

如果 smoke 正常，再跑稍长版本：

```bash
CUDA_VISIBLE_DEVICES=0 python tools/train_net.py \
  --config-file lami_dino/configs/dino_convnext_large_4scale_12ep_lvis_zeroshot_d3_train_roi_verifier_w075.py \
  --num-gpus 1 \
  train.init_checkpoint=/root/autodl-tmp/model_final_ovd_lvis_kang.pth \
  train.max_iter=2000 \
  train.eval_period=1000 \
  train.checkpointer.period=1000 \
  train.output_dir="$TMP_OUT/d3_train_roi_verifier_w075_2k" \
  dataloader.evaluator.output_dir="$TMP_OUT/d3_train_roi_verifier_w075_2k"
```

主要看：

```text
loss_region_verifier
region_verifier_num_pairs
region_verifier_pos_rate
eval bbox AP
```

已跑结果：

| Setting | Init | Train update | AP | AP50 | AP75 | APs | APm | APl | Note |
|:--|:--|:--|--:|--:|--:|--:|--:|--:|:--|
| train config, eval-only | ROI no-score verifier | none | 10.3111 | 12.8845 | 10.4725 | 7.0019 | 12.2422 | 12.0509 | reproduces best internal ROI verifier |
| train config, scratch smoke | random verifier | joint detector+verifier, 200 iters | 7.4178 | 8.8053 | 7.6624 | 3.0858 | 8.6136 | 8.9512 | verifies loss path, but random verifier hurts ranking |
| train config, warm smoke | ROI no-score verifier | joint detector+verifier, 200 iters | 6.3780 | 7.5990 | 6.6460 | 3.0830 | 7.7310 | 7.7920 | naive joint fine-tuning damages the detector/verifier ranking |
| verifier-only warm smoke | ROI no-score verifier | verifier only, 200 iters, lr=1e-4 | 9.9547 | 12.4480 | 10.1159 | 6.7904 | 11.7018 | 11.7026 | freezing detector avoids collapse, but still drifts |
| verifier-only warm smoke | ROI no-score verifier | verifier only, 200 iters, lr=1e-5 | 10.2456 | 12.8119 | 10.4107 | 6.9175 | 12.1713 | 11.9813 | lower LR mostly preserves calibration |
| verifier-only warm 1k | ROI no-score verifier | verifier only, 1000 iters, lr=1e-5 | 10.1137 | 12.6402 | 10.2737 | 6.8667 | 11.9464 | 11.8489 | longer BCE fine-tuning keeps drifting down |

Current conclusion: keep the train-time BCE branch as an ablation behind flags,
but stop using it as the main method. The main result is still the frozen
ROI-trained internal verifier at AP 10.3111. To disable train-time BCE while
keeping test-time verifier fusion:

```text
model.region_verifier_enabled=True
model.region_verifier_train_enabled=False
model.criterion.weight_dict.loss_region_verifier=0.0
```

To fully disable the verifier:

```text
model.region_verifier_enabled=False
model.region_verifier_train_enabled=False
model.criterion.weight_dict.loss_region_verifier=0.0
```

The 200-iter warm smoke shows that the current train-time loss is wired, but
naive joint fine-tuning is not the right training strategy. The next controlled
test is verifier-only warm fine-tuning, which keeps the detector fixed through
an optimizer LR mask:

```text
lami_dino/configs/dino_convnext_large_4scale_12ep_lvis_zeroshot_d3_train_roi_verifier_only_w075.py
```

Run:

```bash
CUDA_VISIBLE_DEVICES=0 python tools/train_net.py \
  --config-file lami_dino/configs/dino_convnext_large_4scale_12ep_lvis_zeroshot_d3_train_roi_verifier_only_w075.py \
  --num-gpus 1 \
  train.init_checkpoint=/root/autodl-tmp/model_final_ovd_lvis_kang.pth \
  model.region_verifier_checkpoint="$TMP_OUT/d3_roi_verifier_w075_no_score/verifier_best.pt" \
  train.max_iter=200 \
  train.eval_period=200 \
  train.checkpointer.period=200 \
  train.output_dir="$TMP_OUT/d3_train_roi_verifier_only_w075_warm_smoke" \
  dataloader.evaluator.output_dir="$TMP_OUT/d3_train_roi_verifier_only_w075_warm_smoke"
```

Verifier-only warm fine-tuning confirms that freezing the detector avoids the
large collapse, but the BCE objective still degrades the frozen ROI verifier
over time. Treat this as a negative finding for naive BCE and revisit training
with a pairwise/ranking objective only after the frozen verifier results are
fully evaluated.

## 13. Pairwise ranking loss for ROI verifier training

After frozen verifier evaluation, the training-side upgrade is to replace BCE
with pairwise ranking while keeping the detector frozen. The ranking objective
uses the same sampled positive/negative verifier pairs, but optimizes:

```text
softplus(margin - (positive_logit - negative_logit))
```

This directly matches the test-time use of the verifier as a ranking/fusion
score and avoids forcing verifier logits to be calibrated binary probabilities.

Config:

```text
lami_dino/configs/dino_convnext_large_4scale_12ep_lvis_zeroshot_d3_train_roi_verifier_ranking_only_w075.py
```

Smoke run:

```bash
export TMP_OUT=/root/autodl-tmp/LaMI-DETR-output

CUDA_VISIBLE_DEVICES=0 python tools/train_net.py \
  --config-file lami_dino/configs/dino_convnext_large_4scale_12ep_lvis_zeroshot_d3_train_roi_verifier_ranking_only_w075.py \
  --num-gpus 1 \
  train.init_checkpoint=/root/autodl-tmp/model_final_ovd_lvis_kang.pth \
  model.region_verifier_checkpoint="$TMP_OUT/d3_roi_verifier_w075_no_score/verifier_best.pt" \
  train.max_iter=200 \
  train.eval_period=200 \
  train.checkpointer.period=200 \
  train.output_dir="$TMP_OUT/d3_train_roi_verifier_ranking_only_w075_warm_200" \
  dataloader.evaluator.output_dir="$TMP_OUT/d3_train_roi_verifier_ranking_only_w075_warm_200"
```

Watch:

```text
loss_region_verifier
region_verifier_num_pairs
region_verifier_pos_rate
region_verifier_num_rank_pairs
eval bbox AP
```

这个版本仍然是保守训练：ROI feature 默认 detach，所以首先训练 verifier 本身，不把梯度推回 detector 主干。若 verifier-only 版本稳定且 AP 有收益，再把：

```text
model.region_verifier_train_detach_region_features=False
```

作为真正端到端版本的下一轮 ablation。
