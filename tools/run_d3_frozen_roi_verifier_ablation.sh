#!/usr/bin/env bash
set -euo pipefail

# Run frozen ROI-verifier evaluations for D3.
#
# Usage:
#   TMP_OUT=/root/autodl-tmp/LaMI-DETR-output \
#   bash tools/run_d3_frozen_roi_verifier_ablation.sh ablation
#
# Modes:
#   quick     d3_intra_full detector-only + best verifier
#   ablation  d3_intra_full detector-only + fusion/top-k sweep
#   splits    all D3 splits detector-only + best verifier
#   all       ablation + splits

MODE="${1:-quick}"
TMP_OUT="${TMP_OUT:-/root/autodl-tmp/LaMI-DETR-output}"
PYTHON="${PYTHON:-python}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
SKIP_MISSING_SPLITS="${SKIP_MISSING_SPLITS:-1}"
DRY_RUN="${DRY_RUN:-0}"

CONFIG="${CONFIG:-lami_dino/configs/dino_convnext_large_4scale_12ep_lvis_zeroshot_d3_internal_roi_verifier_w075.py}"
DETECTOR_CKPT="${DETECTOR_CKPT:-/root/autodl-tmp/model_final_ovd_lvis_kang.pth}"
VERIFIER_CKPT="${VERIFIER_CKPT:-$TMP_OUT/d3_roi_verifier_w075_no_score/verifier_best.pt}"
OUT_ROOT="${OUT_ROOT:-$TMP_OUT/d3_frozen_roi_verifier_ablation}"
SPLIT_VERIFIER_WEIGHT="${SPLIT_VERIFIER_WEIGHT:-0.25}"
SPLIT_VERIFIER_TOPK="${SPLIT_VERIFIER_TOPK:-20}"

SPLITS=(
  d3_intra_full
  d3_intra_pres
  d3_intra_abs
  d3_inter_full
)

d3_annotation_path() {
  local split="$1"
  local data_root="${DETECTRON2_DATASETS:-dataset}"
  local rel_path

  case "$split" in
    d3_inter_full) rel_path="${D3_INTER_FULL_JSON:-d3/annotations/d3_inter_full.json}" ;;
    d3_inter_pres) rel_path="${D3_INTER_PRES_JSON:-d3/annotations/d3_inter_pres.json}" ;;
    d3_inter_abs) rel_path="${D3_INTER_ABS_JSON:-d3/annotations/d3_inter_abs.json}" ;;
    d3_intra_full) rel_path="${D3_INTRA_FULL_JSON:-d3/annotations/d3_intra_full.json}" ;;
    d3_intra_pres) rel_path="${D3_INTRA_PRES_JSON:-d3/annotations/d3_intra_pres.json}" ;;
    d3_intra_abs) rel_path="${D3_INTRA_ABS_JSON:-d3/annotations/d3_intra_abs.json}" ;;
    *) return 1 ;;
  esac

  if [[ "$rel_path" == *"://"* || "$rel_path" = /* ]]; then
    printf '%s\n' "$rel_path"
  else
    printf '%s\n' "$data_root/$rel_path"
  fi
}

split_available() {
  local split="$1"
  local ann_path

  ann_path="$(d3_annotation_path "$split")" || return 0
  if [[ "$ann_path" == *"://"* || -f "$ann_path" ]]; then
    return 0
  fi

  if [[ "$SKIP_MISSING_SPLITS" == "1" ]]; then
    echo "[skip] $split annotation not found: $ann_path" >&2
    return 1
  fi

  echo "[error] $split annotation not found: $ann_path" >&2
  exit 1
}

run_eval() {
  local split="$1"
  local tag="$2"
  local verifier_enabled="$3"
  local fusion_weight="${4:-0.25}"
  local topk="${5:-20}"
  local output_dir="$OUT_ROOT/$split/$tag"

  if ! split_available "$split"; then
    return
  fi

  if [[ "$SKIP_EXISTING" == "1" && -f "$output_dir/coco_instances_results.json" ]]; then
    echo "[skip] $split $tag already has coco_instances_results.json"
    return
  fi

  if [[ "$DRY_RUN" != "1" ]]; then
    mkdir -p "$output_dir"
  fi

  local -a cmd=(
    "$PYTHON" tools/train_net.py
    --config-file "$CONFIG"
    --num-gpus 1
    --eval-only
    "train.init_checkpoint=$DETECTOR_CKPT"
    "model.region_verifier_checkpoint=$VERIFIER_CKPT"
    "model.region_verifier_enabled=$verifier_enabled"
    "model.region_verifier_train_enabled=False"
    "dataloader.test.dataset.names=$split"
    "train.output_dir=$output_dir"
    "dataloader.evaluator.output_dir=$output_dir"
  )

  if [[ "$verifier_enabled" == "True" ]]; then
    cmd+=(
      "model.region_verifier_fusion_weight=$fusion_weight"
      "model.region_verifier_topk_per_image=$topk"
    )
  fi

  echo
  echo "[run] split=$split tag=$tag verifier=$verifier_enabled weight=$fusion_weight topk=$topk"
  printf 'CUDA_VISIBLE_DEVICES=%q ' "$CUDA_VISIBLE_DEVICES"
  printf '%q ' "${cmd[@]}"
  echo

  if [[ "$DRY_RUN" != "1" ]]; then
    CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" "${cmd[@]}"
  fi
}

run_quick() {
  run_eval d3_intra_full detector_only False
  run_eval d3_intra_full verifier_w025_top20 True 0.25 20
}

run_ablation() {
  run_eval d3_intra_full detector_only False

  for weight in 0.1 0.25 0.5; do
    local wtag="${weight/./}"
    run_eval d3_intra_full "verifier_w${wtag}_top20" True "$weight" 20
  done

  for topk in 10 50; do
    run_eval d3_intra_full "verifier_w025_top${topk}" True 0.25 "$topk"
  done
}

run_splits() {
  local wtag="${SPLIT_VERIFIER_WEIGHT/./}"
  local tag="verifier_w${wtag}_top${SPLIT_VERIFIER_TOPK}"

  for split in "${SPLITS[@]}"; do
    run_eval "$split" detector_only False
    run_eval "$split" "$tag" True "$SPLIT_VERIFIER_WEIGHT" "$SPLIT_VERIFIER_TOPK"
  done
}

case "$MODE" in
  quick)
    run_quick
    ;;
  ablation)
    run_ablation
    ;;
  splits)
    run_splits
    ;;
  all)
    run_ablation
    run_splits
    ;;
  *)
    echo "Unknown mode: $MODE" >&2
    echo "Expected one of: quick, ablation, splits, all" >&2
    exit 2
    ;;
esac
