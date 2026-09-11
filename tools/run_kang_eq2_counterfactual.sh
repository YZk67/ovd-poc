#!/usr/bin/env bash
set -euo pipefail

checkpoint="${1:-/root/autodl-tmp/model_final_ovd_lvis_kang.pth}"
output_root="${2:-/root/autodl-tmp/kang_eq2_counterfactual}"
gpu_ids="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
python_bin="${PYTHON_BIN:-python}"
config="lami_dino/configs/dino_convnext_large_4scale_12ep_lvis_kang_eq2_eval.py"

IFS=',' read -r -a gpu_array <<< "${gpu_ids}"
num_gpus="${#gpu_array[@]}"

if [[ ! -f "${checkpoint}" ]]; then
  echo "Checkpoint not found: ${checkpoint}" >&2
  exit 2
fi

run_eval() {
  local name="$1"
  local legacy="$2"
  local output_dir="${output_root}/${name}"

  echo "[run] ${name}: legacy_logsumexp=${legacy}"
  CUDA_VISIBLE_DEVICES="${gpu_ids}" "${python_bin}" tools/train_net.py \
    --config-file "${config}" \
    --num-gpus "${num_gpus}" \
    --eval-only \
    train.init_checkpoint="${checkpoint}" \
    train.output_dir="${output_dir}" \
    dataloader.evaluator.output_dir="${output_dir}" \
    model.classifier.tpa_eval_legacy_logsumexp="${legacy}"
}

run_eval calibrated False
run_eval legacy True

"${python_bin}" tools/summarize_eq2_counterfactual.py \
  --calibrated-log "${output_root}/calibrated/log.txt" \
  --legacy-log "${output_root}/legacy/log.txt"
