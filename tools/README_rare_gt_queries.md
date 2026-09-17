# Single-image pre-top300 trace (8ep versus 12ep)

This follows the completed `trace_single_rare_gt.py` report, not another full
evaluation. The scope is **image 218917, scarecrow GT 137708**, with the no-radius
K5 checkpoints at iterations **56799 and 85199**. It does not train, change any
checkpoint, perform a swap, or compute a new AP estimate.

Run on the server from the repository root:

```bash
cd ~/LaMI-DETR
mkdir -p /root/autodl-tmp/no_radius_scarecrow_queries
set -o pipefail
CUDA_VISIBLE_DEVICES=0 /root/miniconda3/envs/lami/bin/python -u \
  tools/trace_rare_gt_queries.py \
  --trace-json /root/autodl-tmp/no_radius_scarecrow_trace/report.json \
  --old-checkpoint /root/autodl-tmp/instructdet_k5_no_radius_bs32_4ep_seed42/model_0056799.pth \
  --new-checkpoint /root/autodl-tmp/instructdet_k5_no_radius_bs32_4ep_seed42/model_final.pth \
  --cache-search-root /root/autodl-tmp \
  --output-dir /root/autodl-tmp/no_radius_scarecrow_queries \
  2>&1 | tee /root/autodl-tmp/no_radius_scarecrow_queries/run.log
```

Use the actual path of the completed single-GT trace if it differs. Checkpoint
paths are authenticated by SHA256, iteration, explicit no-radius buffers, and
shared TPA aliases; filenames alone are not evidence of identity.

## Bounded reuse and inference

- Search only `*/pairing_cache/manifest.json` and `*/cache/manifest.json` directly
  under the search root. Supply other known dense caches with `--reuse-cache
  /path/to/pairing_cache ...`. No recursive scan through image shards.
- Reuse only a manifest with matching checkpoint hash, source/config code,
  prompt/model assets, annotations, fusion settings, and this image ID. Either
  cached side may supply either endpoint, based on its hash, not its name.
- Sparse fusion-union/top300 dumps are **not** complete query banks and cannot
  establish missing-box coverage. This tool requires all 900 final queries.
- Missing sides get exactly **one image forward each**, at most two forwards.
  The validation annotation index may load in full; inference cannot expand to
  other images. `--cache-only` refuses any missing-cache forward.
- Both modes use the project environment for config/assets. CPU replay is
  independent of GPU availability; a new model forward needs working Detrex.
- A matched cache that fails validation or prediction reproduction causes an
  error, not an automatic rerun or looser comparison. Use a fresh output
  directory when changing inputs/config/code; existing locked caches are not
  overwritten with incompatible inputs.

## Validation and outputs

Each endpoint must reproduce **all 300 saved image predictions** embedded in
the hashed input trace: category, score and box, one-to-one including duplicates.
Tolerance is 0.05 pixels for xywh coordinates and 5e-5 for score. This is in
addition to the native dense replay check (1e-4 logit, 5e-6 score). The 2GB full
prediction files do not need rereading: their original identities and the
trace's identity remain in the output. This is numerical reproduction on this
image, not independent proof of the old run's complete training provenance.

`report.json` is complete only after both sides pass. It contains:

- At IoU 0.5/0.75/0.9: eligible raw query counts, retained correct/other-class
  queries, and the highest true-class fused-score eligible candidate (not just
  the best-IoU query).
- Best-IoU, best detector-score and best CLIP-score eligible candidates, with
  detector/CLIP within-query ranks and exact all-pair fused rank intervals.
- Every query's box, GT IoU, true-class components and top300 membership.
- Hashed `old_all_query_scores.pt` and `new_all_query_scores.pt`: all 900 x 1203
  detector logits, CLIP logits and fused scores, boxes, category IDs, and the
  top300 mask. These are native-checked float32 replays, not modified scoring.

Upload `report.json` first; the dense tensors usually are not needed to interpret
the result. `no_eligible_final_query_box` means none of the **final** queries
covers this GT at that IoU; it says nothing about earlier encoder proposals.
If an eligible query exists but is absent from top300, its actual scores/ranks
locate a selection failure, not the historical training loss that caused it.
Query IDs are local to each endpoint and must not be equated across models.
This one selected example cannot establish a global rare-performance mechanism.
