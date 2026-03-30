#!/bin/bash
# ============================================================================
# Omni-Weather: Evaluate Sat2Rad generation
# ============================================================================
# Usage: bash scripts/eval/eval_sat2rad.sh
# ============================================================================

MODEL_PATH="/mnt/shared-storage-user/zhouzhiwang/bagel/models/BAGEL-7B-MoT"
CKPT_PATH="/mnt/shared-storage-user/zhouzhiwang/bagel/experiments/012_Bagel_radarqa_servir/checkpoints/0020000"
DATA_DIR="/mnt/shared-storage-user/sciprismax/science_uni/sevir_full"
CATALOG_PATH="/mnt/shared-storage-user/sciprismax/science_uni/sevir_full/output.csv"
OUTPUT_DIR="/mnt/shared-storage-user/zhouzhiwang/omni-weather/results/sat2rad_eval"

python eval/gen/sat2rad/inference_sat2rad.py \
  --model_path ${MODEL_PATH} \
  --ckpt_path ${CKPT_PATH} \
  --data_dir ${DATA_DIR} \
  --catalog_path ${CATALOG_PATH} \
  --output_folder ${OUTPUT_DIR} \
  --max_samples 100
