#!/bin/bash
# ============================================================================
# Omni-Weather: Evaluate Radar Nowcasting
# ============================================================================
# Usage: bash scripts/eval/eval_nowcast.sh
# ============================================================================

MODEL_PATH="/mnt/shared-storage-user/zhouzhiwang/bagel/models/BAGEL-7B-MoT"
CKPT_PATH="/mnt/shared-storage-user/zhouzhiwang/bagel/experiments/012_Bagel_radarqa_servir/checkpoints/0020000"
DATA_DIR="/mnt/shared-storage-user/sciprismax/science_uni/RadarQA-related/dataset4Bagel/generation/nowcast_sevir/test/png"
OUTPUT_DIR="/mnt/shared-storage-user/zhouzhiwang/omni-weather/results/nowcast_eval"

python inference/inference_radar.py \
  --model_path ${MODEL_PATH} \
  --ckpt_path ${CKPT_PATH} \
  --data_dir ${DATA_DIR} \
  --output_folder ${OUTPUT_DIR} \
  --max_samples 50
