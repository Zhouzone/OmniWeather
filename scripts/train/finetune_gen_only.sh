#!/bin/bash
# ============================================================================
# Omni-Weather: Generation-only fine-tuning (sat2rad / nowcasting)
# ============================================================================
# Usage: bash scripts/train/finetune_gen_only.sh
# ============================================================================

MODEL_PATH="/mnt/shared-storage-user/zhouzhiwang/bagel/models/BAGEL-7B-MoT"
CONFIG_FILE="./data/configs/radar_nowcast.yaml"  # or sat2rad_radarqa.yaml, sevir_multitask.yaml
OUTPUT_DIR="/mnt/shared-storage-user/zhouzhiwang/omni-weather/experiments/omni_weather_gen"
NUM_GPUS=8

torchrun \
  --nnodes=1 \
  --node_rank=0 \
  --nproc_per_node=${NUM_GPUS} \
  --master_addr=127.0.0.1 \
  --master_port=29500 \
  train/pretrain_unified_navit.py \
  --dataset_config_file ${CONFIG_FILE} \
  --model_path ${MODEL_PATH} \
  --layer_module Qwen2MoTDecoderLayer \
  --max_latent_size 64 \
  --resume_from ${MODEL_PATH} \
  --results_dir ${OUTPUT_DIR} \
  --checkpoint_dir ${OUTPUT_DIR}/checkpoints \
  --finetune_from_hf True \
  --auto_resume True \
  --resume-model-only True \
  --finetune-from-ema True \
  --log_every 1 \
  --visual_gen True \
  --visual_und False \
  --lr 2e-5 \
  --num_worker 1 \
  --num_shard ${NUM_GPUS} \
  --expected_num_tokens 10240 \
  --max_num_tokens 11520 \
  --max_num_tokens_per_sample 10240 \
  --wandb_offline True
