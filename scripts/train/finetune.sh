#!/bin/bash
# ============================================================================
# Omni-Weather: Fine-tune BAGEL for weather tasks
# ============================================================================
# Usage: bash scripts/train/finetune.sh
#
# Before running, set the following variables to match your environment:
#   - MODEL_PATH: Path to pretrained BAGEL-7B-MoT checkpoint
#   - CONFIG_FILE: Dataset config YAML (see data/configs/)
#   - OUTPUT_DIR: Where to save logs and checkpoints
#   - NUM_GPUS: Number of GPUs per node
# ============================================================================

# --- Configuration ---
MODEL_PATH="/mnt/shared-storage-user/zhouzhiwang/bagel/models/BAGEL-7B-MoT"
CONFIG_FILE="./data/configs/sat2rad_radarqa.yaml"
OUTPUT_DIR="/mnt/shared-storage-user/zhouzhiwang/omni-weather/experiments/omni_weather"
NUM_GPUS=8

# --- Training ---
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
  --visual_und True \
  --lr 2e-5 \
  --num_worker 1 \
  --num_shard ${NUM_GPUS} \
  --expected_num_tokens 10240 \
  --max_num_tokens 11520 \
  --max_num_tokens_per_sample 10240 \
  --wandb_offline True
