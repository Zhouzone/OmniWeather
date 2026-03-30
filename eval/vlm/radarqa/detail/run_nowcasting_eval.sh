#!/bin/bash

# 设置路径
JSONL_PATH="/mnt/shared-storage-user/zhouzhiwang/bagel/results/A008_Bagel_radarqa_servir_12k_2/nowcast_init_5_results.jsonl"
SCRIPT_PATH="/mnt/shared-storage-user/zhouzhiwang/bagel/eval/vlm/eval/radarqa/detail/gen_gpt_score_nowcasting.py"

# 检查文件是否存在
if [ ! -f "$JSONL_PATH" ]; then
    echo "Error: JSONL file not found at $JSONL_PATH"
    exit 1
fi

if [ ! -f "$SCRIPT_PATH" ]; then
    echo "Error: Python script not found at $SCRIPT_PATH"
    exit 1
fi

# 运行评分脚本
echo "Starting nowcasting evaluation..."
echo "Input file: $JSONL_PATH"
echo "Script: $SCRIPT_PATH"

python $SCRIPT_PATH --jsonl_path "$JSONL_PATH" --generate_summary

echo "Evaluation completed!"
