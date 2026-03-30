#!/bin/bash

# 设置重试次数（可以根据需要调整）
max_retries=5
retry_count=0

# 循环运行 gen_gpt4open.py，直到成功或达到最大重试次数
while true; do
    echo "正在运行 gen_gpt4open.py (第 $((retry_count + 1)) 次尝试)"
    python gen_gpt_score4open.py --jsonl_path /root/bagel_baseline_img_detail_test_results.jsonl

    # 检查 gen_gpt4open.py 是否成功运行
    if [ $? -eq 0 ]; then
        echo "gen_gpt4open.py 成功运行，任务完成！"
        exit 0
    else
        retry_count=$((retry_count + 1))
        if [ $retry_count -ge $max_retries ]; then
            echo "达到最大重试次数，任务失败！"
            exit 1
        fi
        echo "gen_gpt4open.py 运行失败，将在 5 秒后重试..."
        sleep 5  # 等待 5 秒后再重试
    fi
done