source /jobutils/scripts/worker_init.sh
export PATH="/home/zhouzhiwang/miniconda3/bin:$PATH"
export PATH="$PATH:/usr/local/nvidia/bin"
export LD_LIBRARY_PATH="/usr/local/nvidia/lib64:${LD_LIBRARY_PATH:-}"

# 关闭代理（防止下载/日志受代理干扰）
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
echo "[INFO] Proxy disabled."


# ===== 运行目录 & 环境 =====
source /home/zhouzhiwang/.bashrc

conda activate bagel
cd /mnt/shared-storage-user/zhouzhiwang/bagel

# ===== 训练配置 =====
export WANDB_API_KEY='e6a9e185930dd10939cf9190791308d87dba9a39'
EXP_NAME="012_Bagel_radarqa_servir"

LOG_DIR="/mnt/shared-storage-user/zhouzhiwang/bagel/eval_log"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/A001_train_${EXP_NAME}_$(date +%Y%m%d_%H%M%S).txt"



# 明确暴露 8 张卡
export CUDA_VISIBLE_DEVICES=0,1

torchrun --nproc_per_node=1 -m --master_port=29503 eval.vlm.eval.radarqa.eval \
    --test-file /mnt/shared-storage-user/zhouzhiwang/bagel/inference/inf_test1/inference_data.jsonl \
    --model-path /mnt/shared-storage-user/sciprismax/science_uni/Bagel_experiments/012_Bagel_radarqa_servir/checkpoints/0010000 \
    --out-dir /mnt/shared-storage-user/zhouzhiwang/bagel/inference/inf_results/cot.jsonl

torchrun --nproc_per_node=1 -m --master_port=29501 eval.vlm.eval.radarqa.eval \
    --test-file /mnt/shared-storage-user/zhouzhiwang/bagel/inference/inf_test1/inference_data_nocot.jsonl \
    --model-path /mnt/shared-storage-user/sciprismax/science_uni/Bagel_experiments/012_Bagel_radarqa_servir/checkpoints/0010000 \
    --out-dir /mnt/shared-storage-user/zhouzhiwang/bagel/inference/inf_results/nocot.jsonl

