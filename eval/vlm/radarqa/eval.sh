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
# export WANDB_API_KEY='e6a9e185930dd10939cf9190791308d87dba9a39'
# EXP_NAME="012_Bagel_radarqa_servir"

# LOG_DIR="/mnt/shared-storage-user/zhouzhiwang/bagel/eval_log"
# mkdir -p "$LOG_DIR"
# LOG_FILE="${LOG_DIR}/A001_train_${EXP_NAME}_$(date +%Y%m%d_%H%M%S).txt"

export CUDA_VISIBLE_DEVICES=0,1,2,3

torchrun --nproc_per_node=1 -m --master_port=29504 eval.vlm.eval.radarqa.eval \
    --test-file /mnt/shared-storage-user/sciprismax/science_uni/RadarQA-related/dataset4Bagel/understanding/img_brief_test.jsonl \
    --model-path /mnt/shared-storage-user/sciprismax/science_uni/Bagel_experiments/012_Bagel_radarqa_servir/checkpoints/0008000 \
    --out-dir results/012_Bagel_radarqa_servir_8k/

torchrun --nproc_per_node=1 -m --master_port=29505 eval.vlm.eval.radarqa.eval \
    --test-file /mnt/shared-storage-user/sciprismax/science_uni/RadarQA-related/dataset4Bagel/understanding/img_detail_test.jsonl \
    --model-path /mnt/shared-storage-user/sciprismax/science_uni/Bagel_experiments/012_Bagel_radarqa_servir/checkpoints/0008000 \
    --out-dir results/012_Bagel_radarqa_servir_8k/

torchrun --nproc_per_node=1 -m --master_port=29506 eval.vlm.eval.radarqa.eval \
    --test-file /mnt/shared-storage-user/sciprismax/science_uni/RadarQA-related/dataset4Bagel/understanding/seq_brief_test.jsonl \
    --model-path /mnt/shared-storage-user/sciprismax/science_uni/Bagel_experiments/012_Bagel_radarqa_servir/checkpoints/0008000 \
    --out-dir results/012_Bagel_radarqa_servir_8k/

torchrun --nproc_per_node=1 -m --master_port=29507 eval.vlm.eval.radarqa.eval \
    --test-file /mnt/shared-storage-user/sciprismax/science_uni/RadarQA-related/dataset4Bagel/understanding/seq_detail_test.jsonl \
    --model-path /mnt/shared-storage-user/sciprismax/science_uni/Bagel_experiments/012_Bagel_radarqa_servir/checkpoints/0008000 \
    --out-dir results/012_Bagel_radarqa_servir_8k/