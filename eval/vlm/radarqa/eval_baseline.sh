torchrun --nproc_per_node=1 -m eval.vlm.eval.radarqa.eval \
    --test-file dataset/RadarQA-related/dataset4Bagel/understanding/img_brief_test.jsonl \
    --model-path pretrained/ByteDance-Seed/BAGEL-7B-MoT \
    --out-dir Baseline_Results

torchrun --nproc_per_node=1 -m --master_port=29501 eval.vlm.eval.radarqa.eval \
    --test-file dataset/RadarQA-related/dataset4Bagel/understanding/img_detail_test.jsonl \
    --model-path pretrained/ByteDance-Seed/BAGEL-7B-MoT \
    --out-dir Baseline_Results

torchrun --nproc_per_node=1 -m --master_port=29502 eval.vlm.eval.radarqa.eval \
    --test-file dataset/RadarQA-related/dataset4Bagel/understanding/seq_brief_test.jsonl \
    --model-path pretrained/ByteDance-Seed/BAGEL-7B-MoT \
    --out-dir Baseline_Results

torchrun --nproc_per_node=1 -m --master_port=29502 eval.vlm.eval.radarqa.eval \
    --test-file dataset/RadarQA-related/dataset4Bagel/understanding/seq_detail_test.jsonl \
    --model-path pretrained/ByteDance-Seed/BAGEL-7B-MoT \
    --out-dir Baseline_Results