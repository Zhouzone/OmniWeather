# Omni-Weather

Omni-Weather is the official code release for the paper [Omni-Weather: A Unified Multimodal Model for Weather Radar Understanding and Generation](https://arxiv.org/abs/2512.21643).

[Paper](https://arxiv.org/abs/2512.21643) | [Hugging Face Model](https://huggingface.co/akiwatanabe/Omni-Weather) | [Hugging Face Paper](https://huggingface.co/papers/2512.21643)

## Overview

Omni-Weather adapts the [BAGEL](https://github.com/ByteDance-Seed/Bagel) multimodal foundation model to weather radar understanding and generation. The repository includes training code, inference scripts, evaluation utilities, and dataset configuration files for multimodal weather tasks.

Current tasks covered by this repo include:

- Radar understanding with RadarQA-style visual question answering
- Satellite-to-radar generation on SEVIR-style data
- Radar nowcasting and forecasting variants
- Multi-task weather modeling with shared multimodal training

## Repository Layout

- `train/`: training entrypoints and FSDP utilities
- `inference/`: inference scripts for generation, nowcasting, and radar reasoning
- `eval/`: evaluation scripts for generation and VLM tasks
- `data/`: dataset loaders, transforms, and YAML configs
- `modeling/`: Omni-Weather / BAGEL-based model components
- `scripts/`: example shell scripts for training and evaluation
- `hf_upload/`: auxiliary files used for Hugging Face model packaging

## Environment Setup

```bash
conda create -n omni-weather python=3.10 -y
conda activate omni-weather

pip install -r requirements.txt
pip install flash_attn==2.5.8 --no-build-isolation
```

## Base Model and Released Weights

This project builds on the BAGEL base model and releases Omni-Weather checkpoints separately on Hugging Face.

- Base model: `ByteDance-Seed/BAGEL-7B-MoT`
- Omni-Weather model: `akiwatanabe/Omni-Weather`

To download the released Omni-Weather weights:

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="akiwatanabe/Omni-Weather",
    local_dir="models/Omni-Weather",
    local_dir_use_symlinks=False,
)
```

## Training

Example scripts:

```bash
# Joint understanding + generation fine-tuning
bash scripts/train/finetune.sh

# Generation-focused fine-tuning
bash scripts/train/finetune_gen_only.sh
```

The main training entrypoint is:

```bash
torchrun --nproc_per_node=8 train/pretrain_unified_navit.py \
  --dataset_config_file data/configs/sat2rad_radarqa.yaml \
  --model_path models/BAGEL-7B-MoT \
  --resume_from models/BAGEL-7B-MoT \
  --finetune_from_hf True \
  --visual_gen True \
  --visual_und True
```

Available dataset configs in `data/configs/`:

- `cot_nowcast.yaml`
- `radar_forcastor.yaml`
- `radar_nowcast.yaml`
- `radarqa.yaml`
- `sat2rad_radarqa.yaml`
- `sevir_multitask.yaml`

## Evaluation

```bash
# Sat2Rad generation evaluation
bash scripts/eval/eval_sat2rad.sh

# Radar nowcasting evaluation
bash scripts/eval/eval_nowcast.sh
```

Some evaluation helpers under `eval/vlm/radarqa/detail/` use OpenAI-compatible APIs. They now read credentials from environment variables:

```bash
export OPENAI_API_KEY=...
export OPENAI_BASE_URL=http://your-endpoint/v1
```

## Data Preparation

This repository expects users to prepare datasets locally. Large datasets and checkpoints are intentionally not stored in Git.

Typical formats used here:

- SEVIR-style `.npy` arrays for radar / satellite tasks
- Parquet files for sequence prediction tasks
- JSONL conversation data for RadarQA-style VLM training

After downloading and organizing your data, update dataset paths in the relevant dataset config or loader code under `data/`.

## Open-Source Release Notes

- Source code in this repository is released under Apache-2.0
- Model weights are released separately on Hugging Face
- Data is not redistributed here; please follow each dataset's original license and access policy
- Secrets and local artifacts should not be committed; see `.gitignore` for excluded files

## Citation

```bibtex
@article{omniweather2025,
  title   = {Omni-Weather: A Unified Multimodal Model for Weather Radar Understanding and Generation},
  author  = {Watanabe, Aki and collaborators},
  journal = {arXiv preprint arXiv:2512.21643},
  year    = {2025}
}
```

## Acknowledgements

Omni-Weather builds on the BAGEL codebase and pretrained components from ByteDance Seed and the open-source multimodal ecosystem.
