---
license: apache-2.0
tags:
  - weather
  - multimodal
  - radar
  - satellite
  - nowcasting
  - bagel
language:
  - en
base_model: ByteDance-Seed/BAGEL-7B-MoT
---

# Omni-Weather

Official model release for the paper [Omni-Weather: Unified Multimodal Foundation Model for Weather Generation and Understanding](https://arxiv.org/abs/2512.21643), built on [BAGEL-7B-MoT](https://huggingface.co/ByteDance-Seed/BAGEL-7B-MoT).

## Model Description

Omni-Weather fine-tunes the BAGEL 7B Mixture-of-Transformer-Experts model for weather-domain tasks:

- **Weather Understanding (VLM)** - Answering questions about radar and satellite imagery (RadarQA)
- **Weather Generation (Sat2Rad)** - Generating radar VIL images from satellite IR observations (IR069 + IR107 -> VIL)
- **Weather Nowcasting** - Predicting future radar frames from historical sequences
- **Multi-Task Weather Processing** - Downscaling, interpolation, cross-modal translation on SEVIR data

## Files

| File | Size | Description |
|------|------|-------------|
| `ema.safetensors` | ~55GB | Fine-tuned model weights (EMA) |
| `ae.safetensors` | ~320MB | VAE encoder/decoder |
| `config.json` | - | Model configuration |
| `llm_config.json` | - | LLM backbone config |
| `vit_config.json` | - | Vision encoder config |
| `tokenizer*` | - | Tokenizer files |

## Usage

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="Zhouzone/Omni-Weather",
    local_dir="models/Omni-Weather",
    local_dir_use_symlinks=False,
)
```

See the [GitHub repository](https://github.com/Zhouzone/OmniWeather) for training and inference instructions.

## Training Details

- **Base Model**: ByteDance-Seed/BAGEL-7B-MoT
- **Training Data**: SEVIR (sat2rad, multi-task) + RadarQA (understanding)
- **Training Steps**: 20,000
- **Hardware**: 8x GPUs with FSDP

## Citation

```bibtex
@article{zhou2025omni,
  title={Omni-Weather: Unified Multimodal Foundation Model for Weather Generation and Understanding},
  author={Zhou, Zhiwang and Pu, Yuandong and He, Xuming and Liu, Yidi and Chen, Yixin and Gong, Junchao and Zhuang, Xiang and Xu, Wanghan and Cao, Qinglong and Tang, Shixiang and others},
  journal={arXiv preprint arXiv:2512.21643},
  year={2025}
}
```

## Acknowledgements

Built upon [BAGEL](https://github.com/ByteDance-Seed/Bagel) by ByteDance-Seed.
