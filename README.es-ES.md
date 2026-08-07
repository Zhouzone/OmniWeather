

# Omni-Weather

Omni-Weather es la versión oficial del código correspondiente al artículo [Omni-Weather: A Unified Multimodal Model for Weather Radar Understanding and Generation](https://arxiv.org/abs/2512.21643).

[![Paper](https://img.shields.io/badge/Paper-arXiv%3A2512.21643-b31b1b?logo=arxiv&logoColor=white)](https://arxiv.org/abs/2512.21643)
[![Model](https://img.shields.io/badge/Hugging%20Face-Omni--Weather-fcd022?logo=huggingface&logoColor=black)](https://huggingface.co/akiwatanabe/Omni-Weather)
[![GitHub](https://img.shields.io/badge/GitHub-OmniWeather-000000?logo=github&logoColor=white)](https://github.com/Zhouzone/OmniWeather)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![GitHub](https://img.shields.io/github/stars/Zhouzone/OmniWeather?style=social)](https://github.com/Zhouzone/OmniWeather)

[Paper](https://arxiv.org/abs/2512.21643) | [Hugging Face Model](https://huggingface.co/akiwatanabe/Omni-Weather) | [Hugging Face Paper](https://huggingface.co/papers/2512.21643)

## Descripción

Omni-Weather adapta el modelo base multimodal [BAGEL](https://github.com/ByteDance-Seed/Bagel) a la comprensión y generación de radares meteorológicos. El repositorio incluye código de entrenamiento, scripts de inferencia, utilidades de evaluación y archivos de configuración de conjuntos de datos para tareas meteorológicas multimodales.

Las tareas actuales cubiertas por este repositorio incluyen:

- Comprensión de radares mediante preguntas y respuestas visuales al estilo RadarQA
- Generación de satélite a radar en datos al estilo SEVIR
- Variantes de nowcasting (pronóstico a muy corto plazo) y predicción de radares
- Modelado meteorológico multitarea con entrenamiento multimodal compartido

## Estructura del Repositorio

- `train/`: puntos de entrada para entrenamiento y utilidades FSDP
- `inference/`: scripts de inferencia para generación, nowcasting y razonamiento radar
- `eval/`: scripts de evaluación para generación y tareas VLM
- `data/`: cargadores de conjuntos de datos, transformaciones y configuraciones YAML
- `modeling/`: componentes del modelo Omni-Weather / basados en BAGEL
- `scripts/`: scripts de shell de ejemplo para entrenamiento y evaluación
- `hf_upload/`: archivos auxiliares utilizados para el empaquetado del modelo en Hugging Face

## Configuración del Entorno

```bash
conda create -n omni-weather python=3.10 -y
conda activate omni-weather

pip install -r requirements.txt
pip install flash_attn==2.5.8 --no-build-isolation
```

## Modelo Base y Pesos Publicados

Este proyecto se basa en el modelo base BAGEL y publica los checkpoints de Omni-Weather por separado en Hugging Face.

- Modelo base: `ByteDance-Seed/BAGEL-7B-MoT`
- Modelo Omni-Weather: `akiwatanabe/Omni-Weather`

Para descargar los pesos de Omni-Weather publicados:

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="akiwatanabe/Omni-Weather",
    local_dir="models/Omni-Weather",
    local_dir_use_symlinks=False,
)
```

## Entrenamiento

Scripts de ejemplo:

```bash
# Joint understanding + generation fine-tuning
bash scripts/train/finetune.sh

# Generation-focused fine-tuning
bash scripts/train/finetune_gen_only.sh
```

El punto de entrada principal para el entrenamiento es:

```bash
torchrun --nproc_per_node=8 train/pretrain_unified_navit.py \
  --dataset_config_file data/configs/sat2rad_radarqa.yaml \
  --model_path models/BAGEL-7B-MoT \
  --resume_from models/BAGEL-7B-MoT \
  --finetune_from_hf True \
  --visual_gen True \
  --visual_und True
```

Configuraciones de conjuntos de datos disponibles en `data/configs/`:

- `cot_nowcast.yaml`
- `radar_forcastor.yaml`
- `radar_nowcast.yaml`
- `radarqa.yaml`
- `sat2rad_radarqa.yaml`
- `sevir_multitask.yaml`

## Evaluación

```bash
# Sat2Rad generation evaluation
bash scripts/eval/eval_sat2rad.sh

# Radar nowcasting evaluation
bash scripts/eval/eval_nowcast.sh
```

Algunas herramientas de evaluación bajo `eval/vlm/radarqa/detail/` utilizan API compatibles con OpenAI. Ahora leen las credenciales desde variables de entorno:

```bash
export OPENAI_API_KEY=...
export OPENAI_BASE_URL=http://your-endpoint/v1
```

## Preparación de Datos

Este repositorio espera que los usuarios preparen los conjuntos de datos localmente. Los conjuntos de datos grandes y los checkpoints no se almacenan intencionalmente en Git.

Formatos típicos utilizados aquí:

- Matrices `.npy` al estilo SEVIR para tareas de radar / satélite
- Archivos Parquet para tareas de predicción de secuencias
- Datos de conversación JSONL para entrenamiento de VLM al estilo RadarQA

Después de descargar y organizar sus datos, actualice las rutas de los conjuntos de datos en la configuración del conjunto de datos correspondiente o el código del cargador bajo `data/`.

## Notas de la Versión de Código Abierto

- El código fuente en este repositorio se publica bajo la licencia Apache-2.0
- Los pesos del modelo se publican por separado en Hugging Face
- Los datos no se redistribuyen aquí; siga la licencia y política de acceso original de cada conjunto de datos
- Los secretos y los artefactos locales no deben subirse; consulte `.gitignore` para los archivos excluidos

## Cita

```bibtex
@article{zhou2025omni,
  title={Omni-Weather: Unified Multimodal Foundation Model for Weather Generation and Understanding},
  author={Zhou, Zhiwang and Pu, Yuandong and He, Xuming and Liu, Yidi and Chen, Yixin and Gong, Junchao and Zhuang, Xiang and Xu, Wanghan and Cao, Qinglong and Tang, Shixiang and others},
  journal={arXiv preprint arXiv:2512.21643},
  year={2025}
}
```

## Agradecimientos

Omni-Weather se construye sobre la base de código de BAGEL y los componentes preentrenados de ByteDance Seed y el ecosistema multimodal de código abierto.
