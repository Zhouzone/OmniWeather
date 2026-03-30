# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

from .interleave_datasets import UnifiedEditIterableDataset
from .radar_dataset import RadarIterableDataset
from .t2i_dataset import T2IIterableDataset
from .vlm_dataset import SftJSONLIterableDataset
from .sat2rad_dataset import BagelSat2RadDataset
from .sevir_all_dataset import SevirAlldataset
from .metaquery_dataset import MetaQueryIterableDataset
from .radar_forcastor_dataset import RadarForcastorDataset as RadarForcastorIterableDataset
from .radar_forcastor_single_dataset import RadarForcastorSingleFrameDataset as RadarForcastorSingleDataset
from .cot_cumulative_frame_dataset import CoTCumulativeFrameDataset
from .cot_single_frame_dataset import CoTSingleFrameDataset


DATASET_REGISTRY = {
    't2i_pretrain': T2IIterableDataset,
    'vlm_sft': SftJSONLIterableDataset,
    'unified_edit': UnifiedEditIterableDataset,
    'radar_gen': RadarIterableDataset,
    'sat2rad': BagelSat2RadDataset,
    'sevir_all': SevirAlldataset,
    'metaquery_gen': MetaQueryIterableDataset,
    'radar_forcastor_iterable': RadarForcastorIterableDataset,
    'radar_forcastor_iterable_model': RadarForcastorIterableDataset,
    'radar_forcastor_single': RadarForcastorSingleDataset,
    'cot_cumulative_frame': CoTCumulativeFrameDataset,
    'cot_single_frame': CoTSingleFrameDataset,
}


DATASET_INFO = {

    # -------------------------------------------------------------------------
    # Text-to-Image (general, for base model capability)
    # -------------------------------------------------------------------------
    't2i_pretrain': {
        't2i': {
            'data_dir': '/mnt/shared-storage-user/sciprismax/science_uni/RadarQA-related/bagel_data/bagel_example/t2i',
            'num_files': 10,
            'num_total_samples': 1000,
        },
    },

    # -------------------------------------------------------------------------
    # Image Editing (general, for base model capability)
    # -------------------------------------------------------------------------
    'unified_edit': {
        'seedxedit_multi': {
            'data_dir': '/mnt/shared-storage-user/sciprismax/science_uni/RadarQA-related/bagel_data/bagel_example/editing/seedxedit_multi',
            'num_files': 10,
            'num_total_samples': 1000,
            'parquet_info_path': '/mnt/shared-storage-user/sciprismax/science_uni/RadarQA-related/bagel_data/bagel_example/editing/parquet_info/seedxedit_multi.json',
        },
    },

    # -------------------------------------------------------------------------
    # Radar Nowcasting Generation (SEVIR VIL)
    # -------------------------------------------------------------------------
    'radar_gen': {
        'nowcast_sevir_png': {
            'data_dir': '/mnt/shared-storage-user/sciprismax/science_uni/RadarQA-related/dataset4Bagel/generation/nowcast_sevir/train/png',
            'num_files': 4,
            'num_total_samples': 40000,
        },
    },

    # -------------------------------------------------------------------------
    # Radar Forecasting (EarthFormer-based cascade)
    # -------------------------------------------------------------------------
    'radar_forcastor_single': {
        'radar_forcastor_single': {
            'data_dir': '/mnt/shared-storage-user/sciprismax/science_uni/nowcast_cascastbased_data/earthformer_newpower_data/train',
            'num_files': 5,
            'num_total_samples': 45000,
        },
    },

    # -------------------------------------------------------------------------
    # MetaQuery (general image-to-image generation)
    # -------------------------------------------------------------------------
    'metaquery_gen': {
        'metaquery_instruct': {
            'data_dir': '/mnt/shared-storage-user/sciprismax/science_uni/MetaQuery_Instruct_2.4M_512res',
            'num_files': 100,
            'num_total_samples': 10000,
        },
    },

    # -------------------------------------------------------------------------
    # Satellite-to-Radar (SEVIR: IR069 + IR107 -> VIL)
    # -------------------------------------------------------------------------
    'sat2rad': {
        'sat2rad_png': {
            'data_dir': '/mnt/shared-storage-user/sciprismax/science_uni/sevir_full',
            'catalog_path': '/mnt/shared-storage-user/sciprismax/science_uni/sevir_full/output.csv',
            'num_files': 1,
        },
    },

    # -------------------------------------------------------------------------
    # SEVIR Multi-Task (sat2rad, downscaling, interpolation, translation, prediction)
    # -------------------------------------------------------------------------
    'sevir_all': {
        'ir_to_vis': {
            'data_dir': '/mnt/shared-storage-user/sciprismax/science_uni/sevir_full',
            'catalog_path': '/mnt/shared-storage-user/sciprismax/science_uni/sevir_full/output.csv',
            'num_files': 1,
            'data_type': 'ir_to_vis',
        },
        'sat2rad': {
            'data_dir': '/mnt/shared-storage-user/sciprismax/science_uni/sevir_full',
            'catalog_path': '/mnt/shared-storage-user/sciprismax/science_uni/sevir_full/output.csv',
            'num_files': 1,
            'data_type': 'sat2rad',
        },
        'vil_down_scaling': {
            'data_dir': '/mnt/shared-storage-user/sciprismax/science_uni/sevir_full',
            'catalog_path': '/mnt/shared-storage-user/sciprismax/science_uni/sevir_full/output.csv',
            'num_files': 1,
            'data_type': 'vil_down_scaling',
        },
        'ir_down_scaling': {
            'data_dir': '/mnt/shared-storage-user/sciprismax/science_uni/sevir_full',
            'catalog_path': '/mnt/shared-storage-user/sciprismax/science_uni/sevir_full/output.csv',
            'num_files': 1,
            'data_type': 'ir_down_scaling',
        },
        'interpolation': {
            'data_dir': '/mnt/shared-storage-user/sciprismax/science_uni/sevir_full',
            'catalog_path': '/mnt/shared-storage-user/sciprismax/science_uni/sevir_full/output.csv',
            'num_files': 1,
            'data_type': 'interpolation',
        },
        'ir_translation': {
            'data_dir': '/mnt/shared-storage-user/sciprismax/science_uni/sevir_full',
            'catalog_path': '/mnt/shared-storage-user/sciprismax/science_uni/sevir_full/output.csv',
            'num_files': 1,
            'data_type': 'ir_translation',
        },
        'rad_prediction': {
            'data_dir': '/mnt/shared-storage-user/sciprismax/science_uni/sevir_full',
            'catalog_path': '/mnt/shared-storage-user/sciprismax/science_uni/sevir_full/output.csv',
            'num_files': 1,
            'data_type': 'rad_prediction',
        },
        'ir_prediction': {
            'data_dir': '/mnt/shared-storage-user/sciprismax/science_uni/sevir_full',
            'catalog_path': '/mnt/shared-storage-user/sciprismax/science_uni/sevir_full/output.csv',
            'num_files': 1,
            'data_type': 'ir_prediction',
        },
    },

    # -------------------------------------------------------------------------
    # Vision-Language Model (VLM) SFT datasets
    # -------------------------------------------------------------------------
    'vlm_sft': {
        'llava_ov': {
            'data_dir': '/mnt/shared-storage-user/sciprismax/science_uni/RadarQA-related/bagel_data/bagel_example/vlm/images',
            'jsonl_path': '/mnt/shared-storage-user/sciprismax/science_uni/RadarQA-related/bagel_data/bagel_example/vlm/llava_ov_si.jsonl',
            'num_total_samples': 1000,
        },
        'radar_img_brief': {
            'data_dir': '/mnt/shared-storage-user/sciprismax/science_uni/RadarQA-related/RawRQA-20K',
            'jsonl_path': '/mnt/shared-storage-user/sciprismax/science_uni/RadarQA-related/dataset4Bagel/understanding/img_brief_train.jsonl',
            'num_total_samples': 20000,
        },
        'radar_img_detail': {
            'data_dir': '/mnt/shared-storage-user/sciprismax/science_uni/RadarQA-related/RawRQA-20K',
            'jsonl_path': '/mnt/shared-storage-user/sciprismax/science_uni/RadarQA-related/dataset4Bagel/understanding/img_detail_train.jsonl',
            'num_total_samples': 14500,
        },
        'radar_seq_brief': {
            'data_dir': '/mnt/shared-storage-user/sciprismax/science_uni/RadarQA-related/RawRQA-20K',
            'jsonl_path': '/mnt/shared-storage-user/sciprismax/science_uni/RadarQA-related/dataset4Bagel/understanding/seq_brief_train.jsonl',
            'num_total_samples': 20000,
        },
        'radar_seq_detail': {
            'data_dir': '/mnt/shared-storage-user/sciprismax/science_uni/RadarQA-related/RawRQA-20K',
            'jsonl_path': '/mnt/shared-storage-user/sciprismax/science_uni/RadarQA-related/dataset4Bagel/understanding/seq_detail_train.jsonl',
            'num_total_samples': 14500,
        },
    },
}
