# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

import io
import json
import pandas as pd
import random
import base64
from PIL import Image
from .data_utils import pil_img2rgb
from .distributed_iterable_dataset import DistributedIterableDataset
from .parquet_utils import get_parquet_data_paths

Image.MAX_IMAGE_PIXELS = 20_000_000


class RadarIterableDataset(DistributedIterableDataset):
    def __init__(
        self, dataset_name, transform, tokenizer, data_dir_list, num_used_data, 
        local_rank=0, world_size=1, num_workers=8, data_status=None,
    ):
        """
        data_dir_list: list of data directories contains parquet files
        num_used_data: list of number of sampled data paths for each data directory
        """
        super().__init__(dataset_name, local_rank, world_size, num_workers)
        self.transform = transform
        self.tokenizer = tokenizer
        self.data_status = data_status
        self.data_paths = self.get_data_paths(data_dir_list, num_used_data)
        self.set_epoch()

    def get_data_paths(self, data_dir_list, num_used_data):
        return get_parquet_data_paths(data_dir_list, num_used_data)

    def __iter__(self):
        data_paths_per_worker, worker_id = self.get_data_paths_per_worker()
        if self.data_status is not None:
            parquet_start_id = self.data_status[worker_id][0]
            row_start_id = self.data_status[worker_id][2] + 1
        else:
            parquet_start_id = 0
            row_start_id = 0
        transform_stride = self.transform.stride

        print(
            f"rank-{self.local_rank} worker-{worker_id} dataset-{self.dataset_name}: "
            f"resuming data at parquet#{parquet_start_id}, row#{row_start_id}"
        )

        while True:
            data_paths_per_worker_ = data_paths_per_worker[parquet_start_id:]
            for parquet_idx, parquet_file_path in enumerate(data_paths_per_worker_, start=parquet_start_id):
                try:
                    # 使用pandas的fastparquet引擎读取parquet文件
                    df = pd.read_parquet(parquet_file_path, engine="fastparquet")
                    df = df.iloc[row_start_id:]

                    for row_idx, row in df.iterrows():
                        num_tokens = 0
                        image_tensor_list = []
                        target_tensor_list = []
                        text_ids_list = []
                        sequence_plan = []
                        
                        # 处理输入图像 (image字段: base64字符串列表)
                        try:
                            image_list = row['image']
                            for img_str in image_list:
                                img_bytes = base64.b64decode(img_str)
                                image = pil_img2rgb(Image.open(io.BytesIO(img_bytes)))
                                image_tensor = self.transform(image)
                                height, width = image_tensor.shape[1:]
                                num_tokens += width * height // transform_stride ** 2
                                image_tensor_list.append(image_tensor)
                            # 输入图像的序列规划
                            for _ in image_tensor_list:
                                sequence_plan.append({
                                    'type': 'vae_image',
                                    'enable_cfg': 1,  # 启用CFG作为条件
                                    'loss': 0,        # 不计算损失
                                    'special_token_loss': 0,
                                    'special_token_label': None,
                                })
                        except Exception as e:
                            print(f'Error processing input images: {e} in row#{row_idx}, {parquet_file_path}')
                            continue

                        # 处理目标图像 (caption字段: base64字符串列表)
                        try:
                            caption_list = row['caption']
                            for cap_str in caption_list:
                                cap_bytes = base64.b64decode(cap_str)
                                cap_image = pil_img2rgb(Image.open(io.BytesIO(cap_bytes)))
                                cap_tensor = self.transform(cap_image)
                                height, width = cap_tensor.shape[1:]
                                num_tokens += width * height // transform_stride ** 2
                                target_tensor_list.append(cap_tensor)
                            # 目标图像的序列规划
                            for _ in target_tensor_list:
                                sequence_plan.append({
                                    'type': 'vae_image',
                                    'enable_cfg': 0,  # 不启用CFG
                                    'loss': 1,        # 计算损失
                                    'special_token_loss': 0,
                                    'special_token_label': None,
                                })
                        except Exception as e:
                            print(f'Error processing target images: {e} in row#{row_idx}, {parquet_file_path}')
                            continue

                        # 如果tokenizer存在，添加文本token（可选/占位符）
                        if self.tokenizer is not None:
                            placeholder_text = "image to image generation"
                            text_ids = self.tokenizer.encode(placeholder_text)
                            num_tokens += len(text_ids)
                            text_ids_list.append(text_ids)
                            sequence_plan.append({
                                'type': 'text',
                                'enable_cfg': 1,
                                'loss': 0,
                                'special_token_loss': 0,
                                'special_token_label': None,
                            })

                        # 合并输入和目标图像
                        all_tensor_list = image_tensor_list + target_tensor_list

                        sample = dict(
                            image_tensor_list=all_tensor_list,  # 先输入后目标
                            text_ids_list=text_ids_list,
                            num_tokens=num_tokens,
                            sequence_plan=sequence_plan,
                            data_indexes={
                                "data_indexes": [parquet_idx, 0, row_idx],
                                "worker_id": worker_id,
                                "dataset_name": self.dataset_name,
                            }
                        )
                        yield sample

                    row_start_id = 0
                except Exception as e:
                    print(f'Error reading parquet file {parquet_file_path}: {e}')
                    continue
                    
            parquet_start_id = 0
            print(f"{self.dataset_name} repeat in rank-{self.local_rank} worker-{worker_id}")

if __name__ == "__main__":
    # 1. Import necessary classes for debugging
    import torch
    from torch.utils.data import DataLoader
    import os
    # Assuming the script is run from the project root directory
    from modeling.qwen2 import Qwen2Tokenizer
    from data.transforms import ImageTransform

    print("--- Starting RadarIterableDataset Debug ---")

    # 2. Set up paths for the model and data
    #    Please ensure these paths are correct for your environment.
    model_path = "models/BAGEL-7B-MoT"
    # This should be a directory containing your .parquet files
    data_dir = "your_data_path"

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model directory not found at '{model_path}'. "
            "Please download 'ByteDance-Seed/BAGEL-7B-MoT' first."
        )
    if not os.path.exists(data_dir):
        raise FileNotFoundError(
            f"Data directory not found at '{data_dir}'. "
            "Please ensure your Parquet files are in the correct location."
        )

    # 3. Instantiate the Tokenizer and Transform
    #    This mirrors the setup in the main training script.
    print(f"Loading tokenizer from: {model_path}")
    tokenizer = Qwen2Tokenizer.from_pretrained(model_path)
    
    transform = ImageTransform(
        max_image_size=512,  # Example size, adjust as needed
        min_image_size=256,
        image_stride=16      # This value is important for token calculation
    )
    print("Tokenizer and Transform loaded successfully.")

    # 4. Instantiate the RadarIterableDataset
    print("Instantiating RadarIterableDataset...")
    dataset = RadarIterableDataset(
        dataset_name='radar_debug', # A name for this debug run
        transform=transform,
        tokenizer=tokenizer,
        data_dir_list=[data_dir],
        num_used_data=[-1], # Use all files in the directory
        # Provide dummy values for DDP parameters for single-process debugging
        local_rank=0,
        world_size=1,
        num_workers=1 
    )

    # 5. Use a DataLoader to fetch a batch
    #    Set num_workers=0 for easier debugging in the main process.
    dataloader = DataLoader(dataset, batch_size=2, num_workers=0) 

    print("\nFetching one batch from the dataloader...")
    try:
        # Get the first batch from the iterator
        batch = next(iter(dataloader))
        
        print("\n--- ✅ Successfully fetched a batch! ---")
        print("Batch keys:", batch.keys())
        
        # Print details of the batch content
        print("\n--- Batch Content Details ---")
        print(f"  - image_tensor_list length: {len(batch['image_tensor_list'])}")
        if batch['image_tensor_list']:
            print(f"  - Shape of first image tensor: {batch['image_tensor_list'][0].shape}")
        
        print(f"  - text_ids_list length: {len(batch['text_ids_list'])}")
        if batch['text_ids_list']:
            print(f"  - Length of first text_ids: {len(batch['text_ids_list'][0])}")

        print(f"  - Estimated total tokens: {batch['num_tokens']}")
        print(f"  - Sequence Plan: {batch['sequence_plan']}")
        print(f"  - Data Indexes: {batch['data_indexes']}")

    except StopIteration:
        print("\n❌ Error: The dataloader is empty. This might happen if:")
        print(f"   - The directory '{data_dir}' contains no .parquet files.")
        print("   - The Parquet files are empty or corrupted.")
    except Exception as e:
        print(f"\n❌ An error occurred while fetching a batch: {e}")
        import traceback
        traceback.print_exc()

    print("\n--- Debug script finished ---")