# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

import io
import json
import pandas as pd
import random
import base64
from PIL import Image
from .data_utils import pil_img2radar, pil_img2rgb
from .distributed_iterable_dataset import DistributedIterableDataset
from .parquet_utils import get_parquet_data_paths

Image.MAX_IMAGE_PIXELS = 20_000_000


class RadarForcastorDataset(DistributedIterableDataset):
    def __init__(
        self, dataset_name, transform, tokenizer, data_dir_list, num_used_data, 
        local_rank=0, world_size=1, num_workers=8, data_status=None,
        use_text_prompt=True,
    ):
        """
        data_dir_list: list of data directories contains parquet files
        num_used_data: list of number of sampled data paths for each data directory
        use_text_prompt: whether to include text prompts in the sequence
        """
        super().__init__(dataset_name, local_rank, world_size, num_workers)
        self.transform = transform
        self.tokenizer = tokenizer
        self.data_status = data_status
        self.use_text_prompt = use_text_prompt
        # 更详细的英文提示，包含EarthFormer模型背景
        self.prompt_text = "Given 10 frames VIL (Vertically Integrated Liquid) data, predict the next 12 frames of VIL (Vertically Integrated Liquid) data over the next 1 hour. Notice that the model input are initial predictions from the EarthFormer model for the next 12 frames of 1-hour VIL data."
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
                        
                        # 添加固定的文本提示
                        if self.use_text_prompt:
                            try:
                                # 编码固定的提示文本
                                prompt_tokens = self.tokenizer.encode(self.prompt_text)
                                text_ids_list.append(prompt_tokens)
                                num_tokens += len(prompt_tokens)
                                
                                # 添加文本到序列计划
                                sequence_plan.append({
                                    'type': 'text',
                                    'enable_cfg': 1,
                                    'loss': 0,  # 文本部分不计算损失
                                    'special_token_loss': 0,
                                    'special_token_label': None,
                                })
                                
                            except Exception as e:
                                print(f'Error processing prompt: {e} in row#{row_idx}')
                                continue
                        
                        # 处理输入图像 (caption字段: base64字符串列表) - 12帧输入
                        try:
                            input_list = row['caption']  # caption是input
                            for img_str in input_list:
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

                        # 处理目标图像 (image字段: base64字符串列表) - 12帧目标
                        try:
                            target_list = row['image']  # image是target
                            for cap_str in target_list:
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

                        # 合并输入和目标图像（时间顺序：先输入后目标）
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