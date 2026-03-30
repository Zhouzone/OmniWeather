# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

import io
import json
import random
import os
from PIL import Image
from datasets import load_dataset, load_from_disk

from .data_utils import pil_img2rgb
from .distributed_iterable_dataset import DistributedIterableDataset

Image.MAX_IMAGE_PIXELS = 20_000_000


class MetaQueryIterableDataset(DistributedIterableDataset):
    def __init__(
        self, dataset_name, transform, tokenizer, data_dir_list, num_used_data, 
        local_rank=0, world_size=1, num_workers=8, data_status=None, max_source_images=None,
    ):
        """
        data_dir_list: list of data directories contains arrow files
        num_used_data: list of number of sampled data paths for each data directory
        """
        super().__init__(dataset_name, local_rank, world_size, num_workers)
        self.transform = transform
        self.tokenizer = tokenizer
        self.data_status = data_status
        self.max_source_images = max_source_images
        
        if self.local_rank == 0:
            print(f"Initializing MetaQueryIterableDataset: {dataset_name}")
            print(f"  Data dirs: {data_dir_list}")
            print(f"  Num used data: {num_used_data}")
        
        if self.local_rank == 0:
            print("Loading datasets...")
        self.datasets = self.get_datasets(data_dir_list, num_used_data)
        
        if self.local_rank == 0:
            print("Getting data paths...")
        self.data_paths = self.get_data_paths(data_dir_list, num_used_data)
        
        if self.local_rank == 0:
            print("Setting epoch...")
        self.set_epoch()
        
        if self.local_rank == 0:
            print("MetaQueryIterableDataset initialization completed")

    def get_datasets(self, data_dir_list, num_used_data):
        """获取数据集对象"""
        datasets = []
        for i, (data_dir, num_data) in enumerate(zip(data_dir_list, num_used_data)):
            if self.local_rank == 0:
                print(f"Loading dataset {i+1}/{len(data_dir_list)}: {data_dir}")
            
            # 检查是否是已保存的数据集目录
            if os.path.exists(os.path.join(data_dir, "dataset_info.json")):
                if self.local_rank == 0:
                    print(f"  Loading from disk: {data_dir}")
                # 使用 load_from_disk 加载已保存的数据集
                dataset = load_from_disk(data_dir)
                if self.local_rank == 0:
                    print(f"  Loaded from disk successfully")
            else:
                if self.local_rank == 0:
                    print(f"  Scanning for arrow files...")
                # 兼容原有的 arrow 文件加载方式
                arrow_files = []
                for file in os.listdir(data_dir):
                    if file.endswith('.arrow'):
                        arrow_files.append(os.path.join(data_dir, file))
                
                # 按文件名排序，确保顺序一致
                arrow_files.sort()
                
                # 如果num_data > 0，限制使用的文件数量
                if num_data > 0:
                    arrow_files = arrow_files[:num_data]
                
                if self.local_rank == 0:
                    print(f"  Found {len(arrow_files)} arrow files, loading dataset...")
                
                # 加载所有选中的arrow文件
                dataset = load_dataset(
                    "arrow",
                    data_files=arrow_files,
                    split="train"
                )
                
                if self.local_rank == 0:
                    print(f"  Loaded arrow dataset successfully")
            
            datasets.append(dataset)
        
        if self.local_rank == 0:
            print(f"All datasets loaded successfully")
        return datasets

    def get_data_paths(self, data_dir_list, num_used_data):
        """获取数据集路径（用于基类兼容）"""
        # 返回文件路径列表，用于基类的分片逻辑
        data_paths = []
        for i, (data_dir, num_data) in enumerate(zip(data_dir_list, num_used_data)):
            if self.local_rank == 0:
                print(f"Scanning data paths {i+1}/{len(data_dir_list)}: {data_dir}")
            
            # 获取目录中所有arrow文件
            arrow_files = []
            for file in os.listdir(data_dir):
                if file.endswith('.arrow'):
                    arrow_files.append(os.path.join(data_dir, file))
            
            # 按文件名排序
            arrow_files.sort()
            
            # 如果num_data > 0，限制使用的文件数量
            if num_data > 0:
                arrow_files = arrow_files[:num_data]
            
            if self.local_rank == 0:
                print(f"  Found {len(arrow_files)} arrow files")
            
            data_paths.extend(arrow_files)
        
        if self.local_rank == 0:
            print(f"Total data paths: {len(data_paths)}")
        return data_paths

    def __iter__(self):
        data_paths_per_worker, worker_id = self.get_data_paths_per_worker()
        
        # 初始化数据状态
        if self.data_status is not None and worker_id in self.data_status:
            dataset_start_id = self.data_status[worker_id][0]
            row_start_id = self.data_status[worker_id][1] + 1
        else:
            dataset_start_id = 0
            row_start_id = 0
            
        transform_stride = self.transform.stride

        while True:
            # 遍历所有数据集
            for dataset_idx in range(len(self.datasets)):
                dataset = self.datasets[dataset_idx]
                
                # 确定起始行
                start_row = row_start_id if dataset_idx == dataset_start_id else 0
                
                # 从指定行开始处理
                dataset_subset = dataset.select(range(start_row, len(dataset)))
                
                for row_idx, row in enumerate(dataset_subset, start=start_row):
                    num_tokens = 0
                    image_tensor_list = []
                    text_ids_list = []
                    sequence_plan = []
                    
                    try:
                        # 处理指令文本（放在最前面）
                        prompt = row['prompt']
                        if prompt and prompt.strip():
                            prompt_token = self.tokenizer.encode(prompt.strip())
                            text_ids_list.append(prompt_token)
                            num_tokens += len(prompt_token)
                            sequence_plan.append({
                                'type': 'text',
                                'enable_cfg': 1,
                                'loss': 0,
                                'special_token_loss': 0,
                                'special_token_label': None,
                            })
                        
                        # 处理源图像（作为条件）
                        source_images = row['source_images']
                        try:
                            # 检查源图像数量限制
                            max_source_images = getattr(self, 'max_source_images', None)
                            if max_source_images is not None and len(source_images) > max_source_images:
                                if self.local_rank == 0:
                                    print(f"跳过源图像过多的样本: {len(source_images)} > {max_source_images}")
                                continue
                            
                            # 处理源图像列表中的每个图像
                            for source_image in source_images:
                                if source_image.mode != 'RGB':
                                    source_image = source_image.convert('RGB')
                                
                                source_tensor = self.transform(source_image)
                                height, width = source_tensor.shape[1:]
                                num_tokens += width * height // transform_stride ** 2
                                image_tensor_list.append(source_tensor)
                                
                                sequence_plan.append({
                                    'type': 'vae_image',
                                    'enable_cfg': 1,
                                    'loss': 0,
                                    'special_token_loss': 0,
                                    'special_token_label': None,
                                })
                            
                        except Exception as e:
                            if self.local_rank == 0:
                                print(f"Error processing source images: {e}")
                            continue
                        
                        # 处理目标图像（作为生成目标）
                        target_image = row['target_image']
                        try:
                            if target_image.mode != 'RGB':
                                target_image = target_image.convert('RGB')
                            
                            target_tensor = self.transform(target_image)
                            height, width = target_tensor.shape[1:]
                            num_tokens += width * height // transform_stride ** 2
                            image_tensor_list.append(target_tensor)
                            
                            sequence_plan.append({
                                'type': 'vae_image',
                                'enable_cfg': 0,
                                'loss': 1,
                                'special_token_loss': 0,
                                'special_token_label': None,
                            })
                            
                        except Exception as e:
                            if self.local_rank == 0:
                                print(f"Error processing target image: {e}")
                            continue
                        
                        sample = dict(
                            image_tensor_list=image_tensor_list,
                            text_ids_list=text_ids_list,
                            num_tokens=num_tokens,
                            sequence_plan=sequence_plan,
                            data_indexes={
                                "data_indexes": [dataset_idx, row_idx],
                                "worker_id": worker_id,
                                "dataset_name": self.dataset_name,
                            }
                        )
                        yield sample
                        
                    except Exception as e:
                        if self.local_rank == 0:
                            print(f"Error processing sample: {e}")
                        continue
                
                # 重置行起始位置，为下一个数据集做准备
                row_start_id = 0 