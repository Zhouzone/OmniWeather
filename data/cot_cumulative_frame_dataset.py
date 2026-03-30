# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

import io
import json
import pandas as pd
import random
import base64
import numpy as np
from PIL import Image
from .data_utils import pil_img2rgb
from .distributed_iterable_dataset import DistributedIterableDataset
from .parquet_utils import get_parquet_data_paths

Image.MAX_IMAGE_PIXELS = 20_000_000


class CoTCumulativeFrameDataset(DistributedIterableDataset):
    """
    策略2: CoT累积输入预测训练数据集
    
    随着预测时间步的增长，输入包含更多的历史帧：
    - 预测第1帧：输入第1帧EarthFormer预测 → 真实GT第1帧
    - 预测第2帧：输入第1-2帧EarthFormer预测 → 真实GT第2帧
    - 预测第3帧：输入第1-3帧EarthFormer预测 → 真实GT第3帧
    - ...
    - 预测第12帧：输入第1-12帧EarthFormer预测 → 真实GT第12帧
    
    每个样本包含：
    1. 全局prompt（不计算loss）
    2. 帧索引和输入描述（不计算loss）
    3. CoT推理内容（计算loss）
    4. 累积的输入图像序列（不计算loss）
    5. 当前目标图像（计算loss）
    """
    
    def __init__(
        self, dataset_name, transform, tokenizer, data_dir_list, num_used_data, 
        local_rank=0, world_size=1, num_workers=8, data_status=None,
        use_text_prompt=True,
    ):
        """
        Args:
            dataset_name: 数据集名称
            transform: VAE图像变换
            tokenizer: 文本分词器
            data_dir_list: parquet文件目录列表
            num_used_data: 每个目录使用的数据量列表
            use_text_prompt: 是否使用文本prompt
        """
        super().__init__(dataset_name, local_rank, world_size, num_workers)
        self.transform = transform
        self.tokenizer = tokenizer
        self.data_status = data_status
        self.use_text_prompt = use_text_prompt
        
        # 特殊token定义，用于预测下一个token类型
        self.start_of_image = tokenizer.convert_tokens_to_ids('<|vision_start|>')
        self.end_of_image = tokenizer.convert_tokens_to_ids('<|vision_end|>')
        self.im_start = tokenizer.convert_tokens_to_ids('<|im_start|>')
        
        # 全局CoT提示文本
        self.global_prompt = (
            "You are an AI reasoning assistant for weather analysis, capable of backward causal reasoning: "
            "given an output VIL image, first identify outcome keywords, then infer the influencing causal factors "
            "as the reverse of forward reasoning"
        )
        
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
                    # 读取parquet文件
                    df = pd.read_parquet(parquet_file_path, engine="pyarrow")
                    df = df.iloc[row_start_id:]

                    samples_from_file = 0
                    for row_idx, row in df.iterrows():
                        try:
                            # 解码输入图像列表 (caption字段: 12帧EarthFormer预测)
                            input_list = row['caption']
                            
                            # 处理不同的数据类型
                            if isinstance(input_list, np.ndarray):
                                input_list = input_list.tolist()
                            elif not isinstance(input_list, list):
                                print(f'Warning: caption field is not list or array: {type(input_list)}')
                                continue
                            
                            input_images = []
                            for img_str in input_list:
                                try:
                                    img_bytes = base64.b64decode(img_str)
                                    image = pil_img2rgb(Image.open(io.BytesIO(img_bytes)))
                                    input_images.append(image)
                                except Exception as e:
                                    print(f'Error decoding input image: {e}')
                                    break

                            # 解码目标图像列表 (image字段: 12帧真实GT)
                            target_list = row['image']
                            
                            # 处理不同的数据类型
                            if isinstance(target_list, np.ndarray):
                                target_list = target_list.tolist()
                            elif not isinstance(target_list, list):
                                print(f'Warning: image field is not list or array: {type(target_list)}')
                                continue
                            
                            target_images = []
                            for cap_str in target_list:
                                try:
                                    cap_bytes = base64.b64decode(cap_str)
                                    cap_image = pil_img2rgb(Image.open(io.BytesIO(cap_bytes)))
                                    target_images.append(cap_image)
                                except Exception as e:
                                    print(f'Error decoding target image: {e}')
                                    break

                            # 获取CoT内容
                            cot_content = row.get('cot', '') or row.get('summary', '')
                            if not cot_content:
                                cot_content = 'No CoT available for this sample.'
                            
                            # 获取文件名
                            file_name = row.get('file_name', f'sample_{row_idx}')

                            # 确保输入和目标图像数量都是12帧
                            if len(input_images) != 12 or len(target_images) != 12:
                                print(f'Warning: incorrect image counts - input: {len(input_images)}, target: {len(target_images)}, expected 12 each')
                                continue

                            # 为每一帧生成累积输入的训练样本
                            for frame_idx in range(12):
                                num_tokens = 0
                                image_tensor_list = []
                                target_tensor_list = []
                                text_ids_list = []
                                sequence_plan = []
                                
                                # 1. 添加全局prompt（不计算loss）
                                if self.use_text_prompt:
                                    try:
                                        prompt_tokens = self.tokenizer.encode(self.global_prompt)
                                        text_ids_list.append(prompt_tokens)
                                        num_tokens += len(prompt_tokens)
                                        
                                        sequence_plan.append({
                                            'type': 'text',
                                            'enable_cfg': 1,
                                            'loss': 0,  # 全局prompt不计算损失
                                            'special_token_loss': 0,
                                            'special_token_label': None,
                                        })
                                    except Exception as e:
                                        print(f'Error processing global prompt: {e}')
                                        continue
                                
                                # 2. 添加帧索引和输入描述提示（不计算loss）
                                # 累积输入帧数 = frame_idx + 1 (第1帧到第frame_idx+1帧)
                                total_input_frames = frame_idx + 1
                                frame_prompt = (
                                    f"This is frame {frame_idx + 1} of 12. "
                                    f"Input contains {total_input_frames} EarthFormer prediction frames (frame 1 to {frame_idx + 1})."
                                )
                                try:
                                    frame_tokens = self.tokenizer.encode(frame_prompt)
                                    text_ids_list.append(frame_tokens)
                                    num_tokens += len(frame_tokens)
                                    
                                    sequence_plan.append({
                                        'type': 'text',
                                        'enable_cfg': 1,
                                        'loss': 0,  # 帧索引提示不计算损失
                                        'special_token_loss': 0,
                                        'special_token_label': None,
                                    })
                                except Exception as e:
                                    print(f'Error processing frame prompt: {e}')
                                    continue
                                
                                # 3. 添加CoT推理内容（计算loss）
                                try:
                                    # 将CoT内容包装在<think></think>标签中
                                    wrapped_cot_content = f"<think>{cot_content}</think>"
                                    cot_tokens = self.tokenizer.encode(wrapped_cot_content)
                                    text_ids_list.append(cot_tokens)
                                    num_tokens += len(cot_tokens)
                                    
                                    # CoT文本后面跟着图像，所以预测vision_start token
                                    sequence_plan.append({
                                        'type': 'text',
                                        'enable_cfg': 1,
                                        'loss': 1,  # CoT内容计算损失
                                        'special_token_loss': 1,  # 预测下一个特殊token
                                        'special_token_label': self.start_of_image,  # 预测vision_start
                                    })
                                except Exception as e:
                                    print(f'Error processing CoT content: {e}')
                                    continue
                                
                                # 4. 添加累积的输入图像序列（不计算loss）
                                # 添加第1帧到第frame_idx+1帧的EarthFormer预测作为输入
                                for input_idx in range(frame_idx + 1):
                                    try:
                                        input_image = input_images[input_idx]  # EarthFormer第input_idx+1帧预测
                                        image_tensor = self.transform(input_image)
                                        height, width = image_tensor.shape[1:]
                                        num_tokens += width * height // transform_stride ** 2
                                        image_tensor_list.append(image_tensor)
                                        
                                        sequence_plan.append({
                                            'type': 'vae_image',
                                            'enable_cfg': 1,  # 启用CFG作为条件
                                            'loss': 0,        # 输入图像不计算损失
                                            'special_token_loss': 0,
                                            'special_token_label': None,
                                        })
                                    except Exception as e:
                                        print(f'Error processing cumulative input image {input_idx}: {e}')
                                        continue

                                # 5. 添加当前目标图像（计算loss）
                                try:
                                    target_image = target_images[frame_idx]  # 真实GT第frame_idx+1帧
                                    target_tensor = self.transform(target_image)
                                    height, width = target_tensor.shape[1:]
                                    num_tokens += width * height // transform_stride ** 2
                                    target_tensor_list.append(target_tensor)
                                    
                                    sequence_plan.append({
                                        'type': 'vae_image',
                                        'enable_cfg': 0,  # 不启用CFG
                                        'loss': 1,        # 目标图像计算损失
                                        'special_token_loss': 0,
                                        'special_token_label': None,
                                    })
                                except Exception as e:
                                    print(f'Error processing target image: {e}')
                                    continue

                                # 合并图像张量（先输入后目标）
                                all_tensor_list = image_tensor_list + target_tensor_list

                                sample = dict(
                                    image_tensor_list=all_tensor_list,
                                    text_ids_list=text_ids_list,
                                    num_tokens=num_tokens,
                                    sequence_plan=sequence_plan,
                                    data_indexes={
                                        "data_indexes": [parquet_idx, frame_idx, row_idx],
                                        "worker_id": worker_id,
                                        "dataset_name": self.dataset_name,
                                        "original_row_idx": row_idx,
                                        "frame_idx": frame_idx,
                                        "total_frames": 12,
                                        "total_input_frames": total_input_frames,
                                        "file_name": file_name,
                                        "strategy": "cumulative_frame"
                                    }
                                )
                                samples_from_file += 1
                                yield sample

                        except Exception as e:
                            print(f'Error processing row {row_idx}: {e} in {parquet_file_path}')
                            continue

                    print(f"Generated {samples_from_file} samples from {parquet_file_path}")
                    row_start_id = 0
                except Exception as e:
                    print(f'Error reading parquet file {parquet_file_path}: {e}')
                    continue
                    
            parquet_start_id = 0
            # 添加计数器
            if not hasattr(self, '_repeat_count'):
                self._repeat_count = 0

            self._repeat_count += 1
            if self._repeat_count % 10 == 0:  # 每10次循环才打印一次
                print(f"{self.dataset_name} repeat #{self._repeat_count} in rank-{self.local_rank} worker-{worker_id}")
