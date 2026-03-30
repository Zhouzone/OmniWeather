# data/bagel_sat2rad_dataset.py (最终修正版)
import torch
from torch.utils.data import IterableDataset
import numpy as np
import pandas as pd
import cv2
import os
import random
from PIL import Image

# 导入Bagel项目中的分布式数据集基类
from .distributed_iterable_dataset import DistributedIterableDataset

# z-score归一化所需的常量（仅用于卫星数据）
zscore_normalizations_sevir = {
    'ir069':{'scale':1174.68, 'shift':-3683.58},
    'ir107':{'scale':2562.43, 'shift':-1552.80},
}

class BagelSat2RadDataset(DistributedIterableDataset):
    """
    A custom, distributed-aware IterableDataset for the Bagel model.
    Modified to handle IR069, IR107, and VIL data separately with proper normalization.
    """
    def __init__(self, dataset_name, tokenizer, transform, data_dir_list, num_used_data,
                 phase='train', input_size=256, local_rank=0, world_size=1, num_workers=8,data_status=None,):
        
        # --- FIX: Call super().__init__() FIRST with the correct arguments ---
        # The parent class needs these to set up distributed properties.
        # We extract them from kwargs, which are passed by the framework.
        super().__init__(dataset_name, local_rank, world_size, num_workers)

        
        # --- Now, define all other attributes for this class ---
        
        # Look up path information from the global config
        from .dataset_info import DATASET_INFO
        dataset_config_key = list(DATASET_INFO[dataset_name].keys())[0]
        self.dataset_info = DATASET_INFO[dataset_name][dataset_config_key]
        
        self.data_root = self.dataset_info['data_dir']
        self.catalog_path = self.dataset_info['catalog_path']
        
        # Store other necessary parameters
        self.tokenizer = tokenizer
        self.transform = transform
        self.phase = phase
        self.HQ_size = input_size
        self.data_status = data_status
        
        # Load and filter the data catalog
        self.catalog = pd.read_csv(self.catalog_path, low_memory=False)
        required_img_types = set(['vis', 'ir069', 'ir107', 'vil'])
        events = self.catalog.groupby('id').filter(lambda x: required_img_types.issubset(set(x['img_type']))).groupby('id')
        
        all_event_ids = list(events.groups.keys())
        
        # Apply the train/test split to the event IDs
        if self.phase == 'train':
            self.event_ids = all_event_ids[:11487]
        else:
            self.event_ids = all_event_ids[11488:]
            
        # The parent class needs to know the total number of items to distribute.
        # We update its internal list length here.
        self.data_paths = self.event_ids
        self.set_epoch()

    def get_line_time(self, value):
        #! get year information
        row = self.catalog.loc[self.catalog['id'] == value].iloc[0]
        time = row['file_name'].split('/')[1]
        return time   
    

    def __iter__(self):
        data_paths_per_worker, worker_id = self.get_data_paths_per_worker()
        if self.data_status is not None:
            event_start_id = self.data_status[worker_id][0]
            row_start_id = self.data_status[worker_id][2] + 1
        else:
            event_start_id = 0
            row_start_id = 0
        transform_stride = self.transform.stride
        print(
            f"rank-{self.local_rank} worker-{worker_id} dataset-{self.dataset_name}: "
            f"resuming data at event #{event_start_id}, row#{row_start_id}"
        )
        while True:
            frame_indices = list(range(24)) if self.phase == 'train' else list(range(4))

            data_paths_per_worker_ = data_paths_per_worker[event_start_id:]
            for event_idx, event_ids in enumerate(data_paths_per_worker_, start=event_start_id):
                for frame_idx_in_event in frame_indices:
                # Load the event data
                    
                    num_tokens = 0
                    ir069_pil, ir107_pil, vil_pil = self._load_and_process_frame(event_ids, frame_idx_in_event)
                    
                    # Transform each channel separately
                    ir069_tensor = self.transform(ir069_pil)
                    ir107_tensor = self.transform(ir107_pil)
                    vil_tensor = self.transform(vil_pil)
                    
                    # Fixed prompt for IR069 and IR107 combination
                    prompt = "Generate a radar VIL (Vertically Integrated Liquid) data from the provided IR069 and IR107 satellite infrared images."
                    
                    text_ids = self.tokenizer.encode(prompt, add_special_tokens=True) 
                    image_tensor_list = []
                    target_tensor_list = []
                    text_ids_list = []
                    sequence_plan = []      

                    # Text prompt
                    sequence_plan.append({
                        'type': 'text',
                        'enable_cfg': 1,
                        'loss': 0,
                        'special_token_loss': 0,
                        'special_token_label': None,
                    })
                         
                    # Input satellite images: IR069 first, then IR107
                    # IR069 input image
                    sequence_plan.append({
                        'type': 'vae_image',
                        'enable_cfg': 1,  # 启用CFG作为条件
                        'loss': 0,        # 不计算损失
                        'special_token_loss': 0,
                        'special_token_label': None,
                    })
                    image_tensor_list.append(ir069_tensor)
                    height, width = ir069_tensor.shape[1:]
                    num_tokens += width * height // transform_stride ** 2
                    
                    # IR107 input image
                    sequence_plan.append({
                        'type': 'vae_image',
                        'enable_cfg': 1,  # 启用CFG作为条件
                        'loss': 0,        # 不计算损失
                        'special_token_loss': 0,
                        'special_token_label': None,
                    })
                    image_tensor_list.append(ir107_tensor)
                    height, width = ir107_tensor.shape[1:]
                    num_tokens += width * height // transform_stride ** 2
                    
                    # Target VIL image
                    sequence_plan.append({
                        'type': 'vae_image',
                        'enable_cfg': 0,  
                        'loss': 1,        
                        'special_token_loss': 0,
                        'special_token_label': None,
                    })  
                    target_tensor_list.append(vil_tensor)
                    height, width = vil_tensor.shape[1:]
                    num_tokens += width * height // transform_stride ** 2
                    num_tokens += len(text_ids)
                    
                    
                    text_ids_list.append(text_ids)
                    all_tensor_list = image_tensor_list + target_tensor_list
                    sample = dict(
                        image_tensor_list=all_tensor_list,  # [ir069, ir107, vil]
                        text_ids_list=text_ids_list,
                        num_tokens=num_tokens,
                        sequence_plan=sequence_plan,
                        data_indexes={
                            "data_indexes": [event_idx, 0, 0],
                            "worker_id": worker_id,
                            "dataset_name": self.dataset_name,
                        }
                    )             
                    
                    yield sample

            event_start_id = 0
            print(f"{self.dataset_name} repeat in rank-{self.local_rank} worker-{worker_id}")


    def _load_and_process_frame(self, event_id, frame_idx_in_event):
        """
        加载并处理单帧数据，返回三个独立的PIL图像
        每个通道都repeat成3通道RGB图像
        """
        frame_id = frame_idx_in_event * (2 if self.phase == 'train' else 10)
            
        row = self.catalog.loc[self.catalog['id'] == event_id].iloc[0]
        event_time = row['file_name'].split('/')[1]
        
        # --- 处理IR069卫星数据 ---
        ir069_path = os.path.join(self.data_root, 'ir069', event_time, f"{event_id}.npy")
        ir069_data = np.load(ir069_path)[:, :, frame_id]
        # Z-score归一化
        ir069_data = (ir069_data - zscore_normalizations_sevir['ir069']['shift']) / zscore_normalizations_sevir['ir069']['scale']
        # 调整大小
        H, W = ir069_data.shape
        if H != self.HQ_size or W != self.HQ_size:
            ir069_data = cv2.resize(np.copy(ir069_data), (self.HQ_size, self.HQ_size), interpolation=cv2.INTER_LINEAR)
        # 归一化到0-255并repeat成3通道
        ir069_normalized = cv2.normalize(ir069_data, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        ir069_rgb = np.stack([ir069_normalized, ir069_normalized, ir069_normalized], axis=-1)
        ir069_pil = Image.fromarray(ir069_rgb).convert("RGB")
        
        # --- 处理IR107卫星数据 ---
        ir107_path = os.path.join(self.data_root, 'ir107', event_time, f"{event_id}.npy")
        ir107_data = np.load(ir107_path)[:, :, frame_id]
        # Z-score归一化
        ir107_data = (ir107_data - zscore_normalizations_sevir['ir107']['shift']) / zscore_normalizations_sevir['ir107']['scale']
        # 调整大小
        H, W = ir107_data.shape
        if H != self.HQ_size or W != self.HQ_size:
            ir107_data = cv2.resize(np.copy(ir107_data), (self.HQ_size, self.HQ_size), interpolation=cv2.INTER_LINEAR)
        # 归一化到0-255并repeat成3通道
        ir107_normalized = cv2.normalize(ir107_data, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        ir107_rgb = np.stack([ir107_normalized, ir107_normalized, ir107_normalized], axis=-1)
        ir107_pil = Image.fromarray(ir107_rgb).convert("RGB")

        # --- 处理VIL雷达数据 ---
        vil_path = os.path.join(self.data_root, 'vil', event_time, f"{event_id}.npy")
        vil_data = np.load(vil_path)[:, :, frame_id]
        # 调整大小
        H, W = vil_data.shape
        if H != self.HQ_size or W != self.HQ_size:
            vil_data = cv2.resize(np.copy(vil_data), (self.HQ_size, self.HQ_size), interpolation=cv2.INTER_LINEAR)
        # 转换为uint8并repeat成3通道
        vil_data = vil_data.astype(np.uint8)
        vil_rgb = np.stack([vil_data, vil_data, vil_data], axis=-1)
        vil_pil = Image.fromarray(vil_rgb).convert("RGB")

        return ir069_pil, ir107_pil, vil_pil
    

