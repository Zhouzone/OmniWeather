# data/bagel_sat2rad_dataset.py (最终修正版)
import torch
from torch.utils.data import IterableDataset
import numpy as np
import pandas as pd
import cv2
import os
import random
from PIL import Image
import json

# 导入Bagel项目中的分布式数据集基类
from .distributed_iterable_dataset import DistributedIterableDataset

# z-score归一化所需的常量
zscore_normalizations_sevir = {
    'vil':{'scale':47.54,'shift':33.44},
    'ir069':{'scale':1174.68,'shift':-3683.58},
    'ir107':{'scale':2562.43,'shift':-1552.80},
    'lght':{'scale':0.60517,'shift':0.02990},
    'vis':{'scale':2259.96,'shift':-1347.91}
}

class SevirAlldataset(DistributedIterableDataset):
    """
    A custom, distributed-aware IterableDataset for the Bagel model.
    This version corrects the super().__init__() call to match the parent class.
    """
    def __init__(self, dataset_name, tokenizer, transform, data_dir_list, num_used_data,variable_type=['vil'],data_type = 'sat2rad',
                 scale = '30min',phase='train', input_size=256, local_rank=0, world_size=1, num_workers=8,data_status=None,):
        
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
        self.data_type = data_type
        self.scale = scale
        
        # Load and filter the data catalog
        self.catalog = pd.read_csv(self.catalog_path, low_memory=False)
        required_img_types = set(['vis', 'ir069', 'ir107', 'vil'])
        if self.data_type == 'ir_to_vis':

            with open(os.path.join(os.path.dirname(__file__), 'train_daytime_index.json'), 'r') as f:
                data = json.load(f)
            
            self.daytime_samples = {item['event_id']: item['frame_ids'] for item in data}
            self.event_ids = list(self.daytime_samples.keys())
        else:
            
            events = self.catalog.groupby('id').filter(lambda x: required_img_types.issubset(set(x['img_type']))).groupby('id')
            
            all_event_ids = list(events.groups.keys())
            
            # Apply the train/test split to the event IDs
            if self.phase == 'train':
                self.event_ids = all_event_ids[:11458]
            else:
                self.event_ids = all_event_ids[11488:]

         
        task_configs = {
            'sat2rad': {
                'load_function': self._load_and_process_frame,
                'frame_count': 24 if self.phase == 'train' else 4,
                'prompt': "Generate a radar VIL image from the provided infrared satellite images."
            },
            'vil_down_scaling': {
                'load_function': self.get_sevir_vil_down_scaling,
                'frame_count': 24 if self.phase == 'train' else 4, 
                'prompt': "Generate a high-resolution VIL image from a low-resolution one."
            },
            'ir_down_scaling': {
                'load_function': self.get_sevir_vil_down_scaling,
                'frame_count': 24 if self.phase == 'train' else 4, 
                'prompt': "Generate a high-resolution ir069 image from a low-resolution one."
            },
            'interpolation': {
                'load_function': self.get_sevir_inter,
                'frame_count': 36, # This task can sample from frames 0-36
                'prompt': f"Interpolate the middle frame given the start and end frames at a {scale} interval."
            },
            'ir_translation': {
                'load_function': self.get_ir_trans,
                'frame_count': 24, # This task can sample any frame from 0-48
                'prompt': f"Translate the ir069 satellite image to an ir107 image."
            },
            'ir_to_vis': {
                'load_function': self.get_vis_reconstruction,
                'frame_count': 1, # This task can sample any of the 49 frames
                'prompt': "Generate a visible light (VIS) image from the provided infrared satellite images."
            },
            'rad_prediction': {
                'load_function': self.get_sevir_predict,
                'frame_count': 12, # Can start from frame 0 up to 12 to get a full sequence
                'prompt': "Forecast the next two radar frames given the previous four frames."
            },
            'ir_prediction': {
                'load_function': self.get_sevir_predict,
                'frame_count': 12, # Can start from frame 0 up to 12 to get a full sequence
                'prompt': "Forecast the next two ir069 frames given the previous four frames."
            },

            # Add new tasks here in the future
        }

        if data_type not in task_configs:
            raise ValueError(f"Unknown data_type '{data_type}'. Available types are: {list(task_configs.keys())}")
        
        config = task_configs[data_type]
        self.load_function = config['load_function']
        self.frame_count = config['frame_count']
        self.prompt = config['prompt']
            
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
            frame_indices = list(range(self.frame_count)) 
            

            data_paths_per_worker_ = data_paths_per_worker[event_start_id:]
            for event_idx, event_ids in enumerate(data_paths_per_worker_, start=event_start_id):
                for frame_idx_in_event in frame_indices:
                    # Load the event data
                    num_tokens = 0
                    sequence_plan = []
                    text_ids_list = []
                    image_tensor_list = []
                    target_tensor_list = []
                    if 'prediction' in self.data_type:
                        if self.data_type == 'rad_prediction':
                            input_pils, output_pils = self.load_function(event_ids, frame_idx_in_event,variable_type=['vil'])
                        elif self.data_type == 'ir_prediction':
                            input_pils, output_pils = self.load_function(event_ids, frame_idx_in_event, variable_type=['ir069'])
                        if not input_pils or not output_pils:
                            continue
                        

                        for image in input_pils:
                            image_tensor = self.transform(image)
                            image_tensor_list.append(image_tensor)
                            height, width = image_tensor.shape[1:]
                            num_tokens += width * height // transform_stride ** 2

                        for target in output_pils:
                            target_tensor = self.transform(target)
                            target_tensor_list.append(target_tensor)
                            height, width = target_tensor.shape[1:]
                            num_tokens += width * height // transform_stride ** 2
                        
                        for _ in image_tensor_list:
                                sequence_plan.append({
                                    'type': 'vae_image',
                                    'enable_cfg': 1,  # 启用CFG作为条件
                                    'loss': 0,        # 不计算损失
                                    'special_token_loss': 0,
                                    'special_token_label': None,
                                })

                        for _ in target_tensor_list:
                                sequence_plan.append({
                                    'type': 'vae_image',
                                    'enable_cfg': 0,  # 不启用CFG
                                    'loss': 1,        # 计算损失
                                    'special_token_loss': 0,
                                    'special_token_label': None,
                                })
                        
                        

                    else:
                        if any(k in self.data_type for k in ('down_scaling', 'inter')):
                            if self.data_type == 'vil_down_scaling':
                                sat_in_pil, rad_out_pil = self.load_function(event_ids, frame_idx_in_event, variable_type=['vil'], scale=False)
                            elif self.data_type == 'ir_down_scaling':
                                sat_in_pil, rad_out_pil = self.load_function(event_ids, frame_idx_in_event, variable_type=['ir069'], scale=False)
                            elif self.data_type == 'interpolation':
                                sat_in_pil, rad_out_pil = self.load_function(event_ids, frame_idx_in_event, variable_type=['vil'])
                        else:
                            sat_in_pil, rad_out_pil = self.load_function(event_ids,frame_idx_in_event)
                        sat_tensor = self.transform(sat_in_pil)
                        rad_tensor = self.transform(rad_out_pil)
                              
                        sequence_plan.append({
                                    'type': 'vae_image',
                                    'enable_cfg': 1,  # 启用CFG作为条件
                                    'loss': 0,        # 不计算损失
                                    'special_token_loss': 0,
                                    'special_token_label': None,
                                })
                        image_tensor_list.append(sat_tensor)
                        height, width = sat_tensor.shape[1:]
                        num_tokens += width * height // transform_stride ** 2
                        sequence_plan.append({
                                    'type': 'vae_image',
                                    'enable_cfg': 0,  
                                    'loss': 1,        
                                    'special_token_loss': 0,
                                    'special_token_label': None,
                                })  
                        target_tensor_list.append(rad_tensor)
                        height, width = rad_tensor.shape[1:]
                        num_tokens += width * height // transform_stride ** 2

                    text_ids = self.tokenizer.encode(self.prompt, add_special_tokens=True) 
                    num_tokens += len(text_ids)
                    sequence_plan.append({
                            'type': 'text',
                            'enable_cfg': 1,
                            'loss': 0,
                            'special_token_loss': 0,
                            'special_token_label': None,
                        }) 
                    
                    text_ids_list.append(text_ids)
                    all_tensor_list = image_tensor_list + target_tensor_list
                    sample = dict(
                        image_tensor_list=all_tensor_list,  # 先输入后目标
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


    def _load_and_process_frame(self, event_id, frame_idx_in_event,variable_type=['vil']):
        # This method remains the same
        frame_id = frame_idx_in_event * (2 if self.phase == 'train' else 10)
            
        row = self.catalog.loc[self.catalog['id'] == event_id].iloc[0]
        event_time = row['file_name'].split('/')[1]
        
        # --- Load 2-channel satellite input ---
        sat_in_pils = []
        for variable in ['ir069', 'ir107']:
            path = os.path.join(self.data_root, variable, event_time, f"{event_id}.npy")
            data = np.load(path)[:, :, frame_id]
            data = (data - zscore_normalizations_sevir[variable]['shift']) / zscore_normalizations_sevir[variable]['scale']
            # Convert to PIL Image for compatibility with the Bagel transform pipeline
            # Normalize to 0-255 range for standard image formats
            sat_in_pils.append(data)
        sat_in_pils = np.stack([sat_in_pils[0],sat_in_pils[1],sat_in_pils[1]],axis=-1)  
        H, W, _ = sat_in_pils.shape
        # Resize to the target size for Bagel
        if H != self.HQ_size or W != self.HQ_size:
            sat_in_pils = cv2.resize(np.copy(sat_in_pils), (self.HQ_size, self.HQ_size),interpolation=cv2.INTER_LINEAR)
        sat_in_pils = cv2.normalize(sat_in_pils, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        sat_in_pils = Image.fromarray(sat_in_pils).convert("RGB")

        # --- Load 1-channel radar output ---
        path = os.path.join(self.data_root, 'vil', event_time, f"{event_id}.npy")
        data = np.load(path)[:, :, frame_id]
        data = (data - zscore_normalizations_sevir['vil']['shift']) / zscore_normalizations_sevir['vil']['scale']
        H, W = data.shape
        if H != self.HQ_size or W != self.HQ_size:
            data = cv2.resize(np.copy(data), (self.HQ_size, self.HQ_size), interpolation=cv2.INTER_LINEAR)
        img_normalized = cv2.normalize(data, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        rad_out_pil = Image.fromarray(img_normalized).convert("RGB") # Convert to RGB

        return sat_in_pils, rad_out_pil
    
    def get_sevir_vil_down_scaling(self, event_id,frame_idx_in_event, variable_type=['vil'], scale=False):
        frame_id = frame_idx_in_event * (2 if self.phase == 'train' else 1)

        row = self.catalog.loc[self.catalog['id'] == event_id].iloc[0]
        event_time = row['file_name'].split('/')[1]

        in_variables = variable_type

        for variable in in_variables:
            # path = f"s3://sevir_pair/{variable}/{event_time}/{event}.npy"
            path = os.path.join(self.data_root, variable, event_time, f"{event_id}.npy")
            data = np.load(path)[:, :, frame_id]
            # normalization
            if variable == 'vil':
                data = (data-zscore_normalizations_sevir['vil']['shift'])/zscore_normalizations_sevir['vil']['scale']
                # data = np.expand_dims(data, axis=-1)
            if variable == 'ir069':
                data = (data-zscore_normalizations_sevir['ir069']['shift'])/zscore_normalizations_sevir['ir069']['scale']

            if variable == 'ir107':
                data = (data-zscore_normalizations_sevir['ir107']['shift'])/zscore_normalizations_sevir['ir107']['scale']


        vil_in = data[:,:]
        vil_out = data[:,:]
        vil_in = np.expand_dims(vil_in, axis=-1)
        vil_out = np.expand_dims(vil_out, axis=-1)
        if variable_type[0] == 'vil':
            if scale:
                vil_in = cv2.resize(np.copy(vil_in), (32, 32),
                        interpolation=cv2.INTER_LINEAR)
            # down_scaling: vil_in 384->256->64
            else:
                vil_in = cv2.resize(np.copy(vil_in), (64, 64),
                            interpolation=cv2.INTER_LINEAR)
        else:
            if scale:
                vil_in = cv2.resize(np.copy(vil_in), (32, 32),
                interpolation=cv2.INTER_LINEAR)
            else:
                vil_in = cv2.resize(np.copy(vil_in), (64, 64),
                            interpolation=cv2.INTER_LINEAR)
        vil_in = cv2.normalize(vil_in, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        vil_in = Image.fromarray(vil_in).convert("RGB")
        vil_out = cv2.normalize(vil_out, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        vil_out = Image.fromarray(vil_out).convert("RGB")
        return vil_in, vil_out
    # get_sevir_inter
    def get_sevir_inter(self, event_id, frame_idx_in_event,variable_type=['vil']):
        """
        Refactored data loading logic for the 'interpolation' task.
        Loads two VIL frames as input and the intermediate frame as output.
        """
        row = self.catalog.loc[self.catalog['id'] == event_id].iloc[0]
        event_time = row['file_name'].split('/')[1]

        # Determine the frame indices based on the specified scale
        if self.scale == '30min':
            # Frames at t, t+6, t+12 (30 min interval)
            data_list = [frame_idx_in_event, frame_idx_in_event + 6, frame_idx_in_event + 12]
        elif self.scale == '15min':
            # Frames at t, t+3, t+6 (15 min interval)
            data_list = [frame_idx_in_event, frame_idx_in_event + 3, frame_idx_in_event + 6]
        elif self.scale == '60min':
            # Frames at t, t+12, t+24 (60 min interval)
            data_list = [frame_idx_in_event, frame_idx_in_event + 12, frame_idx_in_event + 24]
        else:
            raise ValueError(f"Invalid scale for interpolation: {self.scale}")

        # Load the VIL data for the selected frames
        path = os.path.join(self.data_root, 'vil', event_time, f"{event_id}.npy")
        all_frames_data = np.load(path)
        
        # Check if all required frames are available
        if max(data_list) >= all_frames_data.shape[2]:
             return None, None # Skip if requested frames are out of bounds
        for variable in variable_type:
            data = all_frames_data[:, :, data_list]
            if variable == 'vil':
                    data = (data-zscore_normalizations_sevir['vil']['shift'])/zscore_normalizations_sevir['vil']['scale']
                    # data = np.expand_dims(data, axis=-1)
            if variable == 'ir069':
                data = (data-zscore_normalizations_sevir['ir069']['shift'])/zscore_normalizations_sevir['ir069']['scale']

        # The first and last frames are input, the middle one is output
        vil_in = data[:, :, [0, 2]]  # Shape (H, W, 2)
        vil_out = data[:, :, 1]    # Shape (H, W)

        # Create 3-channel input image by stacking the two input frames and a copy
        three_channel_data = np.dstack((vil_in[:, :, 0], vil_in[:, :, 1], vil_in[:, :, 0]))
        input_normalized = cv2.normalize(three_channel_data, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        input_pil = Image.fromarray(input_normalized, 'RGB')
        
        # Create 3-channel (grayscale) output image
        output_normalized = cv2.normalize(vil_out, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        output_pil = Image.fromarray(output_normalized).convert("RGB")

        return input_pil, output_pil
    
    def get_ir_trans(self, event_id, frame_idx_in_event):
        """
        Refactored data loading logic for the 'ir_translation' task.
        Loads one IR channel as input and another as output.
        """
        frame_id = frame_idx_in_event # This task samples directly from frames 0-48
        
        row = self.catalog.loc[self.catalog['id'] == event_id].iloc[0]
        event_time = row['file_name'].split('/')[1]

        # --- Load Input Image ---
        in_path = os.path.join(self.data_root, 'ir069', event_time, f"{event_id}.npy")
        in_data = np.load(in_path)[:, :, frame_id]
        in_data =  (in_data -zscore_normalizations_sevir['ir069']['shift'])/zscore_normalizations_sevir['ir069']['scale']
        
        # Convert single-channel input to a 3-channel PIL image by duplicating the channel
        in_normalized = cv2.normalize(in_data, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        input_pil = Image.fromarray(in_normalized).convert("RGB")

        # --- Load Output Image ---
        out_path = os.path.join(self.data_root, 'ir107' , event_time, f"{event_id}.npy")
        out_data = np.load(out_path)[:, :, frame_id]
        out_data = (out_data-zscore_normalizations_sevir['ir107']['shift'])/zscore_normalizations_sevir['ir107']['scale']

        # Convert single-channel output to a 3-channel (grayscale) PIL image
        out_normalized = cv2.normalize(out_data, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        output_pil = Image.fromarray(out_normalized).convert("RGB")
        
        return input_pil, output_pil
    
    def get_vis_reconstruction(self, event_id, frame_idx_in_event):
        """
        Refactored data loading logic for the 'ir_to_vis' task.
        Loads two IR channels as input and the VIS channel as output.
        """
        frame_id = frame_idx_in_event # This task samples directly from frames 0-48
        
        row = self.catalog.loc[self.catalog['id'] == event_id].iloc[0]
        event_time = row['file_name'].split('/')[1]
        frame_list = self.daytime_samples[event_id]
        #从frame_list中随机获取frame_id
        frame_id = random.choice(frame_list)
            

        # --- Load Input: 2 IR channels ---
        ir069_path = os.path.join(self.data_root, 'ir069', event_time, f"{event_id}.npy")
        ir069_data = np.load(ir069_path)[:, :, frame_id]
        ir069_data = (ir069_data - zscore_normalizations_sevir['ir069']['shift']) / zscore_normalizations_sevir['ir069']['scale']
        
        ir107_path = os.path.join(self.data_root, 'ir107', event_time, f"{event_id}.npy")
        ir107_data = np.load(ir107_path)[:, :, frame_id]
        ir107_data = (ir107_data - zscore_normalizations_sevir['ir107']['shift']) / zscore_normalizations_sevir['ir107']['scale']
        
        # Create a 3-channel input image from the two IR channels
        three_channel_data = np.stack([ir069_data, ir107_data, ir069_data], axis=-1)
        input_normalized = cv2.normalize(three_channel_data, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        input_pil = Image.fromarray(input_normalized, 'RGB')
        
        # --- Load Output: 1 VIS channel ---
        vis_path = os.path.join(self.data_root, 'vis', event_time, f"{event_id}.npy")
        vis_data = np.load(vis_path)[:, :, frame_id]
        vis_data = (vis_data - zscore_normalizations_sevir['vis']['shift']) / zscore_normalizations_sevir['vis']['scale']
        
        # Convert single-channel VIS to a 3-channel (grayscale) PIL image
        output_normalized = cv2.normalize(vis_data, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        output_pil = Image.fromarray(output_normalized).convert("RGB")
        
        return input_pil, output_pil
    
    def get_sevir_predict(self, event_id, frame_idx_in_event, variable_type=['vil']):
        """
        Refactored data loading logic for the 'prediction' (forecasting) task.
        Loads 4 VIL frames as input and 2 future VIL frames as output.
        Returns lists of PIL images.
        """
        row = self.catalog.loc[self.catalog['id'] == event_id].iloc[0]
        event_time = row['file_name'].split('/')[1]

        # Define the sequence of 6 frames needed for this task
        # Input: t, t+30, t+60, t+90 min -> Frames: start, start+6, start+12, start+18
        # Output: t+120, t+180 min -> Frames: start+24, start+36
        data_list = [frame_idx_in_event + offset for offset in [0, 6, 12, 18, 24, 36]]
        variable = variable_type[0] 
        path = os.path.join(self.data_root, variable, event_time, f"{event_id}.npy")
        all_frames_data = np.load(path)
        
        # Check if all required frames are available in the .npy file
        if max(data_list) >= all_frames_data.shape[2]:
            return [], [] # Return empty lists if out of bounds

        data = all_frames_data[:, :, data_list]
        data = (data - zscore_normalizations_sevir[variable]['shift']) / zscore_normalizations_sevir[variable]['scale']

        # Input is the first 4 frames
        vil_in = data[:, :, :4]
        # Output is the last 2 frames
        vil_out = data[:, :, 4:]

        # Process each frame into a PIL image
        input_pils = []
        for i in range(vil_in.shape[2]):
            frame_normalized = cv2.normalize(vil_in[:, :, i], None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            input_pils.append(Image.fromarray(frame_normalized).convert("RGB"))

        output_pils = []
        for i in range(vil_out.shape[2]):
            frame_normalized = cv2.normalize(vil_out[:, :, i], None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            output_pils.append(Image.fromarray(frame_normalized).convert("RGB"))
            
        return input_pils, output_pils
    
    def get_random_day_vis(self, ids):

        if self.not_finetune:
            if self.phase == 'train':
                event_id = ids // 24
                frame_id = (ids % 24)*2
            else:
                event_id = ids // 48
                frame_id = ids % 48
        else:
            if self.phase == 'train':
                event_id = ids // 4
                frame_id = ids % 4 + random.randint(0,44)
            else:
                event_id = ids // 48
                frame_id = ids % 48
        if event_id > len(self.event_ids)-1:
            event_id = random.randint(0, len(self.event_ids)-1)


        event = self.event_ids[event_id]
        event_time = self.get_line_time(event)

        path = f"s3://sevir_pair/vis/{event_time}/{event}.npy"
        data = io.BytesIO(client.get(path))
        data = np.load(data)
        data = data[:, :, frame_id]
        if data.mean() > 100 or self.phase != 'train':
            return event_id, frame_id
        else:
            if self.phase == 'train':
                if self.not_finetune:
                    random_ids = random.randint(0, len(self.event_ids)*24-1)
                else:
                    random_ids = random.randint(0, len(self.event_ids)-1)
            else:
                random_ids = random.randint(0, len(self.event_ids)*48-1)
            return self.get_random_day_vis(random_ids)