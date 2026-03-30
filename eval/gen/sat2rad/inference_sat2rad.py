# inference_sat2rad.py

import argparse
import os
import sys
import json
import time
import random
# from pandas.core import frame
import torch
import numpy as np
import pandas as pd
import cv2
from PIL import Image
from tqdm import tqdm
from matplotlib import pyplot as plt
import logging


# --- 1. 设置项目路径并导入所需模块 ---
# 确保可以从项目根目录导入模块``
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from modeling.qwen2 import Qwen2Tokenizer
from data.transforms import ImageTransform,ImageTransform2
from modeling.autoencoder import load_ae
from modeling.bagel import (
    Bagel, BagelConfig, Qwen2Config, Qwen2ForCausalLM, SiglipVisionConfig, SiglipVisionModel
)
from inferencer import InterleaveInferencer # 导入核心推理类
from safetensors.torch import load_file
from accelerate import infer_auto_device_map, load_checkpoint_and_dispatch, init_empty_weights
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from display import get_cmap

zscore_normalizations_sevir = {
    'vil':{'scale':47.54,'shift':33.44},
    'ir069':{'scale':1174.68,'shift':-3683.58},
    'ir107':{'scale':2562.43,'shift':-1552.80},
    'lght':{'scale':0.60517,'shift':0.02990},
    'vis':{'scale':2259.96,'shift':-1347.91},
    'no2':{'scale':2.5403,'shift':1.5490},
    'vil_china':{'scale':3.04,'shift':0.99},
}

minmax_normalizations = {
    'rainnet':{'min':0.0, 'max':140.0},
    'ZR_ir':{'min': [138.643, 54.002], 'max': [350.559, 320.947]},
    'vil_minmax':{'min':0.0, 'max':255.0}
}

vis_cmap,vis_norm,vis_vmin,vis_vmax = get_cmap('vis',encoded=True)
ir069_cmap,ir069_norm,ir069_vmin,ir069_vmax = get_cmap('ir069',encoded=True)
ir107_cmap,ir107_norm,ir107_vmin,ir107_vmax = get_cmap('ir107',encoded=True)
vil_cmap,vil_norm,vil_vmin,vil_vmax = get_cmap('vil',encoded=True)
vil_unet_cmap,vil_unet_norm,vil_vmin,vil_vmax = get_cmap('vil_unet',encoded=True)


thresholds = (16, 74, 133, 160, 181, 219)
total_pod = {thr: 0.0 for thr in thresholds}
total_far = {thr: 0.0 for thr in thresholds}
total_csi = {thr: 0.0 for thr in thresholds}

def setup_logger(logger_name, root, phase, level=logging.INFO, screen=False):
    '''Set up logger with file and optional console output.'''
    l = logging.getLogger(logger_name)
    
    if not l.hasHandlers():  # 确保没有重复添加handler
        formatter = logging.Formatter(
            '%(asctime)s.%(msecs)03d - %(levelname)s: %(message)s', datefmt='%y-%m-%d %H:%M:%S')

        # Ensure the log directory exists
        if not os.path.exists(root):
            os.makedirs(root, exist_ok=True)

        log_file = os.path.join(root, '{}.log'.format(phase))
        fh = logging.FileHandler(log_file, mode='w', encoding='utf-8')  # 确保以文本模式打开文件并设置编码
        fh.setFormatter(formatter)
        l.setLevel(level)
        l.addHandler(fh)

        # Optionally log to the screen
        if screen:
            sh = logging.StreamHandler()
            sh.setFormatter(formatter)
            l.addHandler(sh)

    return l

# --- 2. 辅助函数 ---
try:
    from data.transforms import ImageTransform,ImageTransform2
    from data.data_utils import add_special_tokens, pil_img2rgb
    print("✅ Successfully imported data modules")
except ImportError as e:
    print(f"❌ Failed to import data modules: {e}")
    print("Please check if you're running the script from the correct directory")
    sys.exit(1)


def save_pixel_image(ir069_inp,ir107_inp, pred_img, target_img, epoch, save_dir, idx,frame_id, model_name):
    
    save_path = os.path.join(save_dir, 'results', f'step{epoch}')
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
        
    if len(pred_img.shape) == 3:
        ir069_inp, ir107_inp ,pred_img, target_img = ir069_inp[0], ir107_inp[0], pred_img[0], target_img[0]
        
    #! pred_img 和 target_img 都是三通道的
    # num_channels = pred_img.shape[0]  # 通道数
    # if num_channels == 1:
    #     pred_img, target_img = pred_img[0], target_img[0]
    # else:
    #     random_number = random.randint(0, num_channels - 1)
    #     pred_img, target_img = pred_img[random_number], target_img[random_number]
        
    fig,(ax1, ax2,ax3,ax4) = plt.subplots(1, 4, figsize=(30, 6))
    ax1.imshow(ir069_inp, cmap=ir069_cmap, norm=ir069_norm)
    ax1.set_title(f'ir069_input_step{epoch}')
    ax2.imshow(ir107_inp, cmap=ir107_cmap, norm=ir107_norm)
    ax2.set_title(f'ir107_input_step{epoch}')
    # fig, (ax3, ax4) = plt.subplots(1, 2, figsize=(12, 6))
    ax3.imshow(pred_img, cmap=vil_cmap, norm=vil_norm)
    ax3.set_title(f'pred_pixel_step{epoch}')
    
    im2 = ax4.imshow(target_img, cmap=vil_cmap, norm=vil_norm)
    ax4.set_title(f'target_pixel')
    cbar1 = plt.colorbar(im2, ax=[ax3, ax4])
    # cbar1 = plt.colorbar(ax4.imshow(target_img, cmap=vil_cmap, norm=vil_norm), ax=[ax1, ax2, ax3, ax4])
    plt.axis('off')
    plt.savefig(os.path.join(save_path, f'{idx}_frame{frame_id}.png'), dpi=150, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

    

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def build_inferencer(model_path, ckpt_path, offload_folder, max_mem_per_gpu="40GiB"):
    """构建并加载完整的Bagel模型和推理器"""
    print("--- 正在构建模型 ---")
    # LLM config preparing
    llm_config = Qwen2Config.from_json_file(os.path.join(model_path, "llm_config.json"))
    llm_config.qk_norm = True
    llm_config.tie_word_embeddings = False
    llm_config.layer_module = "Qwen2MoTDecoderLayer"

    # ViT config preparing
    vit_config = SiglipVisionConfig.from_json_file(os.path.join(model_path, "vit_config.json"))
    vit_config.rope = False
    vit_config.num_hidden_layers = vit_config.num_hidden_layers - 1

    # VAE loading
    vae_model, vae_config = load_ae(local_path=os.path.join(model_path, "ae.safetensors"))

    # Bagel config preparing
    config = BagelConfig(
        visual_gen=True,
        visual_und=True,
        llm_config=llm_config, 
        vit_config=vit_config,
        vae_config=vae_config,
        vit_max_num_patch_per_side=70,
        connector_act='gelu_pytorch_tanh',
        latent_patch_size=2,
        max_latent_size=64,
    )

    with init_empty_weights():
        language_model = Qwen2ForCausalLM(llm_config)
        vit_model      = SiglipVisionModel(vit_config)
        model          = Bagel(language_model, vit_model, config)
        model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config, meta=True)

    # Tokenizer Preparing
    tokenizer = Qwen2Tokenizer.from_pretrained(model_path)
    tokenizer, new_token_ids, _ = add_special_tokens(tokenizer)

    # Image Transform Preparing
    vae_transform = ImageTransform(1024, 256, 16)
    vit_transform = ImageTransform(980, 224, 14)

    device_map = infer_auto_device_map(
        model,
        max_memory={i: max_mem_per_gpu for i in range(torch.cuda.device_count())},
        no_split_module_classes=["Bagel", "Qwen2MoTDecoderLayer"],
    )
    same_device_modules = [
        'language_model.model.embed_tokens',
        'time_embedder',
        'latent_pos_embed',
        'vae2llm',
        'llm2vae',
        'connector',
        'vit_pos_embed'
    ]
    
    # 修复设备映射问题 - 参考inference.ipynb的实现
    if torch.cuda.device_count() == 1:
        first_device = device_map.get(same_device_modules[0], "cuda:0")
        for k in same_device_modules:
            if k in device_map:
                device_map[k] = first_device
            else:
                device_map[k] = "cuda:0"
    else:
        first_device = device_map.get(same_device_modules[0])
        for k in same_device_modules:
            if k in device_map:
                device_map[k] = first_device

    model = load_checkpoint_and_dispatch(
        model,
        checkpoint=os.path.join(ckpt_path, "ema.safetensors"),
        device_map=device_map,
        offload_buffers=True,
        dtype=torch.bfloat16,
        offload_folder=offload_folder,
        force_hooks=True,
    )
    model = model.eval()
    
    # 确保VAE模型也在正确的设备上
    # 获取模型的主要设备
    main_device = next(model.parameters()).device
    vae_model = vae_model.to(main_device).eval()
    
    # 确保所有组件都在正确的设备上
    torch.cuda.synchronize()
    print("Device synchronization completed")

    from inferencer import InterleaveInferencer
    inferencer = InterleaveInferencer(
        model=model, 
        vae_model=vae_model, 
        tokenizer=tokenizer, 
        vae_transform=vae_transform, 
        vit_transform=vit_transform, 
        new_token_ids=new_token_ids
    )
    return inferencer

# --- 3. 为 sat2rad 定制的数据加载函数 ---

ZSCORE_NORMALIZATIONS = {
    'vil': {'scale': 47.54, 'shift': 33.44},
    'ir069': {'scale': 1174.68, 'shift': -3683.58},
    'ir107': {'scale': 2562.43, 'shift': -1552.80},
}

def load_sat2rad_data_for_inference(data_root, catalog_path, max_samples=None):
    """
    为sat2rad数据集加载推理数据，返回一个样本列表。
    每个样本是一个包含输入和目标PIL图像的字典。
    """
    print(f"正在从目录 '{catalog_path}' 和数据根目录 '{data_root}' 加载数据...")
    
    catalog = pd.read_csv(catalog_path, low_memory=False)
    required_img_types = {'vis', 'ir069', 'ir107', 'vil'}
    events = catalog.groupby('id').filter(lambda x: required_img_types.issubset(set(x['img_type']))).groupby('id')
    
    # 使用测试集部分的数据
    event_ids = list(events.groups.keys())[11458:]
    # event_ids = list(events.groups.keys())[-2:-1]

    if max_samples is not None:
        event_ids = event_ids[:max_samples]
    
    samples = []
    for event_id in tqdm(event_ids, desc="加载推理样本"):
        try:
            for frame in range(4):
                frame_id = frame
                row = catalog[catalog['id'] == event_id].iloc[0]
                event_time = row['file_name'].split('/')[1]
                
                # 加载并合并输入图像
                ir069_path = os.path.join(data_root, 'ir069', event_time, f"{event_id}.npy")
                ir069_data = np.load(ir069_path)[:, :, frame_id * 10]
                ir069_data = (ir069_data - ZSCORE_NORMALIZATIONS['ir069']['shift']) / ZSCORE_NORMALIZATIONS['ir069']['scale']
                ir107_path = os.path.join(data_root, 'ir107', event_time, f"{event_id}.npy")
                ir107_data = np.load(ir107_path)[:, :, frame_id * 10]
                ir107_data = (ir107_data - ZSCORE_NORMALIZATIONS['ir107']['shift']) / ZSCORE_NORMALIZATIONS['ir107']['scale']
                three_channel_data = np.stack([ir069_data, ir107_data, ir107_data], axis=-1)
                H, W,_ = three_channel_data.shape
                if H != 256 or W != 256:
                    three_channel_data = cv2.resize(np.copy(three_channel_data), (256,256), interpolation=cv2.INTER_LINEAR)

                ir_min_max = [three_channel_data.min(),three_channel_data.max()]
                three_channel_normalized = cv2.normalize(three_channel_data, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                input_image = Image.fromarray(three_channel_normalized, 'RGB')


                # 加载目标图像
                vil_path = os.path.join(data_root, 'vil', event_time, f"{event_id}.npy")
                vil_data = np.load(vil_path)[:, :, frame_id * 10]
                print('vil_data1:', vil_data.min(), vil_data.max())
                vil_data = (vil_data - ZSCORE_NORMALIZATIONS['vil']['shift']) / ZSCORE_NORMALIZATIONS['vil']['scale']


                H, W = vil_data.shape
                if H != 256 or W != 256:
                    vil_data = cv2.resize(vil_data, (256,256), interpolation=cv2.INTER_LINEAR)
                vil_min_max = [vil_data.min(), vil_data.max()]

                vil_normalized = cv2.normalize(vil_data, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                target_image = Image.fromarray(vil_normalized).convert("RGB")
                
                np_img = np.array(target_image)
                vil_data = torch.from_numpy(np_img).permute(2,0,1)[ 0:1, :, :].float().div(255.0)
                vil_data = (vil_data * zscore_normalizations_sevir['vil']['scale'] + zscore_normalizations_sevir['vil']['shift'])

                
                samples.append({
                    'input_image': input_image,
                    'target_image': target_image,
                    'event_id': event_id,
                    'ir_minmax': ir_min_max,
                    'vil_minmax': vil_min_max,
                    'frame_id': frame_id*10
                })
        except Exception as e:
            print(f"警告：处理事件 {event_id} 时出错，已跳过。错误: {e}")
            continue
        
    print(f"成功加载 {len(samples)} 个样本用于推理。")
    return samples

# --- 4. 主推理逻辑 ---

@torch.no_grad()
def main(args):
    step = args.ckpt_path.split('/')[-1]
    set_seed(args.seed)
    save_log = os.path.join(args.output_folder, 'results', f'step{step}')
    logger_path = os.path.join(save_log, 'validation_logger')
    logger_val = setup_logger('val_logger', logger_path, 'val')
    
    # 加载您的sat2rad数据
    samples = load_sat2rad_data_for_inference(args.data_dir, args.catalog_path, args.max_samples)
    # 建立推理器
    inferencer = build_inferencer(args.model_path, args.ckpt_path, args.offload_folder)

    if not samples:
        print("没有可处理的数据，程序退出。")
        return
        
    # 设置推理超参数
    inference_hyper = dict(
        cfg_text_scale=4.0, cfg_img_scale=2.0, cfg_interval=[0.0, 1.0],
        timestep_shift=4.0, num_timesteps=50, cfg_renorm_min=1.0,
        cfg_renorm_type="text_channel",
    )

    # 创建输出目录
    os.makedirs(args.output_folder, exist_ok=True)
    print(f"推理结果将保存在: {args.output_folder}")
    rmse = 0.0
    # 循环处理每个样本
    for i, sample in enumerate(tqdm(samples, desc="正在生成图像")):
        
        input_image = sample['input_image']
        # 转换为tensor
        Transform = ImageTransform2(1024, 256, 16)
        input_tensor = Transform(input_image).unsqueeze(0)
        ir_min,ir_max = sample['ir_minmax']
        input_tensor = input_tensor* (ir_max - ir_min) + ir_min
        ir069_tensor = input_tensor[:, 0:1, :, :]
        ir107_tensor = input_tensor[:, 1:2, :, :]

        scale_069 = zscore_normalizations_sevir['ir069']['scale']
        shift_069 = zscore_normalizations_sevir['ir069']['shift']
        ir069_tensor = (ir069_tensor * scale_069 + shift_069) 
        scale_107 = zscore_normalizations_sevir['ir107']['scale']
        shift_107 = zscore_normalizations_sevir['ir107']['shift']
        ir107_tensor = (ir107_tensor * scale_107 + shift_107) 
        

        target_image = sample['target_image']
        event_id = sample['event_id']
        
        target_tensor = Transform(target_image).unsqueeze(0)
        vil_min,vil_max = sample['vil_minmax']
        target_tensor = target_tensor* (vil_max - vil_min) + vil_min
        vil_tensor0 = target_tensor[:, 0:1, :, :]
        vil_tensor = (vil_tensor0 * zscore_normalizations_sevir['vil']['scale'] + zscore_normalizations_sevir['vil']['shift'])
        # vil_tensor2 = np.load("your_path_here")[:,:,0:1]
        # #转换为tensor
        # vil_tensor2= torch.from_numpy(vil_tensor2).permute(2, 0, 1).float()


        # 定义文本提示
        prompt = "Generate a radar VIL image from the provided infrared satellite images."
        
        # 将文本和图像打包成输入列表
        input_list = [prompt, input_image]
        
        # 调用推理函数
        output_list = inferencer.interleave_inference(
            input_lists=input_list,
            think=args.thinking,
            understanding_output=False,
            **inference_hyper
        )
        
        # 从输出中提取生成的图像
        generated_image = None
        for output in output_list:
            if isinstance(output, Image.Image):
                generated_image = output
                break
        
        if generated_image is None:
            print(f"警告：样本 {i} (Event ID: {event_id}) 未能生成图像。")
            continue
        gen_tensor = Transform(generated_image).unsqueeze(0)
        
        gen_tensor = gen_tensor* (vil_max - vil_min) + vil_min
        gen_tensor0 = gen_tensor[:, 0:1, :, :]
        mse = torch.mean((gen_tensor0 - vil_tensor0) ** 2).item()
        rmse += np.sqrt(mse)


        gen_tensor = (gen_tensor0 * zscore_normalizations_sevir['vil']['scale'] + zscore_normalizations_sevir['vil']['shift'])
        # gen_tensor = target_tensor

        save_pixel_image(
            ir069_inp=ir069_tensor[0],
            ir107_inp=ir069_tensor[0],
            pred_img=gen_tensor[0],
            target_img=vil_tensor[0],
            epoch=args.ckpt_path.split('/')[-1],
            save_dir=args.output_folder,
            idx=sample['event_id'],
            frame_id=sample['frame_id'],
            model_name='bagel'
        )
        prediction = gen_tensor.detach().cpu()[0]
        label = vil_tensor.cpu()[0]
        for thr in thresholds:
            has_event_target = (label >= thr)
            has_event_predict = (prediction >= thr)
            
            hit = torch.sum(has_event_target & has_event_predict).item()
            miss = torch.sum(has_event_target & ~has_event_predict).item()
            false_alarm = torch.sum(~has_event_target & has_event_predict).item()
            no_event = torch.sum(~has_event_target).item()
            
            # 确保分母不为零，避免除零错误
            hit_miss = hit + miss
            hit_miss_false = hit + miss + false_alarm

            pod = hit / hit_miss if hit_miss > 0 else 0.0
            far = false_alarm / no_event if no_event > 0 else 0.0
            csi = hit / hit_miss_false if hit_miss_false > 0 else 0.0

            total_pod[thr] += pod
            total_far[thr] += far
            total_csi[thr] += csi
            
            logger_val.info(f"Epoch: {args.ckpt_path.split('/')[-1]}, Validation idx: {i}, Threshold: {thr}, POD: {pod:.4f}, FAR: {far:.4f}, CSI: {csi:.4f}")
    
    avg_rmse = rmse / i
    logger_val.info('# validation # RMSE: {:.4e}'.format(avg_rmse))
    
    # 如果需要，可以在此处计算平均的 POD、FAR、CSI
    num_samples = i
    for thr in thresholds:
        avg_pod = total_pod[thr] / num_samples
        avg_far = total_far[thr] / num_samples
        avg_csi = total_csi[thr] / num_samples
        logger_val.info(f"Threshold: {thr}, Avg POD: {avg_pod:.4f}, Avg FAR: {avg_far:.4f}, Avg CSI: {avg_csi:.4f}")


    print(f"\n✅ 推理全部完成！请检查 '{args.output_folder}' 目录下的对比图像。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="用于 sat2rad 数据集的 Bagel 推理脚本")
    
    # --- 必须指定的路径 ---
    parser.add_argument("--ckpt_path", type=str,default="/mnt/shared-storage-user/zhouzhiwang/bagel/experiments/012_Bagel_radarqa_servir/checkpoints/0020000", help="您训练好的模型检查点（checkpoint）的路径。")

    # --- 可选的路径和配置 ---
    parser.add_argument("--model_path", type=str, default="/mnt/shared-storage-user/zhouzhiwang/bagel/models/BAGEL-7B-MoT", help="预训练Bagel模型的路径，用于加载tokenizer和VAE。")
    parser.add_argument("--data_dir", type=str, default="/mnt/shared-storage-user/sciprismax/science_uni/sevir_full", help="SEVIR .npy 数据的根目录。")
    parser.add_argument("--catalog_path", type=str, default="/mnt/shared-storage-user/sciprismax/science_uni/sevir_full/output.csv", help="output.csv 目录文件的路径。")
    parser.add_argument("--output_folder", type=str, default="/mnt/shared-storage-user/zhouzhiwang/omni-weather/results/sat2rad_gen_test", help="保存输出图像的目录。")
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--thinking', action='store_true', help='是否使用thinking模式')
    parser.add_argument('--offload_folder', type=str, default="offload_cache", help='offload磁盘缓存目录')
    parser.add_argument('--max_samples', type=int, default=5, help='要处理的最大样本数，用于快速测试。')
    
    args = parser.parse_args()
    main(args)