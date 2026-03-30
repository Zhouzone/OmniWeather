
import os
import json
import argparse
from PIL import Image
from tqdm import tqdm
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import random
import numpy as np
from copy import deepcopy
from typing import Any, Dict
import pandas as pd
import base64
import io
from accelerate import infer_auto_device_map, load_checkpoint_and_dispatch, init_empty_weights
from data.transforms import ImageTransform
from data.data_utils import add_special_tokens, pil_img2rgb
from modeling.bagel import (
    BagelConfig, Bagel, Qwen2Config, Qwen2ForCausalLM, SiglipVisionConfig, SiglipVisionModel
)
from modeling.qwen2 import Qwen2Tokenizer
from modeling.bagel.qwen2_navit import NaiveCache
from modeling.autoencoder import load_ae
from safetensors.torch import load_file
import time
from matplotlib import pyplot as plt

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'eval', 'gen', 'sat2rad'))
from display import get_cmap

# Z-score归一化常量
zscore_normalizations_sevir = {
    'ir069': {'scale': 1174.68, 'shift': -3683.58},
    'ir107': {'scale': 2562.43, 'shift': -1552.80},
    'vil': {'scale': 47.54, 'shift': 33.44},
}

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def save_pixel_image(ir069_inp, ir107_inp, pred_img, target_img, save_dir, idx, frame_id, model_name):
    """
    保存卫星到雷达的对比图像
    
    Args:
        ir069_inp: IR069输入图像 (2D numpy数组)
        ir107_inp: IR107输入图像 (2D numpy数组) 
        pred_img: 预测的VIL图像 (2D numpy数组)
        target_img: 真实的VIL图像 (2D numpy数组)
        save_dir: 保存目录
        idx: 图像索引
        frame_id: 帧ID
        model_name: 模型名称
    """
    # 获取对应的colormap
    ir069_cmap, ir069_norm, ir069_vmin, ir069_vmax = get_cmap('ir069', encoded=True)
    ir107_cmap, ir107_norm, ir107_vmin, ir107_vmax = get_cmap('ir107', encoded=True)
    vil_cmap, vil_norm, vil_vmin, vil_vmax = get_cmap('vil', encoded=True)
    
    # 创建保存路径
    save_path = os.path.join(save_dir, 'results')
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
    
    # 确保输入是2D数组
    if len(ir069_inp.shape) == 3:
        ir069_inp = ir069_inp[0] if ir069_inp.shape[0] == 1 else ir069_inp[:, :, 0]
    if len(ir107_inp.shape) == 3:
        ir107_inp = ir107_inp[0] if ir107_inp.shape[0] == 1 else ir107_inp[:, :, 0]
    if len(pred_img.shape) == 3:
        pred_img = pred_img[0] if pred_img.shape[0] == 1 else pred_img[:, :, 0]
    if len(target_img.shape) == 3:
        target_img = target_img[0] if target_img.shape[0] == 1 else target_img[:, :, 0]
    
    # 创建图像
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(20, 5))
    
    # 显示IR069输入
    ax1.imshow(ir069_inp, cmap=ir069_cmap, norm=ir069_norm)
    ax1.set_title(f'IR069 Input - Frame {frame_id}', fontsize=12)
    ax1.axis('off')
    
    # 显示IR107输入
    ax2.imshow(ir107_inp, cmap=ir107_cmap, norm=ir107_norm)
    ax2.set_title(f'IR107 Input - Frame {frame_id}', fontsize=12)
    ax2.axis('off')
    
    # 显示预测的VIL
    ax3.imshow(pred_img, cmap=vil_cmap, norm=vil_norm)
    ax3.set_title(f'Predicted VIL - {model_name}', fontsize=12)
    ax3.axis('off')
    
    # 显示真实的VIL
    ax4.imshow(target_img, cmap=vil_cmap, norm=vil_norm)
    ax4.set_title(f'Ground Truth VIL', fontsize=12)
    ax4.axis('off')
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图像
    filename = f'{idx}_frame{frame_id}_sat2rad.png'
    filepath = os.path.join(save_path, filename)
    plt.savefig(filepath, dpi=150, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)
    
    print(f"Sat2Rad comparison image saved: {filepath}")
    return filepath

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
    
    # 修复设备映射问题 - 参考inference_sat2rad.py的实现
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

    # 关键修正：使用ckpt_path加载checkpoint
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

def load_sat2rad_data_from_dataset(dataset, max_samples=None):
    """从BagelSat2RadDataset加载数据用于推理"""
    print(f"Loading sat2rad data from dataset...")
    
    # Sat2Rad任务的固定提示文本
    prompt_text = "Generate a radar VIL (Vertically Integrated Liquid) data from the provided IR069 and IR107 satellite infrared images."
    
    samples = []
    sample_count = 0
    
    try:
        # 获取数据集的迭代器
        data_iter = iter(dataset)
        
        for sample in data_iter:
            try:
                # 从数据集样本中提取信息
                image_tensor_list = sample['image_tensor_list']  # [ir069_tensor, ir107_tensor, vil_tensor]
                text_ids_list = sample['text_ids_list']
                data_indexes = sample['data_indexes']
                
                if len(image_tensor_list) != 3:
                    print(f"Warning: Expected 3 images (IR069, IR107, VIL), got {len(image_tensor_list)}")
                    continue
                
                # 提取张量
                ir069_tensor = image_tensor_list[0]  # [C, H, W]
                ir107_tensor = image_tensor_list[1]  # [C, H, W]
                vil_tensor = image_tensor_list[2]    # [C, H, W]
                
                # 转换张量为PIL图像
                # 注意：数据集中每个通道都是重复3次的RGB格式
                def tensor_to_pil_single_channel(tensor):
                    """将单通道重复3次的张量转换为PIL图像"""
                    if tensor.dim() == 3:  # [C, H, W]
                        # 取第一个通道（因为3个通道都是重复的）
                        single_channel = tensor[0:1, :, :]  # [1, H, W]
                        # 重复3次形成RGB
                        rgb_tensor = single_channel.repeat(3, 1, 1)  # [3, H, W]
                    else:
                        rgb_tensor = tensor
                    
                    # 转换为numpy并调整维度顺序
                    rgb_array = rgb_tensor.permute(1, 2, 0).numpy()  # [H, W, 3]
                    # 确保数值在0-255范围内
                    rgb_array = np.clip(rgb_array * 255, 0, 255).astype(np.uint8)
                    return Image.fromarray(rgb_array, 'RGB')
                
                # 创建输入图像（IR069 + IR107组合）
                # 由于数据集已经将IR069和IR107分别处理为3通道，我们需要组合它们
                ir069_pil = tensor_to_pil_single_channel(ir069_tensor)
                ir107_pil = tensor_to_pil_single_channel(ir107_tensor)
                
                # 组合IR069和IR107为6通道输入（每个通道重复3次）
                # 这里我们创建一个包含两个卫星通道信息的图像
                ir069_array = np.array(ir069_pil)[:, :, 0]  # 取单通道
                ir107_array = np.array(ir107_pil)[:, :, 0]  # 取单通道
                
                # 创建6通道图像：IR069重复3次 + IR107重复3次
                combined_array = np.stack([
                    ir069_array, ir069_array, ir069_array,  # IR069重复3次
                    ir107_array, ir107_array, ir107_array   # IR107重复3次
                ], axis=-1)  # [H, W, 6]
                
                # 但是PIL只支持RGB，所以我们先创建一个假的RGB图像用于推理
                # 实际推理时使用前3个通道（IR069）
                input_image = Image.fromarray(combined_array[:, :, :3], 'RGB')
                
                # 目标图像
                target_image = tensor_to_pil_single_channel(vil_tensor)
                
                # 为了可视化，我们需要保存原始的单通道数据
                ir069_data = ir069_array.astype(np.float32)
                ir107_data = ir107_array.astype(np.float32)
                vil_data = np.array(target_image)[:, :, 0].astype(np.float32)
                
                sample_dict = {
                    'input_image': input_image,           # 用于推理的PIL图像
                    'target_image': target_image,         # 目标VIL图像
                    'ir069_data': ir069_data,             # 原始IR069数据（用于可视化）
                    'ir107_data': ir107_data,             # 原始IR107数据（用于可视化）
                    'vil_data': vil_data,                 # 原始VIL数据（用于可视化）
                    'prompt_text': prompt_text,           # 文本提示
                    'sample_idx': sample_count,
                    'data_indexes': data_indexes,
                    'frame_id': data_indexes.get('data_indexes', [0, 0, 0])[1] if 'data_indexes' in data_indexes else 0
                }
                
                samples.append(sample_dict)
                sample_count += 1
                
                if max_samples is not None and sample_count >= max_samples:
                    break
                    
            except Exception as e:
                print(f'Error processing sample {sample_count}: {e}')
                continue
        
        print(f"Successfully loaded {len(samples)} sat2rad samples")
        return samples
        
    except Exception as e:
        print(f'Error loading data from dataset: {e}')
        return []

def main(args):
    """主推理函数 - 参考inference_sat2rad.py的结构"""
    set_seed(args.seed)
    if args.offload_folder is None:
        mode = "think" if args.thinking else "nothink"
        args.offload_folder = f"offload_{mode}_{int(time.time())}"
    
    print(f"Initializing BAGEL model from {args.model_path}")
    print(f"Loading checkpoint from {args.ckpt_path}")
    
    # 构建推理器 - 注意传入ckpt_path
    inferencer = build_inferencer(args.model_path, args.ckpt_path, args.offload_folder)
    print("Model initialization completed")
    
    # 设置推理参数
    if args.thinking:
        inference_hyper = dict(
            max_think_token_n=1000,
            do_sample=False,
            cfg_text_scale=4.0,
            cfg_img_scale=2.0,
            cfg_interval=[0.4, 1.0],
            timestep_shift=3.0,
            num_timesteps=50,
            cfg_renorm_min=0.0,
            cfg_renorm_type="text_channel",
        )
    else:
        inference_hyper = dict(
            cfg_text_scale=4.0,
            cfg_img_scale=2.0,
            cfg_interval=[0.0, 1.0],
            timestep_shift=4.0,
            num_timesteps=50,
            cfg_renorm_min=1.0,
            cfg_renorm_type="text_channel",
        )

    # 创建数据集实例
    print("Creating Sat2Rad dataset...")
    from data.sat2rad_dataset import BagelSat2RadDataset
    from data.transforms import ImageTransform
    from modeling.qwen2 import Qwen2Tokenizer
    
    # 加载tokenizer和transform（用于数据集初始化）
    tokenizer = Qwen2Tokenizer.from_pretrained(args.model_path)
    transform = ImageTransform(max_image_size=256, min_image_size=256, image_stride=16)
    
    # 创建数据集
    dataset = BagelSat2RadDataset(
        dataset_name='sat2rad_inference',
        tokenizer=tokenizer,
        transform=transform,
        data_dir_list=[args.data_dir],
        num_used_data=[args.max_samples if args.max_samples else -1],
        phase='test',  # 使用测试阶段
        input_size=256,
        local_rank=0,
        world_size=1,
        num_workers=1
    )
    
    # 从数据集加载样本
    samples = load_sat2rad_data_from_dataset(dataset, args.max_samples)
    
    if not samples:
        print("No samples loaded from dataset")
        return
    
    # 创建输出目录
    os.makedirs(args.output_folder, exist_ok=True)
    img_output_dir = os.path.join(args.output_folder, 'generated_images')
    os.makedirs(img_output_dir, exist_ok=True)
    
    # 保存生成配置信息
    step = args.ckpt_path.split('/')[-1]
    config_info = {
        'model_path': args.model_path,
        'ckpt_path': args.ckpt_path,
        'checkpoint_step': step,
        'thinking_mode': args.thinking,
        'inference_hyperparameters': inference_hyper,
        'seed': args.seed,
        'max_samples': args.max_samples,
        'data_dir': args.data_dir,
        'catalog_path': args.catalog_path,
        'generation_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'description': 'Sat2Rad generation using BAGEL model: IR069+IR107 satellite images to VIL radar data',
        'dataset_type': 'sat2rad'
    }
    
    config_path = os.path.join(args.output_folder, 'generation_config.json')
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config_info, f, indent=2, ensure_ascii=False)
    
    # 创建结果记录
    results = []
    json_output_path = os.path.join(args.output_folder, 'sat2rad_inference_results.json')
    
    print(f"Processing {len(samples)} sat2rad samples")
    
    # 处理每个样本
    for i, sample in enumerate(tqdm(samples, desc="Generating VIL from satellite images")):
        try:
            # 为每个样本创建输出目录
            sample_output_dir = os.path.join(img_output_dir, f"sample_{i}")
            os.makedirs(sample_output_dir, exist_ok=True)
            
            # 保存输入图像
            input_dir = os.path.join(sample_output_dir, "input")
            os.makedirs(input_dir, exist_ok=True)
            sample['input_image'].save(os.path.join(input_dir, "satellite_input.png"))
            
            # 保存目标图像
            target_dir = os.path.join(sample_output_dir, "target")
            os.makedirs(target_dir, exist_ok=True)
            sample['target_image'].save(os.path.join(target_dir, "vil_target.png"))
            
            # 保存文本提示
            prompt_path = os.path.join(sample_output_dir, "prompt.txt")
            with open(prompt_path, 'w', encoding='utf-8') as f:
                f.write(sample['prompt_text'])
            
            # 生成图像
            generated_dir = os.path.join(sample_output_dir, "generated")
            os.makedirs(generated_dir, exist_ok=True)
            
            print(f"Processing sample {i}: satellite → VIL")
            
            # 检查设备状态
            torch.cuda.empty_cache()
            
            # 构造输入列表：[prompt_text, input_image]
            input_list = [sample['prompt_text'], sample['input_image']]
            
            # 使用文本和图像进行推理
            output_list = inferencer.interleave_inference(
                input_lists=input_list,
                think=args.thinking,
                understanding_output=False,
                **inference_hyper
            )
            
            # 获取生成的图像        
            generated_image = None
            for output in output_list:
                if isinstance(output, Image.Image):
                    generated_image = output
                    break
            
            if generated_image is not None:
                # 保存生成的图像
                gen_path = os.path.join(generated_dir, "generated_vil.png")
                generated_image.save(gen_path)
                
                # 转换生成图像为numpy数组用于可视化
                generated_array = np.array(generated_image)
                # 检查是否是3通道重复的情况
                if len(generated_array.shape) == 3 and generated_array.shape[2] == 3:
                    # 检查三个通道是否相同
                    data_dim1 = generated_array[:, :, 0]
                    data_dim2 = generated_array[:, :, 1]
                    data_dim3 = generated_array[:, :, 2]
                    
                    print(f"Channel differences: (1-2)={np.abs(data_dim1 - data_dim2).mean():.6f}, "
                          f"(1-3)={np.abs(data_dim1 - data_dim3).mean():.6f}, "
                          f"(2-3)={np.abs(data_dim2 - data_dim3).mean():.6f}")
                    
                    # 使用第一个通道作为生成的VIL数据
                    generated_vil_data = data_dim1.astype(np.float32)
                else:
                    generated_vil_data = generated_array.astype(np.float32)
                
                # 创建对比可视化
                save_pixel_image(
                    ir069_inp=sample['ir069_data'],
                    ir107_inp=sample['ir107_data'],
                    pred_img=generated_vil_data,
                    target_img=sample['vil_data'],
                    save_dir=sample_output_dir,
                    idx=i,
                    frame_id=sample['frame_id'],
                    model_name=step
                )
                
                print(f"Generated VIL for sample {i}")
                status = 'success'
            else:
                print(f"Warning: No image generated for sample {i}")
                status = 'failed'
            
            # 记录结果
            result = {
                'sample_idx': i,
                'frame_id': sample['frame_id'],
                'prompt_text': sample['prompt_text'],
                'output_path': f"generated_images/sample_{i}",
                'status': status,
                'data_indexes': sample['data_indexes']
            }
            
            results.append(result)
            
            # 实时更新JSON文件
            with open(json_output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
            # 清理GPU缓存
            torch.cuda.empty_cache()
            
        except Exception as gen_error:
            print(f"Error generating image for sample {i}: {gen_error}")
            result = {
                'sample_idx': i,
                'frame_id': sample.get('frame_id', 0),
                'prompt_text': sample.get('prompt_text', ''),
                'output_path': f"generated_images/sample_{i}",
                'status': 'failed',
                'error': str(gen_error)
            }
            results.append(result)
            continue
    
    print(f"[Complete] All results saved to {args.output_folder}")
    print(f"[Complete] Generated images saved to {img_output_dir}")
    print(f"[Complete] JSON results saved to {json_output_path}")
    
    # 生成总结报告
    successful_samples = sum(1 for r in results if r['status'] == 'success')
    failed_samples = sum(1 for r in results if r['status'] == 'failed')
    
    summary_report = {
        'total_samples_processed': len(results),
        'successful_samples': successful_samples,
        'failed_samples': failed_samples,
        'success_rate': f"{successful_samples/len(results)*100:.1f}%" if len(results) > 0 else "0%",
        'processing_time': time.strftime('%Y-%m-%d %H:%M:%S'),
        'model_used': args.model_path,
        'checkpoint_used': args.ckpt_path,
        'checkpoint_step': step,
        'thinking_mode': args.thinking,
        'dataset_type': 'sat2rad'
    }
    
    summary_path = os.path.join(args.output_folder, 'summary_report.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary_report, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*60}")
    print("SAT2RAD GENERATION SUMMARY REPORT")
    print(f"{'='*60}")
    print(f"Total samples processed: {len(results)}")
    print(f"Successful generations: {successful_samples}")
    print(f"Failed generations: {failed_samples}")
    print(f"Success rate: {summary_report['success_rate']}")
    print(f"Model used: {args.model_path}")
    print(f"Checkpoint used: {args.ckpt_path} (step {step})")
    print(f"Thinking mode: {args.thinking}")
    print(f"Dataset type: sat2rad")
    print(f"Summary report saved to: {summary_path}")
    print(f"{'='*60}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="用于 sat2rad 数据集的 Bagel 推理脚本")
    
    # --- 必须指定的路径 ---
    parser.add_argument("--ckpt_path", type=str,
                       default="/mnt/shared-storage-user/zhouzhiwang/bagel/experiments/012_Bagel_radarqa_servir/checkpoints/0020000",
                       help="训练好的模型检查点（checkpoint）的路径")
    
    # --- 可选的路径和配置 ---
    parser.add_argument("--model_path", type=str,
                       default="/mnt/shared-storage-user/zhouzhiwang/bagel/models/BAGEL-7B-MoT",
                       help="预训练Bagel模型的路径，用于加载tokenizer和VAE")
    parser.add_argument("--data_dir", type=str,
                       default="/mnt/shared-storage-user/sciprismax/science_uni/sevir_full",
                       help="SEVIR .npy 数据的根目录")
    parser.add_argument("--catalog_path", type=str,
                       default="/mnt/shared-storage-user/sciprismax/science_uni/sevir_full/output.csv",
                       help="output.csv 目录文件的路径")
    parser.add_argument("--output_folder", type=str,
                       default="/mnt/shared-storage-user/zhouzhiwang/omni-weather/results/sat2rad_gen_test",
                       help="保存输出图像的目录")
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--thinking', action='store_true', help='是否使用thinking模式')
    parser.add_argument('--offload_folder', type=str, default=None, 
                       help='offload磁盘缓存目录')
    parser.add_argument('--max_samples', type=int, default=10, 
                       help='要处理的最大样本数，用于快速测试')
    
    args = parser.parse_args()
    main(args)

