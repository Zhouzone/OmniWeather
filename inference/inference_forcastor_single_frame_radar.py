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


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_inferencer(model_path, offload_folder, max_mem_per_gpu="40GiB"):
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
        checkpoint=os.path.join(model_path, "ema.safetensors"),
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


def load_forcastor_single_frame_data(parquet_file_path, max_samples=None):
    """加载雷达预测数据集 - 单帧到单帧版本"""
    print(f"Loading forcastor single frame radar data from: {parquet_file_path}")
    
    # 使用单帧预测的提示文本
    prompt_text = "Given 12 frames VIL (Vertically Integrated Liquid) data from EarthFormer model predictions, predict the next 12 frames of VIL (Vertically Integrated Liquid) data. Notice that the model input are initial predictions from the EarthFormer model for the next 12 frames of 1-hour VIL data."
    
    try:
        # 使用pandas读取parquet文件
        df = pd.read_parquet(parquet_file_path, engine="fastparquet")
        
        if max_samples is not None:
            df = df.head(max_samples)
        
        samples = []
        for row_idx, row in tqdm(df.iterrows(), total=len(df), desc="Loading samples"):
            try:
                # 处理输入图像 (caption字段: base64字符串列表) - 12帧输入
                input_images = []
                input_list = row['caption']  # caption是input
                for img_str in input_list:
                    img_bytes = base64.b64decode(img_str)
                    image = pil_img2rgb(Image.open(io.BytesIO(img_bytes)))
                    input_images.append(image)
                
                # 处理目标图像 (image字段: base64字符串列表) - 12帧目标
                target_images = []
                target_list = row['image']  # image是target
                for cap_str in target_list:
                    cap_bytes = base64.b64decode(cap_str)
                    cap_image = pil_img2rgb(Image.open(io.BytesIO(cap_bytes)))
                    target_images.append(cap_image)
                
                # 确保输入和目标图像数量一致
                min_frames = min(len(input_images), len(target_images))
                if min_frames == 0:
                    continue
                
                # 为每一对 (input_frame, target_frame) 生成一个样本
                for frame_idx in range(min_frames):
                    sample = {
                        'input_image': input_images[frame_idx],  # 单张输入图像
                        'target_image': target_images[frame_idx],  # 单张目标图像
                        'prompt_text': prompt_text,  # 文本提示
                        'row_idx': row_idx,
                        'frame_idx': frame_idx,
                        'total_frames': min_frames,
                        'parquet_file': os.path.basename(parquet_file_path)
                    }
                    samples.append(sample)
                
            except Exception as e:
                print(f'Error processing row {row_idx}: {e}')
                continue
        
        print(f"Successfully loaded {len(samples)} single-frame samples from {parquet_file_path}")
        return samples
        
    except Exception as e:
        print(f'Error reading parquet file {parquet_file_path}: {e}')
        return []


def main(data_dir, output_folder, seed=42, thinking=False, offload_folder=None, 
         model_path="pretrained/ByteDance-Seed/BAGEL-7B-MoT", max_samples=None):
    set_seed(seed)
    if offload_folder is None:
        mode = "think" if thinking else "nothink"
        offload_folder = f"offload_{mode}_{int(time.time())}"
    
    print(f"Initializing BAGEL model from {model_path}")
    inferencer = build_inferencer(model_path, offload_folder)
    print("Model initialization completed")
    
    if thinking:
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

    # 获取所有parquet文件
    parquet_files = [f for f in os.listdir(data_dir) if f.endswith('.parquet')]
    parquet_files.sort()
    
    if not parquet_files:
        print(f"No parquet files found in {data_dir}")
        return
    
    # 创建输出目录
    os.makedirs(output_folder, exist_ok=True)
    img_output_dir = os.path.join(output_folder, 'generated_images')
    os.makedirs(img_output_dir, exist_ok=True)
    
    # 保存生成配置信息
    config_info = {
        'model_path': model_path,
        'thinking_mode': thinking,
        'inference_hyperparameters': inference_hyper,
        'seed': seed,
        'max_samples': max_samples,
        'data_dir': data_dir,
        'generation_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'description': 'Single-frame radar forecast generation using BAGEL model with 1 input image and text prompt to generate 1 future image',
        'dataset_type': 'forcastor_single_frame_radar'
    }
    
    config_path = os.path.join(output_folder, 'generation_config.json')
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config_info, f, indent=2, ensure_ascii=False)
    
    # 创建结果记录
    results = []
    json_output_path = os.path.join(output_folder, 'forcastor_single_frame_radar_inference_results.json')
    
    # 在main函数中，需要添加全局样本计数器
    global_sample_idx = 0

    # 处理每个parquet文件
    for parquet_file in tqdm(parquet_files, desc="Processing parquet files"):
        parquet_path = os.path.join(data_dir, parquet_file)
        samples = load_forcastor_single_frame_data(parquet_path, max_samples)
        
        if not samples:
            continue
        
        # 为每个parquet文件创建子目录
        parquet_name = os.path.splitext(parquet_file)[0]
        parquet_output_dir = os.path.join(img_output_dir, parquet_name)
        os.makedirs(parquet_output_dir, exist_ok=True)
        
        print(f"Processing {len(samples)} single-frame samples from {parquet_file}")
        
        # 按row_idx分组，每个row_idx对应一个完整的序列
        sequences = {}
        for sample in samples:
            row_idx = sample['row_idx']
            if row_idx not in sequences:
                sequences[row_idx] = []
            sequences[row_idx].append(sample)
        
        # 处理每个序列（每个row_idx）
        for row_idx, sequence_samples in sequences.items():
            # 按frame_idx排序
            sequence_samples.sort(key=lambda x: x['frame_idx'])
            
            # 为每个序列创建一个sample文件夹
            sample_output_dir = os.path.join(parquet_output_dir, f"sample_{global_sample_idx}")
            os.makedirs(sample_output_dir, exist_ok=True)
            
            # 保存输入图像（12帧）
            input_dir = os.path.join(sample_output_dir, "input")
            os.makedirs(input_dir, exist_ok=True)
            for i, sample in enumerate(sequence_samples):  # 全部12帧作为input
                sample['input_image'].save(os.path.join(input_dir, f"input_{i}.png"))
            
            # 保存目标图像（12帧）
            target_dir = os.path.join(sample_output_dir, "target")
            os.makedirs(target_dir, exist_ok=True)
            for i, sample in enumerate(sequence_samples):  # 全部12帧作为target
                sample['target_image'].save(os.path.join(target_dir, f"target_{i}.png"))
            
            # 保存文本提示
            prompt_path = os.path.join(sample_output_dir, "prompt.txt")
            with open(prompt_path, 'w', encoding='utf-8') as f:
                f.write(sequence_samples[0]['prompt_text'])
            
            # 生成图像
            generated_dir = os.path.join(sample_output_dir, "generated")
            os.makedirs(generated_dir, exist_ok=True)
            
            print(f"Processing sample {global_sample_idx} (row {row_idx}) with 12 input frames and 12 target frames")
            
            try:
                # 生成12张图像
                output_list = []
                for gen_idx in range(12):
                    # 检查设备状态
                    torch.cuda.empty_cache()
                    
                    # 构造输入列表：[所有12帧input图像, prompt_text]
                    input_images = [sample['input_image'] for sample in sequence_samples]
                    input_list = input_images + output_list + [sequence_samples[0]['prompt_text']]
                    
                    # 使用文本和图像进行推理
                    output_list = inferencer.interleave_inference(
                        input_lists=input_list,
                        think=thinking,
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
                        gen_path = os.path.join(generated_dir, f"generated_{gen_idx}.png")
                        generated_image.save(gen_path)
                        print(f"Generated image {gen_idx+1}/12 for sample {global_sample_idx}")
                    else:
                        print(f"Warning: No image generated for gen_idx {gen_idx} in sample {global_sample_idx}")
                        
            except Exception as gen_error:
                print(f"Error generating images for sample {global_sample_idx}: {gen_error}")
                continue
            
            # 记录结果
            result = {
                'parquet_file': parquet_file,
                'row_idx': row_idx,
                'sample_idx': global_sample_idx,
                'input_images_count': 12,  # 12帧输入
                'target_images_count': 12,  # 12帧目标
                'generated_images_count': 12,
                'prompt_text': sequence_samples[0]['prompt_text'],
                'output_path': f"generated_images/{parquet_name}/sample_{global_sample_idx}",
                'status': 'success'
            }
            
            print(f"[Success] Generated 12 images for sample {global_sample_idx} from {parquet_file}")
            
            results.append(result)
            
            # 实时更新JSON文件
            with open(json_output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
            # 清理GPU缓存
            torch.cuda.empty_cache()
            
            # 递增全局样本索引
            global_sample_idx += 1
    
    print(f"[Complete] All results saved to {output_folder}")
    print(f"[Complete] Generated images saved to {img_output_dir}")
    print(f"[Complete] JSON results saved to {json_output_path}")
    
    # 生成总结报告
    successful_samples = sum(1 for r in results if r['status'] == 'success')
    failed_samples = sum(1 for r in results if r['status'] == 'failed')
    total_generated_images = sum(r['generated_images_count'] for r in results)
    
    summary_report = {
        'total_samples_processed': len(results),
        'successful_samples': successful_samples,
        'failed_samples': failed_samples,
        'success_rate': f"{successful_samples/len(results)*100:.1f}%" if len(results) > 0 else "0%",
        'total_generated_images': total_generated_images,
        'average_images_per_sample': f"{total_generated_images/len(results):.1f}" if len(results) > 0 else "0",
        'processing_time': time.strftime('%Y-%m-%d %H:%M:%S'),
        'model_used': model_path,
        'thinking_mode': thinking,
        'dataset_type': 'forcastor_single_frame_radar'
    }
    
    summary_path = os.path.join(output_folder, 'summary_report.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary_report, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*60}")
    print("FORCASTOR SINGLE-FRAME RADAR GENERATION SUMMARY REPORT")
    print(f"{'='*60}")
    print(f"Total samples processed: {len(results)}")
    print(f"Successful generations: {successful_samples}")
    print(f"Failed generations: {failed_samples}")
    print(f"Success rate: {summary_report['success_rate']}")
    print(f"Total images generated: {total_generated_images}")
    print(f"Average images per sample: {summary_report['average_images_per_sample']}")
    print(f"Model used: {model_path}")
    print(f"Thinking mode: {thinking}")
    print(f"Dataset type: forcastor_single_frame_radar")
    print(f"Summary report saved to: {summary_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="雷达预测数据集推理脚本 - 单帧到单帧版本")
    parser.add_argument('--data_dir', type=str, required=False,
                       default='/mnt/shared-storage-user/sciprismax/science_uni/RadarQA-related/dataset4Bagel/generation/nowcast_sevir/test/png',
                       help='包含parquet文件的数据目录路径')
    parser.add_argument('--output_folder', type=str, required=False,
                       default='/mnt/shared-storage-user/zhouzhiwang/omni-weather/results/forcastor_single_gen_test',
                       help='输出文件夹路径')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--thinking', action='store_true', help='是否使用thinking模式')
    parser.add_argument('--offload_folder', type=str, default=None, 
                       help='offload磁盘缓存目录')
    parser.add_argument('--model_path', type=str,
                       default="/mnt/shared-storage-user/zhouzhiwang/bagel/models/BAGEL-7B-MoT",
                       help='模型权重路径')
    parser.add_argument('--max_samples', type=int, default=10, 
                       help='每个parquet文件处理的最大样本数（注意：单帧模式下实际样本数会乘以帧数）')
    
    args = parser.parse_args()
    main(args.data_dir, args.output_folder, args.seed, args.thinking, 
         args.offload_folder, args.model_path, args.max_samples)
