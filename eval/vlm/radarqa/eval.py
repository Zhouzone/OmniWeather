import argparse
import os
import json
import ast
import itertools
import random

import torch
from eval.vlm.utils import load_model_and_tokenizer, build_transform, process_conversation
from PIL import Image
from tqdm import tqdm

# 添加视频处理支持
import sys
sys.path.append('data')
from video_utils import read_frames_decord


def parse_answer(answer_str):
    try:
        return ast.literal_eval(answer_str)
    except (ValueError, SyntaxError):
        return {}


class RadarQADataset(torch.utils.data.Dataset):

    def __init__(self, test_file, prompt=""):
        self.prompt = prompt
        with open(test_file, 'r', encoding='utf-8') as f:
            self.data = [json.loads(line) for line in f]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        
        # 检查数据结构类型
        if 'conversations' in data:
            # 检查是图像还是视频数据
            if 'image' in data:
                # 图像数据格式
                image_paths = data['image']
                # image_paths = image_paths.replace('your_old_prefix_here', 'your_new_prefix_here')
                image_paths = [
                    p.replace(
                        'your_old_prefix_here',
                        'your_new_prefix_here'
                    )
                    for p in image_paths
                ]
                conversations = data['conversations']

                images = [Image.open(p).convert('RGB') for p in image_paths]

                system_prompt = ''
                question = ''
                gt = ''
                for conv in conversations:
                    if conv['from'] == 'system':
                        system_prompt = conv['value']
                    elif conv['from'] == 'human':
                        question = conv['value'].replace('<image>', '').strip()
                    elif conv['from'] == 'gpt':
                        gt = conv['value']

                prompt = f"{system_prompt}\n{question}"

            elif 'video' in data:
                # 视频数据格式
                video_paths = data['video']
                video_paths = [
                    p.replace(
                        'your_old_prefix_here',
                        'your_new_prefix_here'
                    )
                    for p in video_paths
                ]
                conversations = data['conversations']
                
                # 从视频中提取帧
                images = []
                for video_path in video_paths:
                    # 从每个视频中提取8帧
                    video_frames = read_frames_decord(video_path, num_frames=8, sample='rand')
                    images.extend(video_frames)
                
                system_prompt = ''
                question = ''
                gt = ''
                for conv in conversations:
                    if conv['from'] == 'system':
                        system_prompt = conv['value']
                    elif conv['from'] == 'human':
                        # 替换 <video> 标签为 <image> 标签
                        question = conv['value'].replace('<video>', '<image>').strip()
                    elif conv['from'] == 'gpt':
                        gt = conv['value']
                
                prompt = f"{system_prompt}\n{question}"
            else:
                raise ValueError("数据中既没有 'image' 也没有 'video' 字段")
                
        else:
            # 新格式：包含 query、images、response
            if 'images' in data:
                image_paths = data['images']
                image_paths = [
                        p.replace(
                            'your_old_prefix_here',
                            'your_new_prefix_here'
                        )
                        for p in image_paths
                    ]
                query = data['query']
                gt = data['response']
                
                images = [Image.open(p).convert('RGB') for p in image_paths]
                prompt = query
            else:
                raise ValueError("无法识别的数据格式")
        
        # 使用 process_conversation 处理图像和对话
        images, conversation = process_conversation(images, prompt)
        
        return {
            'id': data.get('id', idx),
            'images': images,
            'conversation': conversation,
            'ground_truth': gt
        }


def collate_fn(batch):
    ids = [_['id'] for _ in batch]
    images = [_['images'] for _ in batch]
    conversations = [_['conversation'] for _ in batch]
    ground_truths = [_['ground_truth'] for _ in batch]
    return ids, images, conversations, ground_truths


class InferenceSampler(torch.utils.data.sampler.Sampler):

    def __init__(self, size):
        self._size = int(size)
        assert size > 0
        self._rank = torch.distributed.get_rank()
        self._world_size = torch.distributed.get_world_size()
        self._local_indices = self._get_local_indices(size, self._world_size, self._rank)

    @staticmethod
    def _get_local_indices(total_size, world_size, rank):
        shard_size = total_size // world_size
        left = total_size % world_size
        shard_sizes = [shard_size + int(r < left) for r in range(world_size)]
        begin = sum(shard_sizes[:rank])
        end = min(sum(shard_sizes[:rank + 1]), total_size)
        return range(begin, end)

    def __iter__(self):
        yield from self._local_indices

    def __len__(self):
        return len(self._local_indices)


def is_dict_format(answer_str):
    """检查答案是否为字典格式"""
    try:
        parsed = ast.literal_eval(answer_str)
        return isinstance(parsed, dict)
    except:
        return False


def calculate_accuracy(results_file):
    """计算准确率，支持两种格式：字典格式和文本格式"""
    with open(results_file, 'r', encoding='utf-8') as f:
        first_line = f.readline().strip()
        if not first_line:
            print("No results found.")
            return
        
        # 检查第一个样本的格式
        first_data = json.loads(first_line)
        gt = first_data['ground_truth']
        
        if is_dict_format(gt):
            # 字典格式：计算各项指标的准确率
            scores = {
                'Overall Performance': 0,
                'Miss Performance': 0,
                'False Alarm Performance': 0,
                'Sharpness Performance': 0,
                'High Value Performance': 0,
            }
            total = 0

            # 重新读取文件
            f.seek(0)
            for line in f:
                total += 1
                data = json.loads(line)
                gt = parse_answer(data['ground_truth'])
                pred = parse_answer(data['prediction'])

                for key in scores.keys():
                    if gt.get(key) and pred.get(key) and gt[key].lower() == pred[key].lower():
                        scores[key] += 1

            print(f"\n=== 评估结果 (共 {total} 个样本) ===")
            for key, score in scores.items():
                accuracy = score / total * 100
                print(f"{key}: {accuracy:.2f}% ({score}/{total})")
        else:
            # 文本格式：只显示处理的样本数量
            total = 1  # 已经读取了一行
            for line in f:
                total += 1
            
            print(f"\n=== 评估完成 ===")
            print(f"共处理了 {total} 个样本")
            print("注意：这是文本格式的答案，无法进行自动准确率计算")


def evaluate_chat_model():
    args = parser.parse_args()
    random.seed(args.seed)
    
    os.makedirs(args.out_dir, exist_ok=True)
    assert args.batch_size == 1, 'Only batch size 1 is supported'

    local_rank = int(os.getenv('LOCAL_RANK', '0'))
    device = torch.device('cuda', local_rank)

    torch.distributed.init_process_group(
        backend='nccl',
        world_size=int(os.getenv('WORLD_SIZE', '1')),
        rank=int(os.getenv('RANK', '0')),
    )
    torch.cuda.set_device(device)

    model, tokenizer, new_token_ids = load_model_and_tokenizer(args)
    # model, tokenizer, new_token_ids = load_model_and_tokenizer(args, device)
    image_transform = build_transform()

    total_params = sum(p.numel() for p in model.parameters()) / 1e9
    print(f'[test] total_params: {total_params}B')

    dataset = RadarQADataset(args.test_file, args.prompt)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        sampler=InferenceSampler(len(dataset)),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        drop_last=False,
    )

    outputs = []
    for ids, images, conversations, ground_truths in tqdm(dataloader):
        for id, image, conversation, ground_truth in zip(ids, images, conversations, ground_truths):
            # 生成回复
            response = model.chat(
                tokenizer, 
                new_token_ids,
                image_transform,
                images=image,
                prompt=conversation,
                max_length=args.max_new_tokens,
            )

            # 后处理
            response = response.strip()
            
            # 保存结果
            outputs.append({
                'id': id,
                'ground_truth': ground_truth,
                'prediction': response
            })

    torch.distributed.barrier()
    
    world_size = torch.distributed.get_world_size()
    merged_outputs = [None for _ in range(world_size)]
    torch.distributed.all_gather_object(merged_outputs, outputs)
    
    merged_outputs = [item for sublist in merged_outputs for item in sublist]

    if torch.distributed.get_rank() == 0:
        output_file = os.path.join(args.out_dir, f'{os.path.basename(args.test_file).replace(".jsonl", "")}_results.jsonl')
        with open(output_file, 'w', encoding='utf-8') as f:
            for result in merged_outputs:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')
        
        print(f"结果已保存到: {output_file}")
        
        # 计算准确率
        calculate_accuracy(output_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test-file', type=str, default='dataset/RadarQA-related/dataset4Bagel/understanding/img_detail_test.jsonl')
    parser.add_argument('--out-dir', type=str, default='eval/vlm/eval/radarqa/results')
    parser.add_argument('--model-path', type=str, default='BAGEL_Test')
    parser.add_argument('--prompt', type=str, default='')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--max-new-tokens', type=int, default=512)
    parser.add_argument('--seed', type=int, default=0)
    
    evaluate_chat_model() 