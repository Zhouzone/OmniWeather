import os
import json
from tqdm import tqdm
from openai import OpenAI
import concurrent.futures
import argparse

def text_format(text):
    return {"type": "text", "text": text}

# 专门针对nowcasting的评分设置
DEFAULT_SETTINGS = {
    "system_prompt": "You are an expert meteorologist and AI evaluation specialist with deep knowledge of precipitation nowcasting, atmospheric dynamics, and causal reasoning in weather prediction.",
    "prompt": "We would like to request your expert evaluation of an AI assistant's nowcasting reasoning performance. "
    + "The assistant is tasked with short-range precipitation nowcasting based on VIL (Vertically Integrated Liquid) frames. "
    + "Please evaluate the assistant's response against the ground truth according to the following criteria:\n\n"
    + "1. **Causal Reasoning Quality (0-4 points)**: Does the response demonstrate proper causal relationships between temporal factors, perceptual factors, and structural outcomes? Is the logical chain from observations to predictions sound?\n\n"
    + "2. **Meteorological Accuracy (0-3 points)**: Does the response use appropriate meteorological terminology and concepts? Are the atmospheric processes described correctly?\n\n"
    + "3. **Consistency with Ground Truth (0-3 points)**: How well does the response align with the ground truth in terms of key observations and reasoning structure?\n\n"
    + "The assistant receives an overall score on a scale of 0 to 10, where a higher score indicates better performance. "
    + "Please first output a single line containing ONLY ONE INT NUMBER indicating the score of the assistant. "
    + "In the subsequent line, please provide a detailed evaluation addressing each criterion, highlighting specific strengths and weaknesses in the reasoning process.\n",
}

# 气象领域关键词列表，用于评估专业性
METEOROLOGICAL_KEYWORDS = [
    # 运动相关
    "motion direction", "motion speed", "advection", "translation", "displacement",
    "eastward", "westward", "northward", "southward", "northeast", "northwest", "southeast", "southwest",
    
    # 旋转和动力学
    "rotation center", "cyclonic", "anticyclonic", "swirl", "vortex", "rotational forcing",
    
    # 形态学
    "morphology", "blob-like", "banded", "scattered", "linear", "elongated", "compact",
    "rainband", "convective cluster", "convective core", "convective element",
    
    # 强度相关
    "intensity", "pixel level", "reflectivity", "VIL", "vertical integrated liquid",
    "moderate", "strong", "very strong", "extreme", "intensification", "weakening",
    "updraft", "downdraft", "convective vigor",
    
    # 结构演化
    "areal coverage", "organization", "coverage evolution", "organization evolution",
    "expanding", "contracting", "merging", "fragmentation", "coherent", "dispersed",
    
    # 时间相关
    "temporal", "evolution", "progression", "sequence", "frame", "time step",
    "short-range", "nowcasting", "forecast", "prediction",
    
    # 位置相关
    "initial position", "domain", "quadrant", "central", "peripheral",
    "centered", "positioned", "located",
    
    # 物理过程
    "convective", "precipitation", "atmospheric", "dynamical", "structural",
    "forcing", "environmental", "advective", "kinematic"
]

DEFAULT_SAVE = {
    "save folder": "your_path_here",
    "query": (
        "Please provide a comprehensive nowcasting analysis that includes:\n"
        + "1. Temporal causal factors (motion direction, speed, rotation)\n"
        + "2. Perceptual factors (morphology, intensity, position)\n"
        + "3. Direct outcomes (intensity evolution)\n"
        + "4. Structural outcomes (areal coverage and organization evolution)\n"
        + "Use proper meteorological terminology and demonstrate clear causal reasoning."
    )
}

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", "http://35.220.164.252:3888/v1")

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is required to run this evaluation script.")

client = OpenAI(base_url=OPENAI_BASE_URL, api_key=OPENAI_API_KEY)

def count_meteorological_keywords(text):
    """统计文本中气象关键词的使用情况"""
    text_lower = text.lower()
    keyword_count = 0
    used_keywords = []
    
    for keyword in METEOROLOGICAL_KEYWORDS:
        if keyword.lower() in text_lower:
            keyword_count += 1
            used_keywords.append(keyword)
    
    return keyword_count, used_keywords

def processor(label, response_value, model_name, index):
    # 统计关键词使用情况
    gt_keywords, gt_used = count_meteorological_keywords(label)
    pred_keywords, pred_used = count_meteorological_keywords(response_value)
    
    content = [text_format(
        f"[Question]\n{DEFAULT_SAVE['query']}\n\n"
        + f"[Ground Truth]\n{label}\n\n[End of Ground Truth]\n\n"
        + f"[Assistant Response]\n{response_value}\n\n[End of Assistant Response]\n\n"
        + f"[Evaluation Criteria]\n"
        + f"1. Causal Reasoning: Assess the logical flow from temporal → perceptual → direct → structural outcomes\n"
        + f"2. Meteorological Accuracy: Evaluate use of proper weather terminology and atmospheric concepts\n"
        + f"3. Consistency: Compare key observations and reasoning structure with ground truth\n"
        + f"4. Keyword Usage: Ground truth uses {gt_keywords} meteorological terms, prediction uses {pred_keywords}\n\n"
        + f"[System]\n{DEFAULT_SETTINGS['prompt']}\n\n"
    )]

    try:
        response = client.chat.completions.create(
            model="gpt-4o-2024-11-20",
            messages = [
                {
                    "role": "system",
                    "content": DEFAULT_SETTINGS["system_prompt"],
                },
                {
                    "role": "user",
                    "content": content
                }
            ],
            temperature=0
        )
    except Exception as e:
        print("Request failed at index", index)
        raise e
    
    curr_response = {}
    curr_response.update({"response": response.choices[0].message.content})
    curr_response.update({"ground_truth": label})
    curr_response.update({"prediction": response_value})
    curr_response.update({"gt_keyword_count": gt_keywords})
    curr_response.update({"pred_keyword_count": pred_keywords})
    curr_response.update({"gt_keywords_used": gt_used})
    curr_response.update({"pred_keywords_used": pred_used})
    
    model_save_folder = os.path.join(DEFAULT_SAVE["save folder"], model_name)
    
    with open(os.path.join(model_save_folder, f"{index}.json"), "w") as f:
        json.dump(curr_response, f, indent=4)
    print(f"Successfully processed {index} - GT keywords: {gt_keywords}, Pred keywords: {pred_keywords}")

def generate_summary_report(model_name):
    """生成评分汇总报告"""
    model_save_folder = os.path.join(DEFAULT_SAVE["save folder"], model_name)
    
    if not os.path.exists(model_save_folder):
        print(f"Model folder not found: {model_save_folder}")
        return
    
    scores = []
    keyword_stats = {"gt_total": 0, "pred_total": 0, "gt_avg": 0, "pred_avg": 0}
    
    for file in os.listdir(model_save_folder):
        if file.endswith('.json'):
            with open(os.path.join(model_save_folder, file), 'r') as f:
                data = json.load(f)
                
                # 提取分数
                response_text = data.get('response', '')
                try:
                    score_line = response_text.split('\n')[0]
                    score = int(score_line.strip())
                    scores.append(score)
                except:
                    print(f"Could not extract score from {file}")
                
                # 统计关键词
                keyword_stats["gt_total"] += data.get('gt_keyword_count', 0)
                keyword_stats["pred_total"] += data.get('pred_keyword_count', 0)
    
    if scores:
        avg_score = sum(scores) / len(scores)
        keyword_stats["gt_avg"] = keyword_stats["gt_total"] / len(scores)
        keyword_stats["pred_avg"] = keyword_stats["pred_total"] / len(scores)
        
        summary = {
            "model_name": model_name,
            "total_samples": len(scores),
            "average_score": round(avg_score, 2),
            "score_distribution": {
                "0-2": len([s for s in scores if 0 <= s <= 2]),
                "3-5": len([s for s in scores if 3 <= s <= 5]),
                "6-8": len([s for s in scores if 6 <= s <= 8]),
                "9-10": len([s for s in scores if 9 <= s <= 10])
            },
            "keyword_usage": keyword_stats
        }
        
        with open(os.path.join(model_save_folder, "summary_report.json"), "w") as f:
            json.dump(summary, f, indent=4)
        
        print(f"\n=== Summary Report for {model_name} ===")
        print(f"Total samples: {len(scores)}")
        print(f"Average score: {avg_score:.2f}")
        print(f"Score distribution: {summary['score_distribution']}")
        print(f"GT avg keywords: {keyword_stats['gt_avg']:.1f}")
        print(f"Pred avg keywords: {keyword_stats['pred_avg']:.1f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl_path", type=str, required=True)
    parser.add_argument("--generate_summary", action="store_true", help="Generate summary report after processing")
    args = parser.parse_args()
    
    jsonl_path = args.jsonl_path
    model_name = jsonl_path.split("/")[-1].split(".")[0]
    
    indices = []
    labels = []
    responses = []
    
    test_data = []
    
    with open(jsonl_path, "r") as f:
        for line in f:
            if line.strip():  # 跳过空行
                test_data.append(json.loads(line))
    
    for index, data in enumerate(test_data):
        label = data["ground_truth"]
        response = data["prediction"]
        indices.append(index)
        labels.append(label)
        responses.append(response)
    
    os.makedirs(os.path.join(DEFAULT_SAVE["save folder"], model_name), exist_ok=True)
    Already_generated_files = [file.split(".")[0] for file in os.listdir(os.path.join(DEFAULT_SAVE["save folder"], model_name)) if file.endswith('.json')]
    final_indices = [i for i in indices if str(i) not in Already_generated_files]
    final_labels = [labels[indices.index(i)] for i in final_indices]
    final_responses = [responses[indices.index(i)] for i in final_indices]
    
    print(f"Processing {len(final_indices)} samples for model: {model_name}")
    
    with concurrent.futures.ThreadPoolExecutor() as executor:
        list(tqdm(executor.map(processor, final_labels, final_responses, [model_name]*len(final_indices), final_indices), total=len(final_indices), desc="Processing files"))
    
    if args.generate_summary:
        generate_summary_report(model_name)
