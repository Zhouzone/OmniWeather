from evaluate import load
import os
import json
from tqdm import tqdm
import nltk

nltk_data_path = "your_path_here"
nltk.data.path.append(nltk_data_path)

cache_path = "/root/nltk_data"

meteor_metric = load("meteor")
JSON_PATH = "/root/bagel_baseline_img_detail_test_results.jsonl"
TEST_DATA = []
with open(JSON_PATH, 'r', encoding='utf-8') as file:
    for line in file:
        TEST_DATA.append(json.loads(line))
    
if __name__ == "__main__":
    sum_score = 0
    for data in tqdm(TEST_DATA[:410]):
        response_value = data["prediction"]
        label = data["ground_truth"]

        meteor_score = meteor_metric.compute(predictions=[response_value], references=[label])
        
        # Calculate BLEU score
        score = meteor_score['meteor']
        sum_score += score
    print(f"Average meteor: {sum_score / len(TEST_DATA)}")