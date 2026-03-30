import os
import json
from tqdm import tqdm
from rouge_score import rouge_scorer

JSON_PATH = "/root/bagel_baseline_img_detail_test_results.jsonl"
TEST_DATA = []
with open(JSON_PATH, 'r', encoding='utf-8') as file:
    for line in file:
        TEST_DATA.append(json.loads(line))
        
scorer = rouge_scorer.RougeScorer(['rougeLsum'])

if __name__ == "__main__":
    sum_score = 0
    for data in tqdm(TEST_DATA):
        response_value = data["prediction"]
        label = data["ground_truth"]

        scores = scorer.score(target=label, prediction=response_value)
        
        # Calculate BLEU score
        score = scores['rougeLsum'].fmeasure
        sum_score += score
    print(f"Average rougeLsum: {sum_score / len(TEST_DATA)}")