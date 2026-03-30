from bert_score import BERTScorer
import os
import json
from tqdm import tqdm


model_path = "your_path_here"

# 创建BERTScorer
scorer = BERTScorer(model_path=model_path, model_type="bert-base-uncased")

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
        P, R, F1 = scorer.score([label], [response_value])
                
        sum_score += P.mean()
    print(f"Average Bert score: {sum_score / len(TEST_DATA)}")
    

                