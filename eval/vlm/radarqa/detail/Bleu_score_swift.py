from nltk.translate.bleu_score import sentence_bleu
from nltk.tokenize import word_tokenize
import nltk
import os
import json
from tqdm import tqdm

nltk_data_path = "your_path_here"
# nltk.download('punkt', download_dir=nltk_data_path)
# nltk.download('punkt_tab', download_dir=nltk_data_path)

nltk.data.path.append(nltk_data_path)

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

        # Tokenize the response and label
        response_tokens = word_tokenize(response_value)
        label_tokens = word_tokenize(label)
        
        # Calculate BLEU score
        score = sentence_bleu([label_tokens], response_tokens)

        # Calculate BLEU score
        score = sentence_bleu([label_tokens], response_tokens)
        sum_score += score
    print(f"Average BLEU score: {sum_score / len(TEST_DATA)}")
                