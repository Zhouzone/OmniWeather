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

FOLDERS =[
    "your_path_here"
] 


if __name__ == "__main__":
    
    for folder in FOLDERS:
        sum_score = 0
        for filename in tqdm(os.listdir(folder)):
            if filename.endswith(".json"):
                file_path = os.path.join(folder, filename)
                with open(file_path, 'r', encoding='utf-8') as file:
                    data = json.load(file)
                    response_value = data["response"]
                    label = data["label"]

                    # Tokenize the response and label
                    response_tokens = word_tokenize(response_value)
                    label_tokens = word_tokenize(label)
                    
                    # Calculate BLEU score
                    score = sentence_bleu([label_tokens], response_tokens)
                    
                    sum_score += score
        print(f"Average BLEU score {folder.split('/')[-1]}: {sum_score / len(os.listdir(folder))}")
                