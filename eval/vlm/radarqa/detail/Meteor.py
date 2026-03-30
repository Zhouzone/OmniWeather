from evaluate import load
import os
import json
from tqdm import tqdm
import nltk

nltk_data_path = "your_path_here"
nltk.data.path.append(nltk_data_path)

cache_path = "/root/nltk_data"

meteor_metric = load("meteor")
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

                    meteor_score = meteor_metric.compute(predictions=[response_value], references=[label])
                    
                    # Calculate BLEU score
                    score = meteor_score['meteor']
                    sum_score += score
        print(f"Average meteor {folder.split('/')[-1]}: {sum_score / len(os.listdir(folder))}")