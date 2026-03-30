from rouge_score import rouge_scorer
import os
import json
from tqdm import tqdm

FOLDERS =[
    "your_path_here"
] 

scorer = rouge_scorer.RougeScorer(['rougeLsum'])
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

                    scores = scorer.score(target=label, prediction=response_value)
                    
                    # Calculate BLEU score
                    score = scores['rougeLsum'].fmeasure
                    sum_score += score
                    
        print(f"Average rougeLsum {folder.split('/')[-1]}: {sum_score / len(os.listdir(folder))}")
                