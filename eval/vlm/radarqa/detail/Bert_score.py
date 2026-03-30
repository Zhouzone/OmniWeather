from bert_score import BERTScorer
import os
import json
from tqdm import tqdm


# model_path = "your_path_here"

# model_dir = "your_path_here"
# os.makedirs(model_dir, exist_ok=True)

# 使用cache_dir参数指定模型下载位置
scorer = BERTScorer(
    model_type="bert-base-uncased",
)

# 创建BERTScorer
# scorer = BERTScorer(model_path=model_path, model_type="bert-base-uncased")

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

                    P, R, F1 = scorer.score([label], [response_value])
                    
                    sum_score += P.mean()
        print(f"Average Bert score {folder.split('/')[-1]}: {sum_score / len(os.listdir(folder))}")
    

                