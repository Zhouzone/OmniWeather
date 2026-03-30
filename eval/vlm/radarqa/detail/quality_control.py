import os
import json
from tqdm import tqdm

'''
gpt4o在面对此任务时会经常输出不合格的内容，需要删掉重新生成
'''

if __name__ == "__main__":
    folder_path = "your_path_here"

    deleted = 0
    for filename in tqdm(os.listdir(folder_path)):
        file_path = os.path.join(folder_path, filename)
        if filename.endswith(".json"):
                with open(file_path, 'r', encoding='utf-8') as file:
                    data = json.load(file)
                    response_value = data["score"]
                    try:
                        numeric_value = float(response_value)
                    except:
                        os.remove(file_path)
                        print(f"已删除文件: {filename}")
                        deleted += 1
                    
    print(deleted)