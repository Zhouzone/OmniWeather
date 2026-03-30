import os
import json
def parse_score(review):
    try:
        score = review.split("\n")[0].strip()
        try:
            return float(score)
        except:
            print("error", review)
            return -1
    except Exception as e:
        print(e)
        print("error", review)
        return -1

FOLDER_PATH = "your_path_here"

if __name__ == "__main__":
    scores = []
    for filename in os.listdir(FOLDER_PATH):
        json_file_path = os.path.join(FOLDER_PATH, filename)
        with open(json_file_path, "r") as f:
            json_data = json.load(f)
            score = parse_score(json_data["response"])
            scores.append(score)
    
    print(f"Average score: {sum(scores) / len(scores)}")