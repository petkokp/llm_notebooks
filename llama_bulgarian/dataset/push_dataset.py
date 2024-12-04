from datasets import Dataset
import json
import pandas as pd

QUESION_ANSWER_PAIRS_FILE = "dataset.json"
HUGGINGFACE_DATASET = "petkopetkov/QABGB"

with open(QUESION_ANSWER_PAIRS_FILE) as f:
    data = json.load(f)

df = pd.read_json('dataset.json')

dataset = Dataset.from_pandas(df)

dataset.push_to_hub(HUGGINGFACE_DATASET)