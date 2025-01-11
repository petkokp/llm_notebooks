import fire
import os
from datasets import load_dataset
from tqdm import tqdm

DATASET = "petkopetkov/chitanka"

def main(split="validation", lang="bg", docs_to_sample=30_000, save_path="data"):
    dataset = load_dataset(DATASET, split=split, streaming=True)
    os.makedirs(save_path, exist_ok=True)
    with open(os.path.join(save_path, f"{lang}.txt"), "w") as f:
        count = 0
        for d in tqdm(dataset, desc="Processing dataset"):
            if count >= docs_to_sample:
                break
            f.write(d["text"] + "\n")
            count += 1

if __name__ == "__main__":
    fire.Fire(main)