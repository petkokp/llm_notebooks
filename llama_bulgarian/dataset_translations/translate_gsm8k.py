import os
from datasets import load_dataset, DatasetDict
from transformers import pipeline
import torch
import pandas as pd

HAS_HF_TOKEN = 'HF_TOKEN' not in os.environ

if HAS_HF_TOKEN:
    print("[WARNING] Environment variable 'HF_TOKEN' is not set! The dataset will be translated but not uploaded to HuggingFace.")
    user_input = input("Do you want to continue? (yes/no): ").strip().lower()

    if user_input != "yes":
        print("Terminating the program.")
        exit()

GSM8K_URL = "openai/gsm8k"
TRANSLATION_MODEL = "Helsinki-NLP/opus-mt-tc-big-en-bg"
DATASET_URL = "petkopetkov/gsm8k-bg"
BATCH_SIZE = 32

device = 0 if torch.cuda.is_available() else -1

print("Loading dataset...")
dataset = load_dataset(GSM8K_URL, "main")
translator = pipeline("translation", model=TRANSLATION_MODEL, device=device)

def translate_dataset(dataset_split, batch_size=32):
    def batch_translate(batch):
        questions = batch["question"]
        answers = batch["answer"]
        
        translated_questions = translator(questions, max_length=512, batch_size=batch_size)
        translated_answers = translator(answers, max_length=512, batch_size=batch_size)

        return {
            "question_translated": [t["translation_text"] for t in translated_questions],
            "answer_translated": [t["translation_text"] for t in translated_answers],
        }

    return dataset_split.map(batch_translate, batched=True, batch_size=batch_size)

translated_splits = {}
for split in ["train", "test"]:
    print(f"Processing split: {split}")
    ds_split = dataset[split]

    checkpoint_file = f"translations_{split}.csv"
    if os.path.exists(checkpoint_file):
        print(f"Resuming from checkpoint for '{split}' split.")
        translated_df = pd.read_csv(checkpoint_file)
        ds_split = ds_split.select(range(len(translated_df), len(ds_split)))

    if len(ds_split) > 0:
        translated_ds_split = translate_dataset(ds_split, batch_size=BATCH_SIZE)
        translated_splits[split] = translated_ds_split

        translated_df = translated_ds_split.to_pandas()
        translated_df.to_csv(checkpoint_file, index=False)
        print(f"Checkpoint saved for '{split}' split.")

if translated_splits and HAS_HF_TOKEN:
    combined_dataset = DatasetDict(translated_splits)
    combined_dataset.push_to_hub(DATASET_URL)
    print(f"Combined dataset with train and test splits uploaded to {DATASET_URL}!")
else:
    print("The translated dataset was not uploaded.")