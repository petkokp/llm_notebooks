import os
import pandas as pd
from datasets import load_dataset, Dataset
from transformers import pipeline
import torch

if 'HF_TOKEN' not in os.environ:
        print("[WARNING] Environment variable 'HF_TOKEN' is not set!")
        user_input = input("Do you want to continue? (yes/no): ").strip().lower()

        if user_input != "yes":
            print("Terminating the program.")
            exit()

GSM8K_URL = "openai/gsm8k"

dataset = load_dataset(GSM8K_URL, "main")
dataframe = dataset["train"].to_pandas()

device = 0 if torch.cuda.is_available() else -1

TRANSLATION_MODEL = "Helsinki-NLP/opus-mt-tc-big-en-bg"

translator = pipeline("translation", model=TRANSLATION_MODEL, device=device)

checkpoint_file = "translations.csv"

if os.path.exists(checkpoint_file):
    translated_df = pd.read_csv(checkpoint_file)
    print(f"Resuming from checkpoint: {len(translated_df)} rows already translated.")
else:
    translated_df = pd.DataFrame(columns=["question", "answer"])

for index, row in dataframe.iterrows():
    if index < len(translated_df):
        continue
    
    try:
        translated_question = translator(row["question"], max_length=512)[0]["translation_text"]
        translated_answer = translator(row["answer"], max_length=512)[0]["translation_text"]
        translated_df = pd.concat(
            [translated_df, pd.DataFrame([{"question": translated_question, "answer": translated_answer}])],
            ignore_index=True
        )

        if index % 10 == 0:
            translated_df.to_csv(checkpoint_file, index=False)
            print(f"Checkpoint saved at row {index}.")
    except Exception as e:
        print(f"Error at row {index}: {e}. Skipping...")
        
DATASET_URL = "petkopetkov/gsm8k-bg"

translated_df.to_csv(checkpoint_file, index=False)
translated_dataset = Dataset.from_pandas(translated_df)
translated_dataset.push_to_hub(DATASET_URL)
print("Translation completed and uploaded!")
