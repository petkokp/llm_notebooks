import os
from datasets import load_dataset, DatasetDict
from transformers import pipeline
import torch
import pandas as pd

def translate_dataset(ds, translation_model, output_dataset_url, batch_size=32, hf_token_env='HF_TOKEN', checkpoint_interval=100):
    has_hf_token = hf_token_env not in os.environ
    if has_hf_token:
        print("[WARNING] Environment variable 'HF_TOKEN' is not set! The dataset will be translated but not uploaded to HuggingFace.")
        user_input = input("Do you want to continue? (yes/no): ").strip().lower()
        if user_input != "yes":
            print("Terminating the program.")
            return
    
    device = 0 if torch.cuda.is_available() else -1
    
    print("Loading dataset...")
    dataset = load_dataset(ds['url'], ds['config'])
    
    translator = pipeline("translation", model=translation_model, device=device)
    
    def translate_dataset(dataset_split, batch_size=32):
        def batch_translate(batch):
            translated_batch = {}
            
            for column_name, column_data in batch.items():
                if isinstance(column_data[0], str):
                    translated_texts = translator(column_data, max_length=512, batch_size=batch_size)
                    translated_batch[f"{column_name}_translated"] = [t["translation_text"] for t in translated_texts]
                else:
                    translated_batch[column_name] = column_data
            
            return translated_batch

        return dataset_split.map(batch_translate, batched=True, batch_size=batch_size)

    translated_splits = {}
    for split in dataset.keys():
        print(f"Processing split: {split}")
        ds_split = dataset[split]
        
        checkpoint_file = f"./{ds['url'].split('/')[-1]}_translations_{split}.csv"
        if os.path.exists(checkpoint_file):
            print(f"Resuming from checkpoint for '{split}' split.")
            translated_df = pd.read_csv(checkpoint_file)
            ds_split = ds_split.select(range(len(translated_df), len(ds_split)))
        
        if len(ds_split) > 0:
            translated_ds_split = []
            for start_idx in range(0, len(ds_split), checkpoint_interval):
                end_idx = min(start_idx + checkpoint_interval, len(ds_split))
                batch = ds_split.select(range(start_idx, end_idx))
                
                translated_batch = translate_dataset(batch, batch_size=batch_size)
                
                translated_ds_split.append(translated_batch)
                
                translated_df_batch = translated_batch.to_pandas()
                
                translated_df_batch.to_csv(checkpoint_file, mode='a', header=not os.path.exists(checkpoint_file), index=False)
                print(f"Checkpoint saved for '{split}' split after processing {end_idx} examples.")

            translated_splits[split] = pd.concat(translated_ds_split, ignore_index=True)
    
    if translated_splits and not has_hf_token:
        combined_dataset = DatasetDict(translated_splits)
        combined_dataset.push_to_hub(output_dataset_url)
        print(f"Combined dataset with splits uploaded to {output_dataset_url}!")
    else:
        print("The translated dataset was not uploaded.")
