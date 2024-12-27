import os
from datasets import load_dataset, DatasetDict, Dataset
from transformers import pipeline, AutoTokenizer
import torch
import pandas as pd
#from optimum.nvidia.pipelines import pipeline
#from optimum.pipelines import pipeline

def translate_dataset(ds, translation_model, output_dataset_url, batch_size=32, hf_token_env='HF_TOKEN', checkpoint_interval=1000, exclude_columns=None, preprocess_dataset=None):
    if exclude_columns is None:
        exclude_columns = []

    has_hf_token = hf_token_env not in os.environ
    if has_hf_token:
        print("[WARNING] Environment variable 'HF_TOKEN' is not set! The dataset will be translated but not uploaded to HuggingFace.")
        user_input = input("Do you want to continue? (yes/no): ").strip().lower()
        if user_input != "yes":
            print("Terminating the program.")
            return
    
    device = 0 if torch.cuda.is_available() else -1
    
    print("Loading dataset...")
    dataset = load_dataset(ds['url'], ds['config'] if "config" in ds else None)
    
    tokenizer = AutoTokenizer.from_pretrained(translation_model, use_fast=True)
    translator = pipeline("translation", model=translation_model, tokenizer=tokenizer, truncation=True, device=device, torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16)
    
    def translate_dataset_split(dataset_split, batch_size=32):
        def batch_translate(batch):
            translated_batch = {}
            
            for column_name, column_data in batch.items():
                if column_name in exclude_columns:
                    translated_batch[column_name] = column_data
                elif isinstance(column_data[0], str) and not column_data[0].isdigit():
                    translated_texts = translator(column_data, truncation=True, max_length=512, batch_size=batch_size)
                    translated_batch[column_name] = [t["translation_text"] for t in translated_texts]
                elif isinstance(column_data[0], list) and all(isinstance(item, str) for item in column_data[0]):
                    translated_column = []
                    for sublist in column_data:
                        translated_texts = translator(sublist, truncation=True, max_length=512, batch_size=batch_size)
                        translated_column.append([t["translation_text"] for t in translated_texts])
                    translated_batch[column_name] = translated_column
                else:
                    translated_batch[column_name] = column_data
            
            return translated_batch

        return dataset_split.map(batch_translate, batched=True, batch_size=batch_size)
    
    translated_splits = {}
    
    for split in dataset.keys():
        if split != "auxiliary_train": continue
        
        if preprocess_dataset is not None:
            dataset = preprocess_dataset(dataset, split)
        
        processed_examples = 0
        total_examples = len(dataset[split])
    
        print(f"Processing split: {split}")
        ds_split = dataset[split]
        
        checkpoint_file = f"./{ds['url'].split('/')[-1]}_translations_{split}.csv"
        processed_dataframes = []
        
        if os.path.exists(checkpoint_file):
            print(f"Found checkpoint for '{split}' split. Loading processed data.")
            processed_dataframes.append(pd.read_csv(checkpoint_file))
            processed_rows = sum(len(df) for df in processed_dataframes)
            
            if processed_rows >= len(ds_split):
                print(f"The entire split '{split}' has already been translated. Skipping.")
                continue
            
            ds_split = ds_split.select(range(processed_rows, len(ds_split)))
        else:
            processed_rows = 0
        
        if len(ds_split) > 0:
            for start_idx in range(10, len(ds_split), checkpoint_interval):
                end_idx = min(start_idx + checkpoint_interval, len(ds_split))
                batch = ds_split.select(range(start_idx, end_idx))
                
                translated_batch = translate_dataset_split(batch, batch_size=batch_size)
                translated_df_batch = translated_batch.to_pandas()
                
                translated_df_batch.to_csv(checkpoint_file, mode='a', header=not os.path.exists(checkpoint_file), index=False)
                processed_dataframes.append(translated_df_batch)
                
                processed_examples += len(batch)
                progress = (processed_examples / total_examples) * 100
                print(f"Progress: {processed_examples}/{total_examples} ({progress:.2f}%) - Checkpoint saved for '{split}' split.")
        
        if processed_dataframes:
            translated_splits[split] = Dataset.from_pandas(pd.concat(processed_dataframes, ignore_index=True))

    if translated_splits:
        combined_dataset = DatasetDict(translated_splits)
        if not has_hf_token:
            combined_dataset.push_to_hub(output_dataset_url)
            print(f"Combined dataset with splits uploaded to {output_dataset_url}!")
        else:
            print("The combined translated dataset was created but not uploaded.")
    else:
        print("No data was translated or saved.")
