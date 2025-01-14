from datasets import load_dataset
from typing import List, Dict, Any

def prepare_gsm8k(dataset_path="petkopetkov/gsm8k-bg") -> List[Dict[str, Any]]:
        print("Processing GSM8K dataset...")
        
        dataset = load_dataset(dataset_path)
        
        processed_splits = {}
        
        for split in ['train', 'test']:
            processed_data = []
            split_data = dataset[split]
            for idx in range(len(split_data)):
                item = {
                    'question': split_data[idx]['question'],
                    'answer': split_data[idx]['answer'],
                    'split': split
                }
                processed_data.append(item)
                
            processed_splits[split] = processed_data
                
        print("Processed GSM8K dataset!")
        return processed_splits