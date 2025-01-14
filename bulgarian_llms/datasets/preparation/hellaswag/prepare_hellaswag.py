from datasets import load_dataset
from typing import List, Dict, Any

def prepare_hellaswag(dataset_path="petkopetkov/hellaswag-bg") -> List[Dict[str, Any]]:
        print("Processing Hellaswag dataset...")
        
        dataset = load_dataset(dataset_path)
        
        processed_splits = {}
        
        for split in ['train', 'validation']: # 'test' has no labels
            processed_data = []
            split_data = dataset[split]
            for idx in range(len(split_data)):
                item = {
                    'context': split_data[idx]['ctx'],
                    'activity_label': split_data[idx]['activity_label'],
                    'choices': split_data[idx]['endings'],
                    'answer': split_data[idx]['label'],
                    'split': split
                }
                processed_data.append(item)
                
            processed_splits[split] = processed_data
                
        print("Processed Hellaswag dataset!")
        return processed_splits