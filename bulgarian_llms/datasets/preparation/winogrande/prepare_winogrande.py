from datasets import load_dataset
from typing import List, Dict, Any

def prepare_winogrande(dataset_path="petkopetkov/winogrande_xl-bg") -> List[Dict[str, Any]]:
        print("Processing Winogrande dataset...")
        
        dataset = load_dataset(dataset_path)
        
        processed_splits = {}
        
        for split in ['train', 'validation', 'test']:
            processed_data = []
            split_data = dataset[split]
            for idx in range(len(split_data)):
                item = {
                    'sentence': split_data[idx]['sentence'],
                    'option1': split_data[idx]['option1'],
                    'option2': split_data[idx]['option2'],
                    'answer': split_data[idx]['answer'],
                    'split': split
                }
                processed_data.append(item)
                
            processed_splits[split] = processed_data
                
        print("Processed Winogrande dataset!")
        return processed_splits