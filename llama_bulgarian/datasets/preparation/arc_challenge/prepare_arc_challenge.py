from datasets import load_dataset
from typing import List, Dict, Any

def prepare_arc_challenge(dataset_path="petkopetkov/arc-challenge-bg") -> List[Dict[str, Any]]:
    print("Processing ARC-Challenge dataset...")
    
    dataset = load_dataset(dataset_path)
    
    processed_splits = {}
    
    for split in ['train', 'validation', 'test']:
        processed_data = []
        split_data = dataset[split]
        for idx in range(len(split_data)):
            item = {
                'question': split_data[idx]['question'],
                'choices': split_data[idx]['choices'],
                'answer': split_data[idx]['answerKey'],
                'split': split
            }
            processed_data.append(item)
        
        processed_splits[split] = processed_data
    
    print("Processed ARC-Challenge dataset!")
    return processed_splits
