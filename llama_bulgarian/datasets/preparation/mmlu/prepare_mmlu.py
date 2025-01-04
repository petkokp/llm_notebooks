from datasets import load_dataset
from typing import List, Dict, Any

def prepare_mmlu(subject: str = None, dataset_path="petkopetkov/mmlu-bg") -> List[Dict[str, Any]]:
        print("Processing MMLU dataset...")

        dataset = load_dataset(dataset_path)
        
        processed_splits = {}
        
        for split in ['auxiliary_train', 'dev', 'test']:
            processed_data = []
            split_data = dataset[split]
            for idx in range(len(split_data)):
                item = {
                    'question': split_data[idx]['question'],
                    'choices': [
                        split_data[idx]['choices'][0],
                        split_data[idx]['choices'][1],
                        split_data[idx]['choices'][2],
                        split_data[idx]['choices'][3]
                    ],
                    'answer': split_data[idx]['answer'],
                    'subject': split_data[idx]['subject'],
                    'split': split
                }
                if subject is None or item['subject'] == subject:
                    processed_data.append(item)
                    
            processed_splits[split] = processed_data
                    
        print("Processed MMLU dataset!")
        return processed_splits