from datasets import load_dataset
from typing import List, Dict, Any
import re

def parse_options(options_str: str) -> Dict[str, str]:
        options_list = re.split(r',(?=[a-e]\s*\))', options_str)
        
        options_dict = {}
        for opt in options_list:
            match = re.match(r'\s*([a-e])\s*\)\s*(.*)', opt.strip())
            if match:
                letter, content = match.groups()
                options_dict[letter] = content.strip()
                
        return options_dict


def prepare_mathqa(dataset_path="petkopetkov/math_qa-bg") -> List[Dict[str, Any]]:
        print("Processing MathQA dataset...")
        
        dataset = load_dataset(dataset_path)
        
        processed_splits = {}
        
        for split in ['train', 'validation', 'test']:
            processed_data = []
            split_data = dataset[split]
            for idx in range(len(split_data)):
                options = parse_options(split_data[idx]['options'])
                
                item = {
                    'problem': split_data[idx]['Problem'],
                    'rationale': split_data[idx]['Rationale'],
                    'options': options,
                    'correct': split_data[idx]['correct'],
                    'split': split
                }
                processed_data.append(item)
                
            processed_splits[split] = processed_data
                
        print("Processed MathQA dataset!")
        return processed_splits