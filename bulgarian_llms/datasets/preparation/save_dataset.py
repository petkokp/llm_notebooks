from typing import List, Dict, Any
import json

def save_dataset(data: List[Dict[str, Any]], filename: str):
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)