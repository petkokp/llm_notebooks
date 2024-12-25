from datasets import Dataset, DatasetDict

def process_single_rationale(x):
    if isinstance(x, str):
        replacements = {
            "а .": "a .",
            "отговор: а": "отговор: a .",
            "отговор : а": "отговор: а .",
            "отговор а": "отговор: e .",
            "отговор  а": "отговор: e .",
            "б .": "b .",
            "отговор: б": "отговор: b .",
            "отговор : б": "отговор: b .",
            "отговор б": "отговор: e .",
            "отговор  б": "отговор: e .",
            "в .": "c .",
            "отговор: в": "отговор: c .",
            "отговор : в": "отговор: c .",
            "отговор в": "отговор: e .",
            "отговор  в": "отговор: e .",
            "г .": "d .",
            "отговор: г": "отговор: d .",
            "отговор : г": "отговор: d .",
            "отговор г": "отговор: e .",
            "отговор  г": "отговор: e .",
            "д .": "e .",
            "отговор: д": "отговор: e .",
            "отговор : д": "отговор: e .",
            "отговор д": "отговор: e .",
            "отговор  д": "отговор: e ."
        }
        
        for old, new in replacements.items():
            x = x.replace(old, new)
        
    return x

def process_rationale(x):
    if isinstance(x, list):
        return [process_single_rationale(item) for item in x]
    else:
        return process_single_rationale(x)
    
def process_rationales(dataset, split):
    dataset[split] = dataset[split].map(lambda example: {
        'Rationale': process_rationale(example['Rationale'])
    })
    
    print(f"Processed {split} dataset")

