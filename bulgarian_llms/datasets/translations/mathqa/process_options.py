from datasets import Dataset, DatasetDict

def process_option(x):
    if isinstance(x, list):
        return [process_single_option(item) for item in x]
    else:
        return process_single_option(x)

def process_single_option(x):
    if isinstance(x, str):
        replacements = {
            "а )": "a )",
            "б )": "b )",
            "в )": "c )",
            "г )": "d )",
            "д )": "e )",
            "а)": "a)",
            "б)": "b)",
            "в)": "c)",
            "г)": "d)",
            "д)": "e)"
        }
        
        for old, new in replacements.items():
            x = x.replace(old, new)
        
    return x

def process_options(dataset, split):
    dataset[split] = dataset[split].map(lambda example: {
        'options': process_option(example['options'])
    })
    
    print(f"Processed {split} dataset")

    return dataset