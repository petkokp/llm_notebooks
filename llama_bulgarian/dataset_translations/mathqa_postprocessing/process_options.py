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

splits = ["train", "test", "validation"]

translated_splits = {
    "train": Dataset.from_csv("math_qa_translations_train.csv"),
    "test": Dataset.from_csv("math_qa_translations_test.csv"),
    "validation": Dataset.from_csv("math_qa_translations_validation.csv")
}

combined_dataset = DatasetDict(translated_splits)

def process_options(dataset, split):
    dataset[split] = dataset[split].map(lambda example: {
        'options': process_option(example['options'])
    })
    
    print(f"Processed {split} dataset")

print("Processing train")
process_options(combined_dataset, 'train')
print("Processed train")
print("Processing test")
process_options(combined_dataset, 'test')
print("Processed test")
print("Processing validation")
process_options(combined_dataset, 'validation')
print("Processed validation")