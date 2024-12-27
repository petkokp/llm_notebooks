from datasets import Value

def convert_endings_to_list(endings):
    endings = endings.replace("\n", "")
    endings = endings[1:-1]
    endings = endings.split(".' '")
    endings = [e.strip(" '") + "." for e in endings]
    return endings

def postprocess_hellaswag_endings(dataset, split):
    dataset[split] = dataset[split].map(lambda example: {
        'endings': convert_endings_to_list(example['endings'])
    })
    dataset[split] = dataset[split].cast_column("label", Value("string"))
    print(f"Processed {split} dataset")
    return dataset
