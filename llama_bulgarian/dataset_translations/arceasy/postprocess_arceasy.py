def combine_choice_columns(example):
        example['choices'] = {
            'text': example['choices.text'],
            'label': example['choices.label'],
        }
        return example

def postprocess_arceasy_choices(dataset, split):
    dataset[split] = dataset[split].map(combine_choice_columns)
    dataset[split] = dataset[split].remove_columns(['choices.text', 'choices.label'])
    print(f"Processed {split} dataset")
    return dataset
