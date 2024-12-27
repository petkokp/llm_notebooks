import re

def map_label_choices(choice: str):
    if choice == "А": return "A"
    elif choice == "Б": return "B"
    elif choice == "В": return "C"
    elif choice == "Г": return "D"
    elif choice == "Д": return "E"
    elif choice == "1 бр.": return "1"
    elif choice == "2 пъти": return "2"
    elif choice == "3 пъти": return "3"
    elif choice == "4 пъти": return "4"

def combine_choice_columns(example):
    label_choices = example['choices.label']
    label_choices = list(re.findall(r"'(.*?)'", label_choices))
    label_choices = list(map(map_label_choices, label_choices))

    example['choices'] = {
        'text': example['choices.text'],
        'label': label_choices,
    }
    return example

def postprocess_arc_choices(dataset, split):
    dataset[split] = dataset[split].map(combine_choice_columns)
    dataset[split] = dataset[split].remove_columns(['choices.text', 'choices.label'])
    print(f"Processed {split} dataset")
    return dataset
