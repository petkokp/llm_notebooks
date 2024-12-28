from datasets import Value
import ast

def map_choices(choice: str):
    if choice == "0 пъти": return "0"
    elif choice == "1 бр.": return "1"
    elif choice == "2 пъти": return "2"
    elif choice == "3 пъти": return "3"
    elif choice == "4 пъти": return "4"
    elif choice == "5 пъти": return "5"
    elif choice == "6 пъти": return "6"
    elif choice == "7 пъти": return "7"
    elif choice == "8 пъти": return "8"
    elif choice == "9 пъти": return "9"
    return choice
    
def convert_choices_to_list(choices):
    choices = choices.replace("' '", "', '")
    choices = ast.literal_eval(choices)
    return choices
    
def convert_choices(choices):
    choices = convert_choices_to_list(choices)
    choices = list(map(map_choices, choices))
    return choices

def postprocess_mmlu_choices(dataset, split):
    dataset[split] = dataset[split].map(lambda example: {
        'choices': convert_choices(example['choices'])
    })
    dataset[split] = dataset[split].cast_column("answer", Value("string"))
    print(f"Processed {split} dataset")
    return dataset
