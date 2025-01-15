import re
import ast

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
    
def convert_text_to_list(choice_str):
    try:
        choice_str = choice_str.replace("‘", "'").replace("’", "'")
        choice_str = re.sub(r'\s+', ' ', choice_str.strip())
        
        choice_str = re.sub(r"'\s*(?=[\"'])", "', ", choice_str)
        choice_str = re.sub(r"\"\s*(?=[\"'])", '", ', choice_str)

        if not (choice_str.startswith("[") and choice_str.endswith("]")):
            choice_str = f"[{choice_str}]"

        parsed_list = ast.literal_eval(choice_str)
        
        if isinstance(parsed_list, list):
            return parsed_list
        else:
            raise ValueError("Parsing did not result in a list.")
    except Exception as e:
        print(f"Failed to parse: {choice_str} - Error: {e}")
        return choice_str

def combine_choice_columns(example):
    label_choices = example['choices.label']
    label_choices = list(re.findall(r"'(.*?)'", label_choices))
    label_choices = list(map(map_label_choices, label_choices))
    
    text_choices = example['choices.text']
    new_choices = convert_text_to_list(text_choices)

    example['choices'] = {
        'text': new_choices,
        'label': label_choices,
    }
    return example

def postprocess_arc_choices(dataset, split):
    dataset[split] = dataset[split].map(combine_choice_columns)
    dataset[split] = dataset[split].remove_columns(['choices.text', 'choices.label'])
    print(f"Processed {split} dataset")
    return dataset
