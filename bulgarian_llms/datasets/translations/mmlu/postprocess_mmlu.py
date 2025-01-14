from datasets import Value
import ast
import re

def map_subject(s):
    if s == "абстрактен_алгебра": return "abstract_algebra"
    elif s == "анатомия": return "anatomy"
    elif s == "астрономия": return "astronomy"
    elif s == "вирусология": return "virology"
    elif s == "гимназия_училище_компютър_наука": return "high_school_computer_science"
    elif s == "елементарна_математика": return "elementary_mathematics"
    if s == "иконометрия": 
        return "econometrics"
    elif s == "колеж_биология": 
        return "college_biology"
    elif s == "колеж_медицина": 
        return "college_medicine"
    elif s == "колеж_физика": 
        return "college_physics"
    elif s == "колеж_химия": 
        return "college_chemistry"
    elif s == "концептуална_физика": 
        return "conceptual_physics"
    elif s == "маркетинг": 
        return "marketing"
    elif s == "медицинска_генетика": 
        return "medical_genetics"
    elif s == "международно право": 
        return "international_law"
    elif s == "морални_сценарии": 
        return "moral_scenarios"
    elif s == "праистория": 
        return "prehistory"
    elif s == "професионално_счетоводство": 
        return "professional_accounting"
    elif s == "свят_религии": 
        return "world_religions"
    elif s == "сигурност_изследвания": 
        return "security_studies"
    elif s == "социология": 
        return "sociology"
    elif s == "управление": 
        return "management"
    elif s == "философия": 
        return "philosophy"
    elif s == "формал_логика": 
        return "formal_logic"
    elif s == "хранене": 
        return "nutrition"
    elif s == "юриспруденция": 
        return "jurisprudence" # TODO - check if translations are correct
    if s == "business_ethics (Етика на бизнеса)":
        return "business_ethics"
    elif s == "college_computer_наука":
        return "college_computer_science"
    elif s == "college_mathematics (Математика)":
        return "college_mathematics"
    elif s == "computer_security (сигурност)":
        return "computer_security"
    elif s == "high_school_european_история":
        return "high_school_european_history"
    elif s == "high_school_government_и_политиката":
        return "high_school_government_and_politics"
    elif s == "high_school_microикономика":
        return "high_school_microeconomics"
    elif s == "high_school_us_история":
        return "high_school_us_history"
    elif s == "high_school_world_история":
        return "high_school_world_history"
    elif s == "high_school_психология":
        return "high_school_psychology"
    elif s == "high_school_химия":
        return "high_school_chemistry"
    elif s == "human_aging (стареене)":
        return "human_aging"
    elif s == "human_сексуалност":
        return "human_sexuality"
    elif s == "machine_learning (Обучение)":
        return "machine_learning"
    elif s == "moral_disputes (спорове)":
        return "moral_disputes"
    elif s == "public_relations Български":
        return "public_relations"
    elif s == "us_чуждестранна_политика":
        return "us_foreign_policy"
    elif s == "Гимназия_Биология":
        return "high_school_biology"
    elif s == "Гимназия_Макроикономика":
        return "high_school_macroeconomics"
    elif s == "Гимназия_Математика":
        return "high_school_mathematics"
    elif s == "Гимназия_Статистика":
        return "high_school_statistics"
    elif s == "high_school_статистика":
        return "high school_statistics"
    elif s == "Гимназия_Физика":
        return "high_school_physics"
    elif s == "Гимназия_география":
        return "high_school_geography"
    elif s == "Глобал_факти":
        return "global_facts"
    elif s == "Електроинженерство":
        return "electrical_engineering"
    elif s == "Клинично_знание":
        return "clinical_knowledge"
    elif s == "Логически_фалации":
        return "logical_fallacies"
    elif s == "Професионалист_закон":
        return "professional_law"
    elif s == "Професионалист_медицина":
        return "professional_medicine"
    elif s == "Професионалист_лекарство":
        return "professional_medicine"
    elif s == "Професионалист_психология":
        return "professional_psychology"
    elif s == "Разни":
        return "miscellaneous"
    else:
        return "losho"
    
def map_example(example):
    s = example['subject']
    new_s = map_subject(s)
    if new_s == "losho": print('losh example: ', example['subject'])
    example['subject'] = new_s
    return example

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

def convert_choices_to_list(choice_str):
    try:
        # TODO - not good, include the correct conditions in the transformation below
        if "ГГТКТКАТК" in choice_str:
            return [ "5’ – GCATCCTCATG – 3’", "5’ – TGATCCCAG – 3’", "5’ – GGTCCTCATC – 3’", "5’ – GGATCCATG – 3’" ]
        elif "Място на свързване на рибозомите" in choice_str:
            return [ "Интрон", "3’ Поли А опашка", "Място на свързване на рибозомите", "5’ капачка" ]
        
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
    
def convert_choices(choices):
    choices = convert_choices_to_list(choices)
    choices = list(map(map_choices, choices))
    return choices

def postprocess_mmlu_choices(dataset, split):
    dataset[split] = dataset[split].map(lambda example: {
        'choices': convert_choices(example['choices'])
    })
    if split == "auxiliary_train": dataset[split] = dataset[split].map(lambda example: {
        'subject': ""
    })
    elif split == "dev" or split == "test":
        dataset[split] = dataset[split].map(map_example)
    
    dataset[split] = dataset[split].cast_column("answer", Value("string"))
    print(f"Processed {split} dataset")
    return dataset