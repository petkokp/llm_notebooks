from typing import List, Dict, Any

def format_datasets(dataset_name: str, data: Dict[str, List[Dict[str, Any]]]) -> Dict[str, List[Dict[str, str]]]:
        formatted_splits = {}
        
        for split, split_data in data.items():
            formatted_data = []
            
            for item in split_data:
                if dataset_name == "mmlu":
                    formatted = {
                        "instruction": f"Въпрос: {item['question']}\nОтговори:\n1) {item['choices'][0]}\n2) {item['choices'][1]}\n3) {item['choices'][2]}\n4) {item['choices'][3]}",
                        "input": "",
                        "output": f"Правилният отговор е {item['answer']}"
                    }
                elif dataset_name == "winogrande":
                    formatted = {
                        "instruction": f"Довърши изречението:\n{item['sentence']}\nОтговор:\n1) {item['option1']}\n2) {item['option2']}",
                        "input": "",
                        "output": f"Правилният отговор е {item['answer']}"
                    }
                elif dataset_name == "hellaswag":
                    choices_text = "\n".join([f"{i+1}) {choice}" for i, choice in enumerate(item['choices'])])
                    formatted = {
                        "instruction": f"Контекст: {item['context']}\nЗавърши текста с най-подходящия край:\n{choices_text}",
                        "input": "",
                        "output": f"Правилният край е отговор {int(item['answer']) + 1}"
                    }
                elif dataset_name == "mathqa":
                    options_text = "\n".join([f"{k.upper()}) {v}" for k, v in item['options'].items()])
                    formatted = {
                        "instruction": f"Задача: {item['problem']} \nОтговори:\n{options_text}",
                        "input": "",
                        "output": f"Правилният отговор е {item['correct'].upper()}. Обоснование: {item['rationale']}"
                    }
                elif dataset_name == "gsm8k":
                    formatted = {
                        "instruction": f"Реши следната математическа задача стъпка по стъпка:\n{item['question']}",
                        "input": "",
                        "output": item['answer']
                    }
                elif dataset_name == "arc_easy" or dataset_name == "arc_challenge":
                    choices = item['choices']['text']
                    labels = item['choices']['label']
                    
                    choices_text = "\n".join([f"{label}) {choice}" for label, choice in zip(labels, choices)])
                    
                    formatted = {
                        "instruction": f"Въпрос: {item['question']}\nОтговори:\n{choices_text}",
                        "input": "",
                        "output": f"Правилният отговор е {item['answer']}"
                    }
                formatted_data.append(formatted)
            
            formatted_splits[split] = formatted_data
                
        return formatted_splits
