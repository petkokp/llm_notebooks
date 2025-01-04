from mmlu.prepare_mmlu import prepare_mmlu
from winogrande.prepare_winogrande import prepare_winogrande
from hellaswag.prepare_hellaswag import prepare_hellaswag
from mathqa.prepare_mathqa import prepare_mathqa
from gsm8k.prepare_gsm8k import prepare_gsm8k
from arc_easy.prepare_arc_easy import prepare_arc_easy
from arc_challenge.prepare_arc_challenge import prepare_arc_challenge
from format_datasets import format_datasets
from save_dataset import save_dataset

def main():
    datasets = {
        "mmlu": prepare_mmlu(),
        "winogrande": prepare_winogrande(),
        "hellaswag": prepare_hellaswag(),
        "mathqa": prepare_mathqa(),
        "gsm8k": prepare_gsm8k(),
        "arc_easy": prepare_arc_easy(),
        "arc_challenge": prepare_arc_challenge(),
    }
    
    for name, data in datasets.items():
        formatted_data = format_datasets(name, data)
        save_dataset(formatted_data, f"{name}_processed.json")

if __name__ == "__main__":
    main()