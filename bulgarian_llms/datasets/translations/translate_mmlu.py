from datasets import Dataset, DatasetDict
from translate_dataset import translate_dataset
from mmlu.postprocess_mmlu import postprocess_mmlu_choices

INCLUDE_POSTPROCESSING = True

dataset = {
    "url": "cais/mmlu",
    "config": "all"
}

translation_model = "Helsinki-NLP/opus-mt-tc-big-en-bg"
output_dataset_url = "petkopetkov/mmlu-bg"

exclude_columns = ["subject", "answer"]

#translate_dataset(dataset, translation_model, output_dataset_url, batch_size=38, exclude_columns=exclude_columns, checkpoint_interval=532)

if INCLUDE_POSTPROCESSING:
    splits = ["auxiliary_train", "test", "dev"]

    translated_splits = {
        "auxiliary_train": Dataset.from_csv("mmlu_translations_auxiliary_train.csv"),
        "test": Dataset.from_csv("mmlu_translations_test.csv"),
        "dev": Dataset.from_csv("mmlu_translations_dev.csv")
    }
    
    combined_dataset = DatasetDict(translated_splits)

    for split in splits:
        print(f"Processing {split}")
        postprocess_mmlu_choices(combined_dataset, split)
        print(f"Processed {split}")

    combined_dataset.push_to_hub(output_dataset_url)
