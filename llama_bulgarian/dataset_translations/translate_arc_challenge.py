from datasets import Dataset, DatasetDict
from translate_dataset import translate_dataset
from arc.preprocess_arc import preprocess_arc_choices
from arc.postprocess_arc import postprocess_arc_choices

INCLUDE_POSTPROCESSING = True

dataset = {
    "url": "allenai/ai2_arc",
    "config": "ARC-Challenge"
}

translation_model = "Helsinki-NLP/opus-mt-tc-big-en-bg"
output_dataset_url = "petkopetkov/arc-challenge-bg"

exclude_columns = ["id", "answerKey"]

translate_dataset(dataset, translation_model, output_dataset_url, batch_size=108, exclude_columns=exclude_columns, preprocess_dataset=preprocess_arc_choices)

if INCLUDE_POSTPROCESSING:
    splits = ["train", "test", "validation"]

    translated_splits = {
        "train": Dataset.from_csv("ai2_arc_translations_train.csv"),
        "test": Dataset.from_csv("ai2_arc_translations_test.csv"),
        "validation": Dataset.from_csv("ai2_arc_translations_validation.csv")
    }
    
    combined_dataset = DatasetDict(translated_splits)

    for split in splits:
        print(f"Processing {split}")
        combined_dataset = postprocess_arc_choices(combined_dataset, split)
        print(f"Processed {split}")

    combined_dataset.push_to_hub(output_dataset_url)
