from datasets import Dataset, DatasetDict
from translate_dataset import translate_dataset
from hellaswag.postprocess_hellaswag import postprocess_hellaswag_endings

INCLUDE_POSTPROCESSING = True

dataset = {
    "url": "Rowan/hellaswag",
}
translation_model = "Helsinki-NLP/opus-mt-tc-big-en-bg"
output_dataset_url = "petkopetkov/hellaswag-bg"

exclude_columns = ["ind", "activity_label", "source_id", "split", "split_type", "label"]

translate_dataset(dataset, translation_model, output_dataset_url, exclude_columns=exclude_columns, batch_size=150)

if INCLUDE_POSTPROCESSING:
    splits = ["train", "test", "validation"]

    translated_splits = {
        "train": Dataset.from_csv("hellaswag_translations_train.csv"),
        "test": Dataset.from_csv("hellaswag_translations_test.csv"),
        "validation": Dataset.from_csv("hellaswag_translations_validation.csv")
    }
    
    combined_dataset = DatasetDict(translated_splits)

    for split in splits:
        print(f"Processing {split}")
        postprocess_hellaswag_endings(combined_dataset, split)
        print(f"Processed {split}")

    combined_dataset.push_to_hub(output_dataset_url)
