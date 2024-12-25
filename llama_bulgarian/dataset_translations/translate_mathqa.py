from datasets import Dataset, DatasetDict
from translate_dataset import translate_dataset
from mathqa_postprocessing.process_options import process_options
from mathqa_postprocessing.process_rationale import process_rationales


INCLUDE_POSTPROCESSING = True

dataset = {
    "url": "allenai/math_qa",
}
translation_model = "Helsinki-NLP/opus-mt-tc-big-en-bg"
output_dataset_url = "petkopetkov/math_qa-bg"

exclude_columns = ["correct", "annotated_formula", "linear_formula", "category"]

translate_dataset(dataset, translation_model, output_dataset_url, batch_size=28, exclude_columns=exclude_columns, checkpoint_interval=112)

if INCLUDE_POSTPROCESSING:
    splits = ["train", "test", "validation"]

    translated_splits = {
        "train": Dataset.from_csv("math_qa_translations_train.csv"),
        "test": Dataset.from_csv("math_qa_translations_test.csv"),
        "validation": Dataset.from_csv("math_qa_translations_validation.csv")
    }
    
    combined_dataset = DatasetDict(translated_splits)

    for split in splits:
        print(f"Processing {split}")
        process_options(combined_dataset, split)
        process_rationales(combined_dataset, split)
        print(f"Processed {split}")

    combined_dataset.push_to_hub(output_dataset_url)