

from translate_dataset import translate_dataset

dataset = {
    "url": "Rowan/hellaswag",
}
translation_model = "Helsinki-NLP/opus-mt-tc-big-en-bg"
output_dataset_url = "petkopetkov/hellaswag-bg"

exclude_columns = ["ind", "activity_label", "source_id", "split", "split_type", "label"]

translate_dataset(dataset, translation_model, output_dataset_url, exclude_columns=exclude_columns, batch_size=150)
