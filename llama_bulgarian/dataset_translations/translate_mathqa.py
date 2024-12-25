from translate_dataset import translate_dataset

dataset = {
    "url": "allenai/math_qa",
}
translation_model = "Helsinki-NLP/opus-mt-tc-big-en-bg"
output_dataset_url = "petkopetkov/math_qa-bg"

exclude_columns = ["correct", "annotated_formula", "linear_formula", "category"]

translate_dataset(dataset, translation_model, output_dataset_url, batch_size=28, exclude_columns=exclude_columns, checkpoint_interval=112)
