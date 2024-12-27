from translate_dataset import translate_dataset

dataset = {
    "url": "cais/mmlu",
    "config": "all"
}

translation_model = "Helsinki-NLP/opus-mt-tc-big-en-bg"
output_dataset_url = "petkopetkov/mmlu-bg"

exclude_columns = ["subject", "answer"]

translate_dataset(dataset, translation_model, output_dataset_url, batch_size=32, checkpoint_interval=532)
