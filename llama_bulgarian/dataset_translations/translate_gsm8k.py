from translate_dataset import translate_dataset

dataset = {
    "url": "openai/gsm8k",
    "config": "main"
}
translation_model = "Helsinki-NLP/opus-mt-tc-big-en-bg"
output_dataset_url = "petkopetkov/gsm8k-bg"

translate_dataset(dataset, translation_model, output_dataset_url, batch_size=32)
