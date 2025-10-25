from datasets import get_dataset_config_names
from translate_dataset import translate_dataset

INCLUDE_POSTPROCESSING = True

DATASET_NAME = "AI4Math/MathVista"

dataset = {
    "url": DATASET_NAME,
}

output_dataset_url = "petkopetkov/MathVista-bg"

exclude_columns = ["pid", "image", "decoded_image", "precision", "question_type", "answer_type", "metadata"]

translate_dataset(dataset, output_dataset_url, batch_size=512, exclude_columns=exclude_columns)