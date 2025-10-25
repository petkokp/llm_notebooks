from datasets import get_dataset_config_names
from translate_dataset import translate_dataset

INCLUDE_POSTPROCESSING = True

DATASET_NAME = "lmms-lab/MME"

dataset = {
    "url": DATASET_NAME,
}

output_dataset_url = "petkopetkov/MME-bg"

exclude_columns = ["question_id", "image", "category"]

translate_dataset(dataset, output_dataset_url, batch_size=512, exclude_columns=exclude_columns)
