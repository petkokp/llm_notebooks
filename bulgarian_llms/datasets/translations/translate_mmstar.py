from datasets import get_dataset_config_names
from translate_dataset import translate_dataset

INCLUDE_POSTPROCESSING = True

DATASET_NAME = "Lin-Chen/MMStar"

dataset = {
    "url": DATASET_NAME,
}

output_dataset_url = "petkopetkov/MMStar-bg"

exclude_columns = ["index", "image", "answer", "category", "l2_category", "meta_info"]

translate_dataset(dataset, output_dataset_url, batch_size=512, exclude_columns=exclude_columns)