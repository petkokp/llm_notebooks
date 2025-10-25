from datasets import get_dataset_config_names
from translate_dataset import translate_dataset

INCLUDE_POSTPROCESSING = True

DATASET_NAME = "lmms-lab/MMMU"

dataset = {
    "url": DATASET_NAME,
}

output_dataset_url = "petkopetkov/MMMU-bg"

exclude_columns = ["id", "img_type", "answer", "topic_difficulty", "question_type", "subfield", "image_1", "image_2", "image_3", "image_4", "image_5", "image_6", "image_7"]

translate_dataset(dataset, output_dataset_url, batch_size=512, exclude_columns=exclude_columns)