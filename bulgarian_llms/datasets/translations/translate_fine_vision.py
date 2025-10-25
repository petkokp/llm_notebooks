from datasets import get_dataset_config_names
from translate_dataset import translate_dataset

INCLUDE_POSTPROCESSING = True

DATASET_NAME = "HuggingFaceM4/FineVision"

available_subsets = get_dataset_config_names(DATASET_NAME)

dataset = {
    "url": DATASET_NAME,
    "name": "LLaVA_Instruct_150K",
}

output_dataset_url = "petkopetkov/FineVision-bg"

exclude_columns = ["images", "source", "relevance_ratings", "relevance_min", "visual_dependency_ratings", "visual_dependency_min", "image_correspondence_ratings", "image_correspondence_min", "formatting_ratings", "formatting_min"]

translate_dataset(dataset, output_dataset_url, batch_size=512, exclude_columns=exclude_columns)
