from datasets import Dataset, DatasetDict, Features, Value
from translate_dataset import translate_dataset
from winogrande.preprocess_winogrande import preprocess_winogrande_sentences
from winogrande.postprocess_winogrande import postprocess_winogrande_sentences

INCLUDE_POSTPROCESSING = True

dataset = {
    "url": "allenai/winogrande",
    "config": "winogrande_xl"
}

translation_model = "Helsinki-NLP/opus-mt-tc-big-en-bg"
output_dataset_url = "petkopetkov/winogrande_xl-bg"

exclude_columns = ["answer"]

translate_dataset(dataset, translation_model, output_dataset_url, batch_size=102, exclude_columns=exclude_columns, preprocess_dataset=preprocess_winogrande_sentences)

if INCLUDE_POSTPROCESSING:
    splits = ["train", "test", "validation"]

    translated_splits = {
        "train": Dataset.from_csv("winogrande_translations_train.csv"),
        "test": Dataset.from_csv("winogrande_translations_test.csv"),
        "validation": Dataset.from_csv("winogrande_translations_validation.csv")
    }
    
    from datasets import DatasetDict
    
    features = Features({
        "sentence": Value("string"),
        "option1": Value("string"),
        "option2": Value("string"),
        "answer": Value("int64"), 
    })

    train_dataset = translated_splits["train"].cast(features)
    test_dataset = translated_splits["test"].cast(features)
    validation_dataset = translated_splits["validation"].cast(features)
    
    translated_splits = {
        "train": train_dataset,
        "test": test_dataset,
        "validation": validation_dataset
    }
    
    combined_dataset = DatasetDict(translated_splits)

    for split in splits:
        print(f"Processing {split}")
        postprocess_winogrande_sentences(combined_dataset, split)
        print(f"Processed {split}")

    combined_dataset.push_to_hub(output_dataset_url)