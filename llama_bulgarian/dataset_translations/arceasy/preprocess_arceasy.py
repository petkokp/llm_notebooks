def preprocess_arceasy_choices(dataset, split):
    flattened = dataset.flatten() # splits "choices" into two columns - "choices.text" and "choices.label" to allow easier translation of the text
    print(f"Processed {split} dataset")
    return flattened
