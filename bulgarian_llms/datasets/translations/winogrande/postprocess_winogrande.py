from datasets import Value

def postprocess_sentence(sentence):
    return sentence.replace(" (това е само пример) ", " _ ") # replace the bulgarian expression to get the underscore back in the sentence

def postprocess_winogrande_sentences(dataset, split):
    dataset[split] = dataset[split].map(lambda example: {
        'sentence': postprocess_sentence(example['sentence'])
    })
    dataset[split] = dataset[split].cast_column("answer", Value("string"))
    print(f"Processed {split} dataset")

    return dataset
