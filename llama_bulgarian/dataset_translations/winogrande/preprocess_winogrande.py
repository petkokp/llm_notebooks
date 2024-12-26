def preprocess_sentence(sentence):
    return sentence.replace(" _ ", " (this is just an example) ") # replace underscore with english expression and replace it after translation to preserve the underscore

def preprocess_winogrande_sentences(dataset, split):
    dataset[split] = dataset[split].map(lambda example: {
        'sentence': preprocess_sentence(example['sentence'])
    })
    
    print(f"Processed {split} dataset")
    
    return dataset
