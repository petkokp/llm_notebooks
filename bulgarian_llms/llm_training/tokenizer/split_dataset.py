from datasets import load_dataset, DatasetDict

DATASET_PATH = "petkopetkov/chitanka"

dataset = load_dataset(DATASET_PATH, split='train')

train_testvalid = dataset.train_test_split(test_size=0.1)
test_valid = train_testvalid['test'].train_test_split(test_size=0.5)
train_test_valid_dataset = DatasetDict({
    'train': train_testvalid['train'],
    'test': test_valid['test'],
    'validation': test_valid['train']
})

train_test_valid_dataset.push_to_hub(DATASET_PATH)
