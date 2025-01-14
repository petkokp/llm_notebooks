from transformers import AutoTokenizer
import logging
from pathlib import Path
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_training_corpus(paths):
    for path_str in paths:
        path = Path(path_str)
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield line

def compare_tokenization(base_tokenizer, new_tokenizer, text, language):
    logging.info(f"\n{language} Text: {text}")
    
    base_tokens = base_tokenizer.tokenize(text)
    base_ids = base_tokenizer.convert_tokens_to_ids(base_tokens)
    logging.info("\nBase Tokenizer:")
    logging.info(f"Number of tokens: {len(base_tokens)}")
    logging.info(f"Tokens: {base_tokens}")
    logging.info(f"Input IDs: {base_ids}")
    
    new_tokens = new_tokenizer.tokenize(text)
    new_ids = new_tokenizer.convert_tokens_to_ids(new_tokens)
    logging.info("\nNew Tokenizer:")
    logging.info(f"Number of tokens: {len(new_tokens)}")
    logging.info(f"Tokens: {new_tokens}")
    logging.info(f"Input IDs: {new_ids}")

pretrained_tokenizer_name = "unsloth/Llama-3.2-1B"
base_tokenizer = AutoTokenizer.from_pretrained(pretrained_tokenizer_name)

tokenizer_config = {
    "vocab_size": 128256,
    "min_frequency": 2,
    "special_tokens": base_tokenizer.special_tokens_map,
    "use_padding": True,
    "use_truncation": True
}

data_paths = ["data/bg.txt"]
total_lines = sum(1 for _ in get_training_corpus(data_paths))

new_tokenizer = base_tokenizer.train_new_from_iterator(
    tqdm(get_training_corpus(data_paths), total=total_lines, desc="Training Tokenizer"),
    vocab_size=tokenizer_config["vocab_size"],
    min_frequency=tokenizer_config["min_frequency"]
)

# texts for testing the tokenizers
bulgarian_text = "Здравейте, как сте днес? Това е български текст."
english_text = "Hello, how are you today? This is an English text."

compare_tokenization(base_tokenizer, new_tokenizer, bulgarian_text, "Bulgarian")
compare_tokenization(base_tokenizer, new_tokenizer, english_text, "English")

base_vocab = base_tokenizer.get_vocab()
new_vocab = new_tokenizer.get_vocab()

new_only_tokens = set(new_vocab.keys()) - set(base_vocab.keys())
if new_only_tokens:
    sample_new_tokens = list(new_only_tokens)[:10]  # show first 10 new tokens
    logging.info("\nSample of new tokens added to vocabulary:")
    for token in sample_new_tokens:
        logging.info(f"Token: {token}, ID: {new_vocab[token]}")

output_dir = "bulgarian_tokenizer"
new_tokenizer.save_pretrained(output_dir)
logging.info(f"\nNew tokenizer saved to {output_dir}")