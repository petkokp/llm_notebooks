import logging
from pathlib import Path
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.normalizers import NFKC
from tokenizers.processors import ByteLevel as ByteLevelProcessor
from transformers import AutoTokenizer
from tqdm import tqdm

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 1. Load the Pre-trained Llama 3.2 Tokenizer
pretrained_tokenizer_name = "unsloth/Llama-3.2-1B"
logging.info(f"Loading pre-trained tokenizer: {pretrained_tokenizer_name}")
llama_tokenizer = AutoTokenizer.from_pretrained(pretrained_tokenizer_name)
logging.info(f"Pre-trained tokenizer loaded successfully. Vocabulary size: {llama_tokenizer.vocab_size}")

# 2. Prepare Your Bulgarian Training Data
data_paths = ["data/bg.txt"]
logging.info(f"Preparing training data from: {data_paths}")
for path_str in data_paths:
    path = Path(path_str)
    if not path.exists():
        logging.error(f"Training data file not found: {path}")
        raise FileNotFoundError(f"Training data file not found: {path}")

def get_training_corpus(paths):
    for path_str in paths:
        path = Path(path_str)
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield line

total_lines = sum(1 for _ in get_training_corpus(data_paths))
logging.info(f"Found {total_lines} lines in the training data.")

# 3. Initialize a New Tokenizer based on Llama's Configuration
logging.info("Initializing the new tokenizer...")
new_tokenizer = Tokenizer(BPE())
new_tokenizer.pre_tokenizer = ByteLevel()
new_tokenizer.normalizer = NFKC()
new_tokenizer.post_processor = ByteLevelProcessor(trim_offsets=True)
special_tokens = list(llama_tokenizer.special_tokens_map.values())
logging.info("New tokenizer initialized with Llama configuration.")

# 4. Initialize the Trainer
trainer = BpeTrainer(
    vocab_size=16000,
    min_frequency=2,
    special_tokens=special_tokens,
    initial_alphabet=ByteLevel.alphabet(),
    show_progress=True
)
logging.info(f"BPE Trainer initialized. Target vocabulary size: {trainer.vocab_size}")

# 5. Train the New Tokenizer on Your Bulgarian Data with Progress Bar
logging.info("Starting tokenizer training...")
new_tokenizer.train_from_iterator(
    tqdm(get_training_corpus(data_paths), total=total_lines, desc="Training Tokenizer"),
    trainer=trainer,
    length=total_lines
)
logging.info("Tokenizer training finished.")

# 6. Add the Original Llama Tokens to the New Tokenizer (Optimized) with Progress Bar
logging.info("Adding original Llama tokens (optimized)...")
original_vocab_set = set(llama_tokenizer.get_vocab().keys())
new_vocab_set = set(new_tokenizer.get_vocab().keys())
tokens_to_add = list(original_vocab_set - new_vocab_set)

if tokens_to_add:
    logging.info(f"Found {len(tokens_to_add)} original Llama tokens to add.")
    new_tokenizer.add_tokens(tokens_to_add)
    logging.info(f"Successfully added {len(tokens_to_add)} original Llama tokens.")
else:
    logging.info("All original Llama tokens are already in the new tokenizer.")

# 7. Save the Extended Tokenizer
extended_tokenizer_path = "bg_tokenizer"
Path(extended_tokenizer_path).mkdir(parents=True, exist_ok=True)
tokenizer_save_path = str(Path(extended_tokenizer_path) / "tokenizer.json")
logging.info(f"Saving extended tokenizer (tokenizer.json) to: {tokenizer_save_path}")
new_tokenizer.save(tokenizer_save_path)
logging.info(f"Extended tokenizer (tokenizer.json) saved successfully.")

extended_hf_tokenizer = AutoTokenizer.from_pretrained(pretrained_tokenizer_name)
extended_hf_tokenizer.tokenizer = new_tokenizer
hf_save_path = str(Path(extended_tokenizer_path))
logging.info(f"Saving extended tokenizer (Hugging Face format) to: {hf_save_path}")
extended_hf_tokenizer.save_pretrained(hf_save_path)
logging.info(f"Extended tokenizer (Hugging Face format) saved successfully.")

# 8. Load and Test the Extended Tokenizer
logging.info("Loading and testing the extended tokenizer...")
loaded_tokenizer = AutoTokenizer.from_pretrained(extended_tokenizer_path)

bulgarian_text_example = "Здравейте, как сте днес? Това е български текст."
tokens = loaded_tokenizer.tokenize(bulgarian_text_example)
input_ids = loaded_tokenizer.convert_tokens_to_ids(tokens)

logging.info(f"Example Bulgarian text: {bulgarian_text_example}")
logging.info(f"Tokens: {tokens}")
logging.info(f"Input IDs: {input_ids}")

logging.info("Extended tokenizer pipeline completed.")