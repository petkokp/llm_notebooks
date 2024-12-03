import os
import json
import logging
from tqdm import tqdm
import random
from llama_cpp import Llama
from llama_cpp.llama_speculative import LlamaPromptLookupDecoding

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

DATA_PATH = './archive'
CHARACTER_COUNT = 1000  # Character limit for each subsection
OUTPUT_FILE = 'question_answer_pairs.json'
REPO_ID = "INSAIT-Institute/BgGPT-7B-Instruct-v0.2-GGUF" #"INSAIT-Institute/BgGPT-Gemma-2-9B-IT-v1.0-GGUF"
GGUF_FILE = "BgGPT-7B-Instruct-v0.2.F16.gguf" # "BgGPT-Gemma-2-9B-IT-v1.0.F16.gguf"
GPU_LAYERS = 34

llm = Llama.from_pretrained(
	repo_id=REPO_ID,
	filename=GGUF_FILE,
    n_gpu_layers=GPU_LAYERS,
    n_ctx=2048,
    flash_attn=True,
)

def generate_prompt(input_text):
    """Generate a prompt for the LLM."""
    return f"""
    Генерирай въпрос-отговор двойки спрямо предоставения текст със следната JSON структура където ключовете са на английски език, а стойностите на български език:
        {{
            "question": "<Въпрос на български>",
            "answer": "<Точен отговор на български>",
        }},
        
        Генерирай въпроси само на български, които могат да бъдат отговорени директно чрез текста. Избягвай неясни и спекулативни въпроси. Увери се, че отговорите са кратки, точни и основани на предоставения текст.

        Това е текста:
        {input_text}
        """

def has_valid_response_structure(response):
    """Check if the response string has only the keys 'question' and 'answer'."""
    try:
        # Parse the JSON string
        response_json = json.loads(response)
        
        # Define required keys
        required_keys = {"question", "answer"}
        response_keys = set(response_json.keys())
        
        # Ensure only the required keys are present
        return response_keys == required_keys
    except (json.JSONDecodeError, AttributeError):
        # Return False if the response is not a valid JSON or is malformed
        return False


def split_text_into_subsections(text, character_count):
    """Split text into chunks of specified character count."""
    return [text[i:i + character_count] for i in range(0, len(text), character_count)]

def load_existing_data(file_path):
    """Load existing JSON data from the file."""
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            return json.load(file)
    return []

def save_data_to_file(data, file_path):
    """Save data to a JSON file."""
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)

def process_subsection(subsection):
    """Process a single subsection using the LLM."""
    prompt = generate_prompt(subsection)
    
    response = llm.create_chat_completion(
        messages=[
            {"role": "user", "content": prompt},
        ],
        max_tokens=2048,        # Choose maximum generated tokens
        temperature=0.1,
        top_p=0.9,
        repeat_penalty=1.0,
        stop=["<eos>","<end_of_turn>"],
        response_format={
            "type": "json_object",
            "schema": {
                "type": "object",
                "properties": {"question": {"type": "string"}, "answer": {"type": "string"}},
                "required": ["question", "answer"],
            },
        },
    )
    
    content = None
    is_valid = False
    
    try:
        content = response['choices'][0]["message"]["content"]
    except Exception as e:
        logging.error(f"COULD NOT PROCESS CONTENT: {e}")
        return None
    
    try:
        is_valid = has_valid_response_structure(content)
    except Exception as e:
        logging.error(f"COULD NOT VALIDATE CONTENT: {e}")
        return None

    try:
        if is_valid:  
            response_json = json.loads(content)      
            response_json["context"] = subsection
            return response_json
    except Exception as e:
        logging.error(f"COULD NOT PARSE JSON: {e}")
        return None

def process_text_file(file_path, character_count, output_file):
    """Process a text file and extract question-answer pairs."""
    existing_data = load_existing_data(output_file)
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            file_content = f.read()

        subsections = split_text_into_subsections(file_content, character_count)
        logging.info(f"Split file into {len(subsections)} subsections.")
        
        batch_size = 50
        batch = []

        for subsection in tqdm(subsections, desc=f"Processing {os.path.basename(file_path)}"):
            processed = process_subsection(subsection)
            if processed:
                batch.append(processed)
                # If batch size reaches the limit, save the data and clear the batch
                if len(batch) == batch_size:
                    existing_data.extend(batch)
                    save_data_to_file(existing_data, output_file)
                    batch = []  # Clear the batch

        # Save any remaining data in the batch after processing all subsections
        if batch:
            existing_data.extend(batch)
            save_data_to_file(existing_data, output_file)
        
        logging.info(f"Processed  entries from {file_path}.")
    except Exception as e:
        logging.error(f"Error processing file {file_path}: {e}")
                
def process_all_files(data_path, character_count, output_file):
    """Process all text files in the specified directory."""
    file_paths = []
    for root, _, files in os.walk(data_path):
        for file in files:
            if file.endswith('.txt'):
                file_paths.append(os.path.join(root, file))
    
    random.shuffle(file_paths)
    
    for file_path in file_paths:
        process_text_file(file_path, character_count, output_file)

# Main execution
if __name__ == "__main__":
    logging.info("Starting processing of text files.")
    process_all_files(DATA_PATH, CHARACTER_COUNT, OUTPUT_FILE)
    logging.info("Processing completed.")
