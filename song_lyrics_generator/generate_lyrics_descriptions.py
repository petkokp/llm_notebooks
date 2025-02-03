import torch
from datasets import load_dataset
import os
import json
from tqdm import tqdm
import time
from transformers import pipeline

INPUT_DATASET_REPO = "vishnupriyavr/spotify-million-song-dataset"
OUTPUT_DATASET_REPO = "petkopetkov/spotify-million-song-dataset-descriptions"
HF_DATASET_SPLIT = "train"
CHECKPOINT_FILE = "songs_descriptions_checkpoint.json"

dataset = load_dataset(INPUT_DATASET_REPO, split=HF_DATASET_SPLIT)

if os.path.exists(CHECKPOINT_FILE):
    with open(CHECKPOINT_FILE, "r") as f:
        checkpoint_data = json.load(f)
else:
    checkpoint_data = {}

completed_indices = set(checkpoint_data.keys())

pipe = pipeline(
    "text-generation",
    model="google/gemma-2-2b-it",
    model_kwargs={"torch_dtype": torch.bfloat16},
    device="cuda",
)

def generate_description(artist, song, lyrics):
    prompt = f"""Describe the following song based on the lyrics in a **comma-separated list** of adjectives and stylistic traits (can be more complex expressions or just simple words that a person would use to describe the song). 
    The description should include **mood, atmosphere, style, lyrical structure, and the artist's name**.
    
    Artist: {artist}
    Song: {song}
    Lyrics: {lyrics[:2000]}
    
    Description:"""

    try:
        messages = [
            {"role": "user", "content": prompt}
        ]

        outputs = pipe(messages, max_new_tokens=100, batch_size=32)
        
        return outputs[0]["generated_text"][-1]["content"].strip()
    except Exception as e:
        print(f"Error generating description: {e}")
        return None

requests_made = 0
start_time = time.time()

if "description" not in dataset.column_names:
    dataset = dataset.add_column("description", [""] * len(dataset))
    
new_descriptions = dataset["description"]

for i, desc in checkpoint_data.items():
    new_descriptions[int(i)] = desc

for i in tqdm(range(len(dataset)), desc="Generating descriptions"):
    if str(i) in completed_indices:
        continue

    artist = dataset[i]["artist"]
    song = dataset[i]["song"]
    lyrics = dataset[i]["text"]

    description = generate_description(artist, song, lyrics)
    
    if description:
        new_descriptions[i] = description
        checkpoint_data[str(i)] = description
        completed_indices.add(str(i))

    if i % 10 == 0:
        with open(CHECKPOINT_FILE, "w") as f:
            json.dump(checkpoint_data, f)
            
dataset = dataset.remove_columns("description")
dataset = dataset.add_column("description", new_descriptions)

dataset.push_to_hub(OUTPUT_DATASET_REPO)

print("Dataset successfully updated and pushed to Hugging Face Hub!")
