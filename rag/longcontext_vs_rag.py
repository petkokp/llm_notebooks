import time
import torch
import random
import string
import numpy as np
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

# ==========================================
# 1. EXPERIMENT CONFIGURATION
# ==========================================
MODEL_NAME = "Qwen/Qwen3-0.6B"
OUTPUT_FILE = "rag_vs_long_context_results.csv"

# Experimental Variables
CONTEXT_LENGTHS = [4000, 14000, 24000] # Small, Medium, Large
SAMPLES_PER_LENGTH = 15                # 15 samples per length = 45 total runs
RAG_CHUNK_SIZE = 300
RAG_TOP_K = 4                          # Increased from 3 to 4 to give RAG a better chance

# ==========================================
# 2. INITIALIZATION
# ==========================================
print(">>> [Setup] Initializing Engines...")

# vLLM (Allocating 75% GPU to leave room for embeddings)
llm_engine = LLM(
    model=MODEL_NAME,
    trust_remote_code=True,
    gpu_memory_utilization=0.75, 
    max_model_len=32768,
    enforce_eager=True,
)
sampling_params = SamplingParams(temperature=0.0, max_tokens=50)

# Embeddings
device = "cuda" if torch.cuda.is_available() else "cpu"
embed_model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

# Data Stream
dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="test", streaming=True)
data_iter = iter(dataset)

# ==========================================
# 3. CORE FUNCTIONS
# ==========================================

def get_text_chunk(min_words):
    """Accumulates text until it hits the requested word count."""
    text_accum = ""
    global data_iter
    while len(text_accum.split()) < min_words:
        try:
            item = next(data_iter)
            if len(item['text']) > 50:
                text_accum += item['text'] + "\n"
        except StopIteration:
            # Reload dataset if we run out
            data_iter = iter(load_dataset("wikitext", "wikitext-103-raw-v1", split="test", streaming=True))
    
    words = text_accum.split()
    return " ".join(words[:min_words])

def apply_template(system, user):
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)

def run_single_trial(target_len, trial_id):
    # 1. Prepare Data
    haystack = get_text_chunk(min_words=int(target_len * 0.8)) # approx conversion tokens->words
    
    # 2. Prepare Needle
    code = ''.join(random.choices(string.ascii_uppercase + string.digits, k=8))
    needle = f" [SECRET_DATA: The alien mothership is hidden in sector 'NEBULA-{code}-{trial_id}'.] "
    query = "Where is the alien mothership hidden?"
    ground_truth = f"NEBULA-{code}-{trial_id}"
    
    # 3. Insert Needle (Random Position)
    words = haystack.split()
    pos_percent = random.random() # 0.0 to 1.0
    insert_idx = int(len(words) * pos_percent)
    haystack = " ".join(words[:insert_idx] + [needle] + words[insert_idx:])
    
    results = {
        "context_length": target_len,
        "needle_depth": round(pos_percent, 2),
        "ground_truth": ground_truth
    }

    # --- METHOD A: RAG ---
    t0 = time.time()
    # Chunking
    chunk_words = haystack.split()
    chunks = [" ".join(chunk_words[i:i+RAG_CHUNK_SIZE]) for i in range(0, len(chunk_words), RAG_CHUNK_SIZE)]
    
    # Retrieve
    chunk_emb = embed_model.encode(chunks, convert_to_tensor=True)
    query_emb = embed_model.encode(query, convert_to_tensor=True)
    scores = torch.matmul(chunk_emb, query_emb)
    top_k_indices = torch.topk(scores, k=RAG_TOP_K).indices.cpu().numpy()
    retrieved_txt = "\n---\n".join([chunks[i] for i in top_k_indices])
    
    # Generate
    rag_prompt = apply_template("Answer strictly based on context.", f"Context:\n{retrieved_txt}\n\nQuestion: {query}")
    rag_out = llm_engine.generate([rag_prompt], sampling_params)[0].outputs[0].text.strip()
    
    results["rag_time"] = time.time() - t0
    results["rag_success"] = 1 if ground_truth in rag_out else 0
    results["rag_tokens"] = len(tokenizer.encode(rag_prompt))
    
    # --- METHOD B: LONG CONTEXT ---
    t1 = time.time()
    lc_prompt = apply_template("You are a helpful assistant. Read the text and answer.", f"Text:\n{haystack}\n\nQuestion: {query}")
    lc_out = llm_engine.generate([lc_prompt], sampling_params)[0].outputs[0].text.strip()
    
    results["lc_time"] = time.time() - t1
    results["lc_success"] = 1 if ground_truth in lc_out else 0
    results["lc_tokens"] = len(tokenizer.encode(lc_prompt))
    
    return results

# ==========================================
# 4. MAIN LOOP
# ==========================================
all_data = []

print(f"\n>>> Starting Study: {len(CONTEXT_LENGTHS)} Context Sizes x {SAMPLES_PER_LENGTH} Samples")
print("="*70)

progress_bar = tqdm(total=len(CONTEXT_LENGTHS) * SAMPLES_PER_LENGTH, desc="Running Experiments")

for length in CONTEXT_LENGTHS:
    for i in range(SAMPLES_PER_LENGTH):
        trial_data = run_single_trial(length, i)
        all_data.append(trial_data)
        progress_bar.update(1)
        
        # Optional: Print error if LC fails (Debug check)
        if trial_data['lc_success'] == 0:
            tqdm.write(f"⚠️ LC Failed at {length} tokens (Depth: {trial_data['needle_depth']})")

progress_bar.close()

# ==========================================
# 5. ANALYSIS & EXPORT
# ==========================================
df = pd.DataFrame(all_data)
df.to_csv(OUTPUT_FILE, index=False)

print("\n" + "="*70)
print("FINAL STUDY RESULTS")
print("="*70)

# Group by Context Length to show scaling
summary = df.groupby("context_length").agg({
    "rag_success": "mean",
    "lc_success": "mean",
    "rag_time": "mean",
    "lc_time": "mean",
    "rag_tokens": "mean",
    "lc_tokens": "mean"
}).reset_index()

# Formatting for display
summary["rag_success"] = (summary["rag_success"] * 100).map("{:.1f}%".format)
summary["lc_success"]  = (summary["lc_success"] * 100).map("{:.1f}%".format)
summary["rag_time"]    = summary["rag_time"].map("{:.3f}s".format)
summary["lc_time"]     = summary["lc_time"].map("{:.3f}s".format)
summary["rag_tokens"]  = summary["rag_tokens"].astype(int)
summary["lc_tokens"]   = summary["lc_tokens"].astype(int)

print(summary.to_string(index=False))
print("="*70)
print(f"Detailed data saved to: {OUTPUT_FILE}")
print("Tip: Use the CSV to plot 'Needle Depth' vs 'Success' to check for Lost-in-the-Middle.")