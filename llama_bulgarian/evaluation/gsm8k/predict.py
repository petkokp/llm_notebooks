import torch
from transformers import pipeline

MODEL_ID = "petkopetkov/Llama3.2-3B-Instruct-bg"
PROMPT = "Колко е 2+2?"

def predict(prompt=PROMPT, max_new_tokens=5000, model_id=MODEL_ID):
    pipe = pipeline(
        "text-generation", 
        model=model_id, 
        torch_dtype=torch.bfloat16, 
        device_map="auto",
        max_new_tokens=max_new_tokens,
    )

    return pipe(prompt)[0]['generated_text']