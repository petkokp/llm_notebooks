import torch
from transformers import pipeline

MODEL_ID = "petkopetkov/Llama3.2-3B-Instruct-bg"
PROMPT = "Колко е 2+2?"

def predict(model_id=MODEL_ID, prompt=PROMPT):
    pipe = pipeline(
        "text-generation", 
        model=model_id, 
        torch_dtype=torch.bfloat16, 
        device_map="auto",
    )

    return pipe(PROMPT)[0]['generated_text']