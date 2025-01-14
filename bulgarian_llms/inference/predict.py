import torch
from transformers import pipeline

MODEL_ID = "petkopetkov/Llama3.2-3B-Instruct-bg"
PROMPT = "Колко е 2+2?"

def predict(prompt=PROMPT, model_id=MODEL_ID):
    pipe = pipeline(
        "text-generation", 
        model=model_id, 
        torch_dtype=torch.bfloat16, 
        device_map="auto",
        repetition_penalty=1.2,
    )

    return pipe(prompt)[0]['generated_text']

print(predict())