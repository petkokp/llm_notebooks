from unsloth import FastLanguageModel

output = "../../smollm2-360M_finetuned_model"

REPO_PATH = "petkopetkov/SmolLM2-360M-bg"

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=output,
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=True,
)

model.push_to_hub_merged(REPO_PATH, tokenizer, save_method = "merged_16bit")