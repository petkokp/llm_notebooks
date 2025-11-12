from transformers import AutoTokenizer

base_tok = AutoTokenizer.from_pretrained("unsloth/Qwen3-VL-2B-Instruct", trust_remote_code=True, use_fast=True)

relearn_tok = AutoTokenizer.from_pretrained("../tokenizers/Qwen3-VL-2B-Instruct_relearn", trust_remote_code=True, use_fast=True)

print(f"Base tokenizer vocab size:    {len(base_tok)}")
print(f"Relearned tokenizer vocab size: {len(relearn_tok)}")

# Compare some tokens
sample_sentence = "This is a sample sentence to compare tokenizers."
sample_sentence_bg = "Това е примерно изречение за сравнение на токенизатори."

base_tokens = base_tok.tokenize(sample_sentence)
relearn_tokens = relearn_tok.tokenize(sample_sentence)

print(f"Base tokenizer tokens count: {len(base_tokens)}")
print(f"Relearned tokenizer tokens count: {len(relearn_tokens)}")

base_tokens_bg = base_tok.tokenize(sample_sentence_bg)
relearn_tokens_bg = relearn_tok.tokenize(sample_sentence_bg)

print(f"Base tokenizer tokens count (BG): {len(base_tokens_bg)}")
print(f"Relearned tokenizer tokens count (BG): {len(relearn_tokens_bg)}")
