

# LLMs finetuned on Bulgarian language

This project contains various experiments on finetuning small LLMs (SmolLM, Llama, Gemma) on Bulgarian language. Multiple datasets translated to Bulgarian are included (MMLU, Hellaswag, MathQA, Winogrande, GSM8K, ARC Easy/Challenge, OASST1, OASST2):

- MMLU - https://huggingface.co/datasets/petkopetkov/mmlu-bg
- Hellaswag - https://huggingface.co/datasets/petkopetkov/hellaswag-bg
- MathQA - https://huggingface.co/datasets/petkopetkov/math_qa-bg
- Winogrande - https://huggingface.co/datasets/petkopetkov/winogrande_xl-bg
- GSM8K - https://huggingface.co/datasets/petkopetkov/gsm8k-bg
- ARC Easy - https://huggingface.co/datasets/petkopetkov/arc-easy-bg
- ARC Challenge - https://huggingface.co/datasets/petkopetkov/arc-challenge-bg
- OASST1 - https://huggingface.co/datasets/petkopetkov/oasst1_bg
- OASST2 - https://huggingface.co/datasets/petkopetkov/oasst2_bg
  
More information on the datasets can be found [here](./datasets/README.md).

## Finetune on MMLU, Hellaswag, MathQA, Winogrande, GSM8K, ARC Easy/Challenge

The MMLU, Hellaswag, MathQA, Winogrande, GSM8k, ARC Easy/Challenge datasets are translated to Bulgarian (the scripts for the translations are in the datasets/translations folder).

### How to finetune:

Prepare the datasets for training:

```
python ./datasets/preparation/process_datasets.py
```

Start the training process:

```
python ./training/llm_trainer.py
```

### Custom tokenizer

Since the Llama tokenizer doesn't handle cyrillic tokens very well, a new tokenizer can be trained. It can handle Bulgarian text better but its performance on English text will probably be worse unless english is included in the training data too. The data for the training is the https://huggingface.co/datasets/petkopetkov/chitanka dataset which includes only Bulgarian text.

Prepare the tokenizer training data:

```
python ./training/tokenizer/prepare_tokenizer_data.py --split=validation --lang=bg --docs_to_sample=300000 --save_path=./data
```

Train a new tokenizer:

```
python ./training/tokenizer/train_tokenizer.py
```

### Finetuned models

Finetuned models on the translated MMLU, Hellaswag, MathQA, Winogrande, GSM8k, ARC Easy/Challenge datasets:

- SmolLM2-135M - https://huggingface.co/petkopetkov/SmolLM2-135M-bg
- Llama3.2-1B - https://huggingface.co/petkopetkov/Llama3.2-1B-bg
- Llama3.2-1B-Instruct - https://huggingface.co/petkopetkov/Llama3.2-1B-Instruct-bg
- Llama3.2-1B (custom tokenizer) - https://huggingface.co/petkopetkov/Llama3.2-1B-bg-tokenizer
- Gemma-2-2B - https://huggingface.co/petkopetkov/gemma-2-2b-bg

### Evaluation

To evaluate a model (in this case SmolLM2-135M-bg) on all of the available tasks:

```
lm_eval \
  --model hf \
  --model_args pretrained=petkopetkov/SmolLM2-135M-bg \
  --tasks \
    mmlu_bg,hellaswag_bg,mathqa_bg,winogrande_bg,gsm8k_bg,arc_easy_bg,arc_challenge_bg \
  --device cuda:0 \
  --batch_size auto:4 \
  --include_path ./evaluation/tasks \
  --output_path ./results
```

More information [here](./evaluation/README.md).

## Finetune on OASST1 Bulgarian dataset

OpenAssistant Conversations Dataset (OASST1 and OASST2) translated to Bulgarian language can be used for the finetuning. The datasets are available on https://huggingface.co/datasets/petkopetkov/oasst1_bg and https://huggingface.co/datasets/petkopetkov/oasst2_bg.

### How to finetune:

```
python ./training/oasst_1/finetune.py tuned_model dataset_name instruction_prompt
```

### Example parameters:

```
python ./training/oasst_1/finetune.py petkopetkov/Llama3.2-1B-Instruct-bg petkopetkov/oasst1_bg "Ти си полезен асистент, който отговаря само на български език."
```

### Finetuned models on OASST1 Bulgarian dataset

Finetuned models on the translated OASST1 Bulgarian dataset:

- SmolLM2-1-7B - https://huggingface.co/petkopetkov/SmolLM2-1.7B-bg
- SmolLM2-1-7B-Instruct - https://huggingface.co/petkopetkov/SmolLM2-1.7B-Instruct-bg
- SmolLM-1-7B - https://huggingface.co/petkopetkov/SmolLM-1-7B-bg
- Llama3.2-3B-Instruct - https://huggingface.co/petkopetkov/Llama3.2-3B-Instruct-bg

## Finetune on synthetic dataset from books translated to Bulgarian

The project also contains an experiment with generating synthetic question-answer dataset based on books translated to Bulgarian language using other language models such as [BgGPT](https://huggingface.co/INSAIT-Institute/BgGPT-Gemma-2-9B-IT-v1.0) and finetuning models on this dataset. The dataset is available on https://huggingface.co/datasets/petkopetkov/QABGB.

### Generate synthetic question-answer dataset on Bulgarian books:

```
python ./datasets/preparation/generate_dataset_pairs.py
```

### How to finetune on the synthetic dataset:

The experiment is implemented in the following notebook:

[llama_QABGB_dataset_finetune.ipynb](./training/QABGB/llama_QABGB_dataset_finetune.ipynb)
