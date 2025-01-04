

# Llama finetuned on Bulgarian language

This project contains experiments on finetuning Llama models on Bulgarian language. Multiple datasets translated to Bulgarian are included (MMLU, Hellaswag, MathQA, Winogrande, GSM8k, OASST1, OASST2)

## Finetune on MMLU, Hellaswag, MathQA, Winogrande, GSM8k

The MMLU, Hellaswag, MathQA, Winogrande, GSM8k datasets are translated to Bulgarian (the scripts for the translations are in the datasets/translations folder).

## How to finetune:

Prepare the datasets for training:

```python ./datasets/preparation/process_datasets.py```

Start the training process:

```python ./training/llm_trainer.py```

## Finetune on OASST1 dataset

OpenAssistant Conversations Dataset (OASST1 and OASST2) translated to Bulgarian language is used for the finetuning. The datasets are available on https://huggingface.co/datasets/petkopetkov/oasst1_bg and https://huggingface.co/datasets/petkopetkov/oasst2_bg.

### How to finetune:

```python ./finetune/finetune.py tuned_model dataset_name instruction_prompt```

### Example:

```python finetune.py petkopetkov/Llama3.2-1B-Instruct-bg petkopetkov/oasst1_bg "Ти си полезен асистент, който отговаря само на български език."```

## Finetune on synthetic dataset from books translated to Bulgarian

The project also contains an experiment with generating synthetic question-answer dataset based on books translated to Bulgarian language using other language models such as [BgGPT](https://huggingface.co/INSAIT-Institute/BgGPT-Gemma-2-9B-IT-v1.0) and finetuning Llama models on this dataset. The synthetic dataset generation is implemented in the [dataset folder](./dataset/generate_dataset_pairs.py).

### Generate synthetic question-answer dataset on Bulgarian books:

```python ./dataset/generate_dataset_pairs.py```

### How to finetune on the synthetic dataset:

[llama_QABGB_dataset_finetune.ipynb](./llama_QABGB_dataset_finetune.ipynb)

### Finetuned models

Llama-3.2-1B and Llama-3.2-3B models are finetuned using the OASST1 dataset translated to Bulgarian language. They are available on https://huggingface.co/petkopetkov/Llama3.2-1B-Instruct-bg and https://huggingface.co/petkopetkov/Llama3.2-3B-Instruct-bg.

