

# Llama finetuned on Bulgarian language

This project contains experiments on finetuning Llama models on Bulgarian language. OpenAssistant Conversations Dataset (OASST1 and OASST2) translated to Bulgarian language is used for the finetuning. The datasets are available on https://huggingface.co/datasets/petkopetkov/oasst1_bg and https://huggingface.co/datasets/petkopetkov/oasst2_bg.

The project also contains an experiment with generating synthetic question-answer dataset based on books translated to Bulgarian language using other language models such as [BgGPT](https://huggingface.co/INSAIT-Institute/BgGPT-Gemma-2-9B-IT-v1.0) and finetuning Llama models on this dataset. The synthetic dataset generation is implemented in the [dataset folder](./dataset/generate_dataset_pairs.py).

### How to finetune:

```python ./finetune/finetune.py tuned_model dataset_name instruction_prompt```

### Example:

```python finetune.py petkopetkov/Llama-3.2-1B-Instruct-bg petkopetkov/oasst1_bg "Ти си полезен асистент, който отговаря само на български език."```

### Generate synthetic question-answer dataset on Bulgarian books:

```python ./dataset/generate_dataset_pairs.py```

### Finetune Llama models on the synthetic dataset:

[llama_QABGB_dataset_finetune.ipynb](./llama_QABGB_dataset_finetune.ipynb)

### Finetuned models

Llama-3.2-1B and Llama-3.2-3B models are finetuned using the OASST1 dataset translated to Bulgarian language. They are available on https://huggingface.co/petkopetkov/Llama3.2-1B-Instruct-bg and https://huggingface.co/petkopetkov/Llama3.2-3B-Instruct-bg.

