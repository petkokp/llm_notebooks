# Datasets

The datasets that were chosen are MMLU, Hellaswag, MathQA, Winogrande, GSM8K and ARC Easy/Challenge. Each dataset wastranslated to Bulgarian language with the [opus-mt-tc-big-en-bg model](https://huggingface.co/Helsinki-NLP/opus-mt-tc-big-en-bg). There are specific preprocessing and postprocessing scripts that are applied to some of the datasets. All of the translations code is in the [translations directory](./translations/). All of the datasets contain ~227,000 train samples.

After the datasets are translated and uploaded to HuggingFace, they can be loaded, processed and formatted into prompts that are needed for the finetuning process. All of the preparation code is in the [preparation directory](./preparation/).

### All of the translated datasets:

- MMLU - https://huggingface.co/datasets/petkopetkov/mmlu-bg
- Hellaswag - https://huggingface.co/datasets/petkopetkov/hellaswag-bg
- MathQA - https://huggingface.co/datasets/petkopetkov/math_qa-bg
- Winogrande - https://huggingface.co/datasets/petkopetkov/winogrande_xl-bg
- GSM8K - https://huggingface.co/datasets/petkopetkov/gsm8k-bg
- ARC Easy - https://huggingface.co/datasets/petkopetkov/arc-easy-bg
- ARC Challenge - https://huggingface.co/datasets/petkopetkov/arc-challenge-bg
- OASST1 - https://huggingface.co/datasets/petkopetkov/oasst1_bg
- OASST2 - https://huggingface.co/datasets/petkopetkov/oasst2_bg