# Evaluate finetuned models

For evaluating the finetuned models, the [Language Model Evaluation Harness library](https://github.com/EleutherAI/lm-evaluation-harness) is used.
It can be instealled with:

```
pip install git+https://github.com/EleutherAI/lm-evaluation-harness
```

The models can be evaluated on the following tasks (Bulgarian language):
- [MMLU](./tasks/mmlu_bg/mmlu_bg.yaml)
- [Hellaswag](./tasks/hellaswag_bg/hellaswag_bg.yaml)
- [MathQA](./tasks/mathqa_bg/mathqa_bg.yaml)
- [Winogrande](./tasks/winogrande_bg/winogrande_bg.yaml)
- [GSM8K](./tasks/gsm8k_bg/gsm8k_bg.yaml)
- [ARC Easy](./tasks/arc_easy_bg/arc_easy_bg.yaml)
- [ARC Challenge](./tasks/arc_challenge_bg/arc_challenge_bg.yaml)

To evaluate a model (in this case SmolLM2-360M-bg) on all of the available tasks:

```
lm_eval \
  --model hf \
  --model_args pretrained=petkopetkov/SmolLM2-360M-bg \
  --tasks \
    mmlu_bg,hellaswag_bg,mathqa_bg,winogrande_bg,gsm8k_bg,arc_easy_bg,arc_challenge_bg \
  --device cuda:0 \
  --batch_size auto:4 \
  --include_path ./evaluation/tasks \
  --output_path ./evaluation/results
```