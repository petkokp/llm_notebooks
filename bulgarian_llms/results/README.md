# Training results

Includes train and eval loss plots from Tensorboard. Each plot contains different colors because there were multiple runs to finish the whole training process (each color in a plot represents result from a particular run).

### SmolLM2-135M (batch size 128, 3 epochs)
  
All Tensorboard metrics: https://huggingface.co/petkopetkov/SmolLM2-135M-bg/tensorboard

<div style="display: flex; justify-content: space-around; align-items: center;">
  <figure style="text-align: center;">
    <img src="./images/SmolLM2-135M_plots/train_loss.svg" alt="Train Loss" style="width: 100%;">
    <figcaption><b>Train loss</b></figcaption>
  </figure>
  <figure style="text-align: center;">
    <img src="./images/SmolLM2-135M_plots/eval_loss.svg" alt="Eval Loss" style="width: 100%;">
    <figcaption><b>Eval loss</b></figcaption>
  </figure>
</div>

### Llama3.2-1B (batch size 64, 3 epochs)

All Tensorboard metrics: https://huggingface.co/petkopetkov/Llama3.2-1B-bg/tensorboard

<div style="display: flex; justify-content: space-around; align-items: center;">
  <figure style="text-align: center;">
    <img src="./images/Llama3.2-1B_plots/train_loss.svg" alt="Train Loss" style="width: 100%;">
    <figcaption><b>Train loss</b></figcaption>
  </figure>
  <figure style="text-align: center;">
    <img src="./images/Llama3.2-1B_plots/eval_loss.svg" alt="Eval Loss" style="width: 100%;">
    <figcaption><b>Eval loss</b></figcaption>
  </figure>
</div>

### Llama3.2-1B-Instruct (batch size 64, 3 epochs)

All Tensorboard metrics: https://huggingface.co/petkopetkov/Llama3.2-1B-Instruct-bg/tensorboard

<div style="display: flex; justify-content: space-around; align-items: center;">
  <figure style="text-align: center;">
    <img src="./images/Llama3.2-1B-Instruct_plots/train_loss.svg" alt="Train Loss" style="width: 100%;">
    <figcaption><b>Train loss</b></figcaption>
  </figure>
  <figure style="text-align: center;">
    <img src="./images/Llama3.2-1B-Instruct_plots/eval_loss.svg" alt="Eval Loss" style="width: 100%;">
    <figcaption><b>Eval loss</b></figcaption>
  </figure>
</div>

### Llama3.2-1B (custom tokenizer, batch size 64, 3 epochs)

All Tensorboard metrics: https://huggingface.co/petkopetkov/Llama3.2-1B-Instruct-bg-tokenizer/tensorboard

<div style="display: flex; justify-content: space-around; align-items: center;">
  <figure style="text-align: center;">
    <img src="./images/Llama3.2-1B-tokenizer_plots/train_loss.svg" alt="Train Loss" style="width: 100%;">
    <figcaption><b>Train loss</b></figcaption>
  </figure>
  <figure style="text-align: center;">
    <img src="./images/Llama3.2-1B-tokenizer_plots/eval_loss.svg" alt="Eval Loss" style="width: 100%;">
    <figcaption><b>Eval loss</b></figcaption>
  </figure>
</div>

### Gemma-2-2B (batch size 64, 3 epochs)

All Tensorboard metrics: https://huggingface.co/petkopetkov/gemma-2-2b-bg/tensorboard

<div style="display: flex; justify-content: space-around; align-items: center;">
  <figure style="text-align: center;">
    <img src="./images/gemma-2-2b_plots/train_loss.svg" alt="Train Loss" style="width: 100%;">
    <figcaption><b>Train loss</b></figcaption>
  </figure>
  <figure style="text-align: center;">
    <img src="./images/gemma-2-2b_plots/eval_loss.svg" alt="Eval Loss" style="width: 100%;">
    <figcaption><b>Eval loss</b></figcaption>
  </figure>
</div>