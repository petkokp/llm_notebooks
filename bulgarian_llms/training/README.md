# Training

Before the training process begins, quantized versions of the models are loaded using the [unsloth](https://github.com/unslothai/unsloth) library because all of the experiments are performed on a local device which lacks computational resources for finetuning the whole models.

After that, the datasets preparation is performed. All of the datasets are combined to a single dataset which contains ~227,000 train samples. Different dataset mixing strategies are supported.

The training is done using the Supervised Fine-tuning (SFT) trainer from the [trl](https://github.com/huggingface/trl) library. All the models are finetuned for 3 epochs. For some of them batch size of 128 is used and for others 64 (depending on the model's size).

### How to train:

Prepare the datasets for training:

```
python ./datasets/preparation/process_datasets.py
```

Start the training process:

```
python ./training/llm_trainer.py
```

Results from the training (train and eval loss plots) can be [here](../results/README.md).

