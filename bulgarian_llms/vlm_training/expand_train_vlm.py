from unsloth import FastVisionModel
from transformers import AutoTokenizer
from datasets import load_dataset, interleave_datasets, concatenate_datasets
from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTTrainer, SFTConfig
import os
import torch
import numpy as np
from utils import UnslothView, dynamic_steps

os.environ["TOKENIZERS_PARALLELISM"] = "False"
os.environ["WANDB_PROJECT"] = "bg_vlm"
# os.environ["WANDB_RESUME"] = "must"
# os.environ["WANDB_RUN_ID"] = "t7wlhf4o"

tok_dir = "../tokenizers/Qwen3-VL-2B-Instruct_relearn" # "unsloth/Qwen3-VL-2B-Instruct" 

TOKENIZER_STRATEGY = "relearn" # "relearn" if "relearn" in tok_dir else "expand" ### default

DATASET_STRATEGY = "concatenate" # "interleave"

print(f"Using {TOKENIZER_STRATEGY} tokenizer")

print(f"Using {DATASET_STRATEGY} dataset strategy")

model, processor = FastVisionModel.from_pretrained(
    "unsloth/Qwen3-VL-2B-Instruct",
    full_finetuning=True,
    load_in_4bit=False,
)

base_tok = getattr(processor, "tokenizer", processor)
custom_tok = AutoTokenizer.from_pretrained(tok_dir, trust_remote_code=True, use_fast=True)
processor.tokenizer = custom_tok

if TOKENIZER_STRATEGY == "expand" and len(custom_tok) > model.get_input_embeddings().weight.shape[0]:
    n_new = len(custom_tok) - model.get_input_embeddings().weight.shape[0]
    print("new tokens: ", n_new)
    model.resize_token_embeddings(len(custom_tok))
    with torch.no_grad():
        W = model.get_input_embeddings().weight
        W[-n_new:] = W[:-n_new].mean(dim=0, keepdim=True)
    if getattr(model, "tie_weights", None):
        model.tie_weights()

# ValueError: Mismatch in `image` token count between text and `input_ids`. Got ids=[0] and text=[256]. Likely due to `truncation='max_length'`. Please disable truncation or increase `max_length`.
# if TOKENIZER_STRATEGY == "relearn":
#     raise Exception("TODO - 'relearn' tokenizer does not work yet!")
    # old_v, new_v = base_tok.get_vocab(), custom_tok.get_vocab()
    # W = model.get_input_embeddings().weight.data
    # d = W.size(1)
    # newW = W.new_empty((len(custom_tok), d))
    # with torch.no_grad():
    #     newW[:] = W.mean(dim=0, keepdim=True)
    #     for tok, nid in new_v.items():
    #         oid = old_v.get(tok)
    #         if oid is not None and oid < W.size(0):
    #             newW[nid] = W[oid]
    # model.resize_token_embeddings(len(custom_tok))
    # model.get_input_embeddings().weight.data.copy_(newW)
    # if getattr(model, "tie_weights", None):
    #     model.tie_weights()

SEEDS = 3407
SUBSETS = ["vsr", "vqarad", "indoor_qa", "geomverse", "geo3k", "face_emotion", "diagram_image_to_text", "chartqa", "chart2text", "ai2d_merged", "a_okvqa", "CoSyn_400k_graphic", "CoSyn_400k_chemical"]

# EXPERIMENT_NAME = f"{TOKENIZER_STRATEGY}_tok_interleaved_ds"
EXPERIMENT_NAME = f"{TOKENIZER_STRATEGY}_tok_{DATASET_STRATEGY}_ds"

if DATASET_STRATEGY == "concatenate":
    train_parts = []

    for name in SUBSETS:
        ds = load_dataset("petkopetkov/FineVision-bg", name=name, split="train")
        train_parts.append(ds)

    raw_train = concatenate_datasets(train_parts).shuffle(seed=SEEDS)
    raw_val = raw_train.train_test_split(test_size=0.1, seed=SEEDS)["test"]
elif DATASET_STRATEGY == "interleave":
    trains, vals, sizes = [], [], []
    for name in SUBSETS:
        ds = load_dataset("petkopetkov/FineVision-bg", name=name, split="train")
        split = ds.train_test_split(test_size=0.1, seed=SEEDS)
        trains.append(split["train"])
        vals.append(split["test"])
        sizes.append(len(split["train"]))

        # Mix ratios:
        #   alpha = 1.0  -> proportional to size (natural)
        #   alpha = 0.0  -> equal probability for all subsets (balanced)
        #   0<alpha<1    -> in-between
        alpha = 0.3
        weights = np.array(sizes, dtype=float) ** alpha
        probs = (weights / weights.sum()).tolist()

        raw_train = interleave_datasets(
            datasets=trains, probabilities=probs, seed=SEEDS, stopping_strategy="first_exhausted"
        ).shuffle(seed=SEEDS)

        raw_val = interleave_datasets(
            datasets=vals, probabilities=probs, seed=SEEDS, stopping_strategy="first_exhausted"
        ).shuffle(seed=SEEDS)
else:
    raise Exception(f"Dataset strategy '{DATASET_STRATEGY}' is not supported")

train_ds = UnslothView(raw_train)
eval_ds  = UnslothView(raw_val)

FastVisionModel.for_training(model)

num_train_epochs = 1
gradient_accumulation_steps = 1
batch_size = 1

config = dynamic_steps(
    n_examples=len(train_ds),
    per_device_train_batch_size=batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    num_train_epochs=num_train_epochs,
)

trainer = SFTTrainer(
    model=model,
    tokenizer=processor,
    data_collator=UnslothVisionDataCollator(model, processor, max_seq_length=100_000),
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    args=SFTConfig(
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=2e-4,
        optim="adamw_8bit",
        weight_decay=0.01,
        seed=3407,
        output_dir=f"./checkpoints/{EXPERIMENT_NAME}",

        remove_unused_columns=False,
        dataset_text_field="",
        dataset_kwargs={"skip_prepare_dataset": True},

        max_length=100_000,
        report_to="wandb",
        run_name=EXPERIMENT_NAME,
        load_best_model_at_end=True,
        save_total_limit=1,
        num_train_epochs=num_train_epochs,
        logging_strategy="steps",
        eval_strategy="steps",
        save_strategy="steps",
        logging_steps=config["logging_steps"],
        eval_steps=config["eval_steps"],
        save_steps=config["save_steps"],
        warmup_steps=config["warmup_steps"],
    ),
)

trainer.train()

print(f"Successfully finished training for {EXPERIMENT_NAME}")
