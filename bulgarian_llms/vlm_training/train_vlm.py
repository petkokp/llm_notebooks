from unsloth import FastVisionModel
from datasets import load_dataset, interleave_datasets, concatenate_datasets
from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTTrainer, SFTConfig
import os
import torch
import numpy as np
from utils import UnslothView, dynamic_steps

os.environ["TOKENIZERS_PARALLELISM"] = "False"
os.environ["WANDB_PROJECT"] = "bg_vlm"

tok_dir = "unsloth/Qwen3-VL-2B-Instruct"

TOKENIZER_STRATEGY = "default"
DATASET_STRATEGY = "interleave" # "concatenate"

print(f"Using {TOKENIZER_STRATEGY} tokenizer")
print(f"Using {DATASET_STRATEGY} dataset strategy")

model, processor = FastVisionModel.from_pretrained(
    "unsloth/Qwen3-VL-2B-Instruct",
    load_in_4bit=False,
    torch_dtype=torch.bfloat16,
)

model = FastVisionModel.get_peft_model(
    model,
    finetune_vision_layers     = True,
    finetune_language_layers   = True,
    finetune_attention_modules = True,
    finetune_mlp_modules       = True,
    r = 64,
    lora_alpha = 128,
    lora_dropout = 0.05,
    bias = "none",
    random_state = 3407,
    use_rslora = False,
    loftq_config = None,
)

tok = processor.tokenizer
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

# ensure consistent image preprocessing
ip = processor.image_processor
ip.size = {"shortest_edge": 512, "longest_edge": 512}
ip.crop_size = {"height": 512, "width": 512}
ip.do_resize = True
ip.do_center_crop = True

# Optional but safe for training
model.config.use_cache = False

SEEDS = 3407
SUBSETS = [
    "vsr","vqarad","indoor_qa","geomverse","geo3k","face_emotion",
    "diagram_image_to_text","chartqa","chart2text","ai2d_merged",
    "a_okvqa","CoSyn_400k_graphic","CoSyn_400k_chemical"
]

EXPERIMENT_NAME = f"lora16_{TOKENIZER_STRATEGY}_tok_{DATASET_STRATEGY}_ds"

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
    # compute mixing after collecting all shards
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
gradient_accumulation_steps = 8
batch_size = 4

config = dynamic_steps(
    n_examples=len(train_ds),
    per_device_train_batch_size=batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    num_train_epochs=num_train_epochs,
)

trainer = SFTTrainer(
    model=model,
    tokenizer=processor,
    data_collator=UnslothVisionDataCollator(model, processor, train_on_responses_only=True, instruction_part="<|im_start|>user", response_part="<|im_start|>assistant"),
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    args=SFTConfig(
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=5e-5,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        max_grad_norm=1.0,
        weight_decay=0.0,
        seed=3407,
        output_dir=f"./checkpoints/{EXPERIMENT_NAME}",
        bf16=True,
        remove_unused_columns=False,
        dataset_text_field="",
        dataset_kwargs={"skip_prepare_dataset": True},
        max_seq_length=1024,
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

model.save_pretrained_merged(f"./checkpoints/merged_{EXPERIMENT_NAME}", tok)

print(f"Successfully finished training for {EXPERIMENT_NAME}")
