import json
import torch
from datasets import Dataset, interleave_datasets
from transformers import (
    TrainingArguments,
    AutoTokenizer,
)
from unsloth import FastLanguageModel, is_bfloat16_supported
from trl import SFTTrainer
import os
from typing import Dict, List
import random
import re

def format_for_sft(example):
    return {
        "text": f"### Инструкция: {example['instruction']}\n###{example['input']}\n### Отговор: {example['output']}\n"
    }

class LLMTrainer:
    def __init__(
        self,
        model_name: str = "unsloth/SmolLM2-360M",
        output_dir: str = "smollm2-360M_finetuned_model",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        custom_tokenizer_path = None,
    ):
        self.model_name = model_name
        # self.data_dir = data_dir
        self.output_dir = output_dir
        self.device = device
        
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=2048,
            dtype=None,
            load_in_4bit=True,
            attn_implementation="flash_attention_2",
        )
        
        if custom_tokenizer_path:
            self.tokenizer = AutoTokenizer.from_pretrained(custom_tokenizer_path)
        
        self.model = FastLanguageModel.get_peft_model(
            self.model,
            r=16,
            lora_alpha=16,
            lora_dropout=0,
            target_modules=["q_proj", "k_proj", "v_proj", "up_proj", "down_proj", "o_proj", "gate_proj"],
            use_rslora=True,
            use_gradient_checkpointing="unsloth",
            random_state = 32,
            loftq_config = None,
        )
        
    def load_dataset_split(self, dataset_name: str, split: str) -> Dataset:
        file_path = os.path.join(f"{dataset_name}_processed.json")
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            data = data[split]
            
        # add dataset source to each example
        for item in data:
            item['source_dataset'] = dataset_name
            
        return Dataset.from_list(data)
    
    def load_and_mix_datasets(self, dataset_names: List[str], split: str, mixing_strategy: str = "proportional") -> Dataset:
        datasets = {}
        for name in dataset_names:
            # handle MMLU-specific split names
            dataset_split = split
            if name == "mmlu":
                if split == "train":
                    dataset_split = "auxiliary_train"
                elif split == "validation":
                    dataset_split = "dev"
            
            datasets[name] = self.load_dataset_split(name, dataset_split)

        
        if mixing_strategy == "equal":
            # find the smallest dataset size
            min_size = min(len(dataset) for dataset in datasets.values())
            
            # subsample larger datasets to match the smallest
            balanced_datasets = []
            for name, dataset in datasets.items():
                if len(dataset) > min_size:
                    indices = random.sample(range(len(dataset)), min_size)
                    balanced_datasets.append(dataset.select(indices))
                else:
                    balanced_datasets.append(dataset)
            
            final_dataset = interleave_datasets(
                balanced_datasets,
                probabilities=[1/len(datasets)] * len(datasets),
                stopping_strategy="first_exhausted"
            )
            
        elif mixing_strategy == "proportional":
            # calculate total size
            total_size = sum(len(dataset) for dataset in datasets.values())
            
            # calculate mixing probabilities based on dataset sizes
            probabilities = [len(dataset)/total_size for dataset in datasets.values()]
            
            final_dataset = interleave_datasets(
                list(datasets.values()),
                probabilities=probabilities,
                stopping_strategy="all_exhausted"
            )
            
        elif mixing_strategy == "custom":
            mixing_weights = {
                "mmlu": 0.35,
                "winogrande": 0.15,
                "hellaswag": 0.15,
                "mathqa": 0.18,
                "gsm8k": 0.18,
                "arc-easy": 0.02,
                "arc-challenge": 0.02
            }
            
            probabilities = [mixing_weights[name] for name in dataset_names]
            final_dataset = interleave_datasets(
                list(datasets.values()),
                probabilities=probabilities,
                stopping_strategy="all_exhausted"
            )
        
        print(f"Loaded {split} split with {len(final_dataset)} examples")
        return final_dataset
    
    def train(
        self,
        dataset_names: List[str],
        mixing_strategy: str = "proportional",
        num_epochs: int = 2,
        batch_size: int = 4,
        gradient_accumulation_steps: int = 4,
        learning_rate: float = 3e-4,
        warmup_steps: float = 10,
        max_steps: int = -1,
        resume_from_checkpoint: str = None
    ):
        # load and mix datasets for each split
        train_dataset = self.load_and_mix_datasets(dataset_names, "train", mixing_strategy)
        
        test_dataset_names = list(filter(lambda x : x != 'hellaswag', dataset_names)) # "hellaswag" doesn't have test set
        test_dataset = self.load_and_mix_datasets(test_dataset_names, "test", mixing_strategy)
        
        validation_dataset_names = list(filter(lambda x : x != 'gsm8k', dataset_names)) # "gsm8k" doesn't have validation set
        val_dataset = self.load_and_mix_datasets(validation_dataset_names, "validation", mixing_strategy)
        
        train_dataset = train_dataset.map(format_for_sft)
        val_dataset = val_dataset.map(format_for_sft)
        test_dataset = test_dataset.map(format_for_sft)
            
        self.trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            dataset_text_field="text",
            max_seq_length=2048,
            args=TrainingArguments(
                output_dir=self.output_dir,
                num_train_epochs=num_epochs,
                per_device_train_batch_size=batch_size,
                gradient_accumulation_steps=gradient_accumulation_steps,
                learning_rate=learning_rate,
                warmup_steps=warmup_steps,
                max_steps=max_steps,
                fp16=not is_bfloat16_supported(),
                bf16=is_bfloat16_supported(),
                logging_steps=5,
                eval_strategy="steps",
                eval_steps=25,
                save_strategy="steps",
                save_steps=25,
                load_best_model_at_end=True,
                max_grad_norm=1.0,
                weight_decay=0.01,
                optim="adamw_8bit",
                lr_scheduler_type="linear",
                metric_for_best_model="eval_loss",
                report_to="tensorboard",
            )
        )
        
        self.trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        
        self.trainer.save_model(self.output_dir)
        self.tokenizer.save_pretrained(self.output_dir)
        
        print(f"\nSaved the trained model to {self.output_dir}")

def main():
    trainer = LLMTrainer()
    
    dataset_names = ["mmlu", "winogrande", "hellaswag", "mathqa", "gsm8k", "arc_easy", "arc_challenge"]
    
    checkpoint_dir = trainer.output_dir
    latest_checkpoint = None
    
    if os.path.exists(checkpoint_dir):
        checkpoint_pattern = re.compile(r"checkpoint-(\d+)")
        checkpoints = [
            os.path.join(checkpoint_dir, d)
            for d in os.listdir(checkpoint_dir)
            if os.path.isdir(os.path.join(checkpoint_dir, d)) and checkpoint_pattern.match(d)
        ]
        
        if checkpoints:
            checkpoints = sorted(
                checkpoints,
                key=lambda x: int(checkpoint_pattern.search(os.path.basename(x)).group(1))
            )
            latest_checkpoint = checkpoints[-1]
    
    trainer.model.print_trainable_parameters()
    
    trainer.train(
        dataset_names=dataset_names,
        mixing_strategy="proportional",  # Can be "equal", "proportional", or "custom"
        num_epochs=3,
        batch_size=128,
        gradient_accumulation_steps=8,
        learning_rate=3e-4,
        warmup_steps=2,
        resume_from_checkpoint=latest_checkpoint,
    )

if __name__ == "__main__":
    main()