import os
import torch
from datasets import load_dataset, load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig
from trl import SFTTrainer
import argparse
from create_prompts import create_prompts
from datasets import Dataset, DatasetDict
import pandas as pd

def main():
    parser = argparse.ArgumentParser(description="Finetune a base instruct/chat model using (Q)LoRA and PEFT")
    parser.add_argument('tuned_model', type=str,
                        help='The name of the resulting tuned model.')
    parser.add_argument('dataset_name', type=str,
                        help='The name of the dataset to use for fine-tuning. This should be the output of the combine_checkpoints script.')
    parser.add_argument('instruction_prompt', type=str, 
                        help='An instruction message added to every prompt given to the chatbot to force it to answer in the target language. Example: "You are a generic chatbot that always answers in English."')
    parser.add_argument('--base_model', type=str, default="HuggingFaceTB/SmolLM2-1.7B-Instruct",
                        help='The base foundation model. Default is "HuggingFaceTB/SmolLM2-1.7B-Instruct".')
    parser.add_argument('--base_dataset_text_field', type=str, default="text",
                        help="The dataset's column name containing the actual text to translate. Defaults to text")
    parser.add_argument('--base_dataset_rank_field', type=str, default="rank",
                        help="The dataset's column name containing the rank of an answer given to a prompt. Defaults to rank")
    parser.add_argument('--base_dataset_id_field', type=str, default="message_id",
                        help="The dataset's column name containing the id of a text. Defaults to message_id")
    parser.add_argument('--base_dataset_parent_field', type=str, default="parent_id",
                        help="The dataset's column name containing the parent id of a text. Defaults to parent_id")
    parser.add_argument('--base_dataset_role_field', type=str, default="role",
                        help="The dataset's column name containing the role of the author of the text (eg. prompter, assistant). Defaults to role")
    parser.add_argument('--quant8', action='store_true',
                        help='Finetunes the model in 8 bits. Requires more memory than the default 4 bit.')
    parser.add_argument('--noquant', action='store_true',
                        help='Do not quantize the finetuning. Requires more memory than the default 4 bit and optional 8 bit.')
    parser.add_argument('--max_seq_length', type=int, default=512,
                        help='The maximum sequence length to use in finetuning. Should most likely line up with your base model\'s default max_seq_length. Default is 512.')
    parser.add_argument('--num_train_epochs', type=int, default=2,
                        help='Number of epochs to use. 2 is default and has been shown to work well.')
    parser.add_argument('--batch_size', type=int, default=2,
                        help='The batch size to use in finetuning. Adjust to fit in your GPU vRAM. Default is 2')
    parser.add_argument('--threads_output_name', type=str, default=None,
                        help='If specified, the threads created in this script for finetuning will also be saved to disk or HuggingFace Hub.')
    parser.add_argument('--thread_template', type=str, default="template_default.txt",
                        help='A file containing the thread template to use. Default is threads/template_fefault.txt')
    parser.add_argument('--padding', type=str, default="left",
                        help='What padding to use, can be either left or right.')
    parser.add_argument('--force_cpu', action='store_true',
                        help="Forces usage of CPU. By default GPU is taken if available.")
    
    args = parser.parse_args()
    base_model = args.base_model
    tuned_model = args.tuned_model
    dataset_name = args.dataset_name
    instruction_prompt = args.instruction_prompt
    base_dataset_text_field = args.base_dataset_text_field
    base_dataset_rank_field = args.base_dataset_rank_field
    base_dataset_id_field = args.base_dataset_id_field
    base_dataset_parent_field = args.base_dataset_parent_field
    base_dataset_role_field = args.base_dataset_role_field
    quant8 = args.quant8
    noquant = args.noquant
    max_seq_length = args.max_seq_length
    num_train_epochs = args.num_train_epochs
    per_device_train_batch_size = args.batch_size
    threads_output_name = args.threads_output_name
    thread_template_file = args.thread_template
    padding = args.padding
    force_cpu = args.force_cpu
    device = torch.device("cuda:0" if torch.cuda.is_available() and not (force_cpu) else "cpu")
    
    if 'HF_TOKEN' not in os.environ:
        print("[WARNING] Environment variable 'HF_TOKEN' is not set!")
        user_input = input("Do you want to continue? (yes/no): ").strip().lower()

        if user_input != "yes":
            print("Terminating the program.")
            exit()
    
    if os.path.isdir(dataset_name):
        dataset = load_from_disk(dataset_name)
    else:
        dataset = load_dataset(dataset_name)
    
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    with open(thread_template_file, 'r', encoding="utf8") as f:
        chat_template = f.read()
    
    prompts = {k: [] for k in dataset.keys()}
    for fold in prompts:
        print(f"Generating prompts using chat template {thread_template_file} for fold {fold}")
        templated_prompts = create_prompts(dataset[fold], tokenizer, base_dataset_rank_field, base_dataset_parent_field, base_dataset_id_field, base_dataset_text_field, base_dataset_role_field, instruction_prompt, chat_template)
        prompts[fold] = Dataset.from_pandas(pd.DataFrame(data=templated_prompts))

    prompts = DatasetDict(prompts)
    if threads_output_name is not None:
        print(f"Writing out thread prompts dataset to {threads_output_name}")
        if os.path.isdir(threads_output_name):
            prompts.save_to_disk(threads_output_name)
        else:
            prompts.push_to_hub(threads_output_name)

    if noquant:
        model = AutoModelForCausalLM.from_pretrained(base_model, device_map=device, trust_remote_code=True)
    elif quant8:
        quant_config = BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_8bit_quant_type="qat8",
            bnb_8bit_compute_dtype=getattr(torch, "float32"),
            bnb_8bit_use_double_quant=False
        )
        model = AutoModelForCausalLM.from_pretrained(base_model, quantization_config=quant_config, device_map=device, trust_remote_code=True)
    else:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=getattr(torch, "float16"),
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(base_model, quantization_config=quant_config, device_map=device, trust_remote_code=True)

    model.config.use_cache = False
    model.config.pretraining_tp = 1

    if padding == 'left':
        tokenizer.pad_token_id = 0
    else:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = padding

    peft_params = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.1,
        r=64,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules = 'all-linear',
    )

    use_fp16 = not(noquant or quant8)
    training_params = TrainingArguments(
        output_dir="./results",
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=1,
        optim="paged_adamw_32bit",
        save_steps=1000,
        logging_steps=500,
        learning_rate=2e-4,
        weight_decay=0.001,
        fp16=use_fp16,
        bf16=False,
        max_grad_norm=0.3,
        max_steps=-1,
        warmup_ratio=0.03,
        group_by_length=True,
        lr_scheduler_type="constant",
        report_to="tensorboard"
    )
    trainer = SFTTrainer(
        model=model,
        train_dataset=prompts['train'],
        peft_config=peft_params,
        dataset_text_field=base_dataset_text_field,
        max_seq_length=max_seq_length,
        tokenizer=tokenizer,
        args=training_params,
        packing=False,
    )

    torch.cuda.empty_cache()
    print(f"Starting training")
    trainer.train()

    print(f"Writing model and tokenizer out to {tuned_model}")
    if os.path.isdir(tuned_model):
        trainer.model.save_to_disk(tuned_model)
        trainer.tokenizer.save_to_disk(tuned_model)
    else:
        trainer.model.push_to_hub(tuned_model)
        trainer.tokenizer.push_to_hub(tuned_model)

if __name__ == "__main__":
    main()
