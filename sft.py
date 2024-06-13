import os
import statistics
from dataclasses import dataclass, field
from typing import Optional
from tqdm import tqdm
from datetime import datetime

import numpy as np
import pandas as pd

import torch
from accelerate import Accelerator
from datasets import load_dataset, concatenate_datasets, load_from_disk, Dataset, DatasetDict
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM, BitsAndBytesConfig, HfArgumentParser, pipeline, DataCollatorForLanguageModeling, Trainer, TrainingArguments, logging, set_seed, EarlyStoppingCallback
from peft import get_peft_model, LoraConfig, AutoPeftModelForCausalLM
from trl.trainer import ConstantLengthDataset
from trl import SFTTrainer
from huggingface_hub import login
import wandb


class ConstantLengthDataset(ConstantLengthDataset): # Fixes wrong size due to packing

    def __len__(self):
        is_infinite = self.infinite
        if is_infinite:
            self.infinite = False
        length = len(list(self.__iter__()))
        if is_infinite:
            self.infinite = True
        return length

def chars_token_ratio(dataset, tokenizer, nb_examples=400):
    """
    Estimate the average number of characters per token in the dataset.
    """
    total_characters, total_tokens = 0, 0
    for _, example in tqdm(zip(range(nb_examples), iter(dataset)), total=nb_examples):
        text = prepare_sample_text(example, tokenizer)
        #print(text)
        total_characters += len(text)
        if tokenizer.is_fast:
            total_tokens += len(tokenizer(text).tokens())
        else:
            total_tokens += len(tokenizer.tokenize(text))
    #raise
    return total_characters / total_tokens


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


def prepare_sample_text(example, tokenizer, remove_indent=False, start=None, end=None):
    """Prepare the text from a sample of the dataset."""
    thread = example["event_list"]
    if start and end:
        thread = thread[start:end]
    text = ""
    for message in thread:
        text += f"{message}{tokenizer.eos_token}\n"
    return text


def create_datasets(tokenizer, args):
    dataset = load_dataset(
        args.dataset_name,
        token=True,
        num_proc=args.num_workers,
        download_mode='force_redownload'
    )
    train_data = dataset["train"].train_test_split(train_size=0.8, shuffle=True, seed=args.random_seed)
    test_data = train_data["test"]
    train_data = train_data["train"].train_test_split(train_size=0.8, shuffle=True, seed=args.random_seed)
    valid_data = train_data["test"]
    train_data = train_data["train"]

    chars_per_token = chars_token_ratio(train_data, tokenizer)
    print(f"The character to token ratio of the dataset is: {chars_per_token:.2f}")

    train_dataset = ConstantLengthDataset(
        tokenizer,
        train_data,
        formatting_func=lambda x: prepare_sample_text(x, tokenizer),
        infinite=True,
        seq_length=args.seq_length,
        chars_per_token=chars_per_token,
    )
    valid_dataset = ConstantLengthDataset(
        tokenizer,
        valid_data,
        formatting_func=lambda x: prepare_sample_text(x, tokenizer),
        infinite=False,
        seq_length=args.seq_length,
        chars_per_token=chars_per_token,
    )
    print(f"Size of the train set: {len(train_dataset)}. Size of the validation set: {len(valid_dataset)}")
    return train_dataset, valid_dataset

def create_datasets_all(tokenizer, args):
    dataset_names = args.dataset_name.split(",")
    train_data = []
    valid_data = []
    for dataset_name in dataset_names:
        dataset = load_dataset(
            dataset_name,
            token=True,
            num_proc=args.num_workers,
            download_mode='force_redownload'
        )
        train_data = dataset["train"].train_test_split(train_size=0.8, shuffle=True, seed=args.random_seed)
        test_data = train_data["test"]
        train_data = train_data["train"].train_test_split(train_size=0.8, shuffle=True, seed=args.random_seed)
        valid_data = train_data["test"]
        train_data = train_data["train"]
        train_data.append(dataset["train"])
        valid_data.append(dataset["valid"])


    train_data = concatenate_datasets(train_data)
    valid_data = concatenate_datasets(valid_data)
    
    chars_per_token = chars_token_ratio(train_data, tokenizer)
    print(f"The character to token ratio of the dataset is: {chars_per_token:.2f}")

    train_dataset = ConstantLengthDataset(
        tokenizer,
        train_data,
        formatting_func=lambda x: prepare_sample_text(x, tokenizer),
        infinite=True,
        seq_length=args.seq_length,
        chars_per_token=chars_per_token,
    )
    valid_dataset = ConstantLengthDataset(
        tokenizer,
        valid_data,
        formatting_func=lambda x: prepare_sample_text(x, tokenizer),
        infinite=False,
        seq_length=args.seq_length,
        chars_per_token=chars_per_token,
    )
    print(f"Size of the train set: {len(train_dataset)}. Size of the validation set: {len(valid_dataset)}")
    return train_dataset, valid_dataset

@dataclass
class ScriptArguments:
    model_name: Optional[str] = field(default="meta-llama/Meta-Llama-3-8B", metadata={"help": "the model name"})
    log_with: Optional[str] = field(default="wandb", metadata={"help": "use 'wandb' to log with wandb"})
    random_seed: Optional[int] = field(default=42, metadata={"help": "random seed for model training"})
    big_model_training: Optional[bool] = field(default=False, metadata={"help": "whether to use multiple gpus for one big model"})
    use_all_datasets: Optional[bool] = field(default=False, metadata={"help": "use all datasets"})

    dataset_name: Optional[str] = field(default="skaltenp/sepsis_cases", metadata={"help": "dataset name"})
    use_fast_tokenizer: Optional[bool] = field(default=True, metadata={"help": "whether to use a fast tokenizer"})
    evaluation_strategy: Optional[str] = field(default="steps", metadata={"help": "the evaluation strategy"})
    num_workers: Optional[int] = field(default=1, metadata={"help": "the workers for loading dataset"})
    seq_length: Optional[int] = field(default=4096, metadata={"help": "the sequence length"})
    max_steps: Optional[int] = field(default=-1, metadata={"help": "the maximum number of sgd steps"})
    num_train_epochs: Optional[int] = field(default=1, metadata={"help": "the number of train epochs"})
    logging_steps: Optional[int] = field(default=1, metadata={"help": "the logging frequency"})
    save_steps: Optional[int] = field(default=-1, metadata={"help": "the saving frequency"})
    eval_steps: Optional[int] = field(default=-1, metadata={"help": "the eval frequency"})
    steps_factor: Optional[int] = field(default=4, metadata={"help": "the number to divide the whole epoch for eval, log, and save steps calculation"})
    per_device_train_batch_size: Optional[int] = field(default=1, metadata={"help": "the per device train batch size"})
    per_device_eval_batch_size: Optional[int] = field(default=1, metadata={"help": "the per device eval batch size"})
    gradient_accumulation_steps: Optional[int] = field(default=1, metadata={"help": "the gradient accumulation steps"})
    gradient_checkpointing: Optional[bool] = field(default=True, metadata={"help": "whether to use gradient checkpointing"})

    lora_alpha: Optional[float] = field(default=16, metadata={"help": "the lora alpha parameter"})
    lora_dropout: Optional[float] = field(default=0.05, metadata={"help": "the lora dropout parameter"})
    lora_r: Optional[int] = field(default=8, metadata={"help": "the lora r parameter"})
    target_modules: Optional[str] = field(default="q_proj,v_proj", metadata={"help": "peft target modules"})

    learning_rate: Optional[float] = field(default=1e-4, metadata={"help": "the learning rate"})
    lr_scheduler_type: Optional[str] = field(default="cosine", metadata={"help": "the lr scheduler type"})
    num_warmup_steps: Optional[int] = field(default=100, metadata={"help": "the number of warmup steps"})
    weight_decay: Optional[float] = field(default=0.05, metadata={"help": "the weight decay"})
    optimizer_type: Optional[str] = field(default="paged_adamw_32bit", metadata={"help": "the optimizer type"})
    bf16: Optional[bool] = field(default=False, metadata={"help": "whether to use bf16 precision"})
    fp16: Optional[bool] = field(default=True, metadata={"help": "whether to use fp16 precision"})

    output_dir: Optional[str] = field(default="./results", metadata={"help": "the output directory"})
    model_name_target: Optional[str] = field(default="", metadata={"help": "name for trained model"})
    push_to_hub: Optional[bool] = field(default=True, metadata={"help": "whether to push the model to hub"})
    hub_strategy: Optional[str] = field(default="checkpoint", metadata={"help": "the strategy for push to hub"})
    hub_private_repo: Optional[bool] = field(default=True, metadata={"help": "whether the repo shall be private"})


def main():
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]

    set_seed(script_args.random_seed)

    if script_args.log_with == "wandb":
        wandb.login(key="YOURTOKENHERE")

    if script_args.model_name_target == "":
        if script_args.use_all_datasets:
            script_args.model_name_target = f"{script_args.model_name.split('/')[-1]}-{'all_logs'}-{script_args.random_seed}"
        else:
            script_args.model_name_target = f"{script_args.model_name.split('/')[-1]}-{script_args.dataset_name.split('/')[-1]}-{script_args.random_seed}"
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    if script_args.big_model_training:
        device_map = "auto"
        print("Using big model training")
    else:
        device_map = {"": Accelerator().process_index}
    
    base_model = AutoModelForCausalLM.from_pretrained(
        script_args.model_name,
        quantization_config=bnb_config,
        device_map=device_map,
        trust_remote_code=True,
        token=True,
    )
    base_model.config.use_cache = False

    target_modules = None
    target_modules = script_args.target_modules.split(",")
    
    peft_config = LoraConfig(
        r=script_args.lora_r,
        lora_alpha=script_args.lora_alpha,
        lora_dropout=script_args.lora_dropout,
        target_modules=target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    tokenizer = AutoTokenizer.from_pretrained(
        script_args.model_name, 
        trust_remote_code=True,
        use_fast=script_args.use_fast_tokenizer
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"  # Fix weird overflow issue with fp16 training

    if script_args.use_all_datasets:
        train_dataset, eval_dataset = create_datasets_all(tokenizer, script_args)
        script_args.dataset_name = "all_logs"
    else:
        train_dataset, eval_dataset = create_datasets(tokenizer, script_args)

    num_devices = torch.cuda.device_count()
    print(f"There are {num_devices} torch devices.")
    if script_args.logging_steps == -1:
        script_args.logging_steps = round(len(train_dataset) / script_args.steps_factor / num_devices / script_args.per_device_train_batch_size)
        print(f"Logging every {script_args.logging_steps} steps.")

    if script_args.save_steps == -1:
        script_args.save_steps = round(len(train_dataset) / script_args.steps_factor / num_devices / script_args.per_device_train_batch_size)
        print(f"Saving every {script_args.save_steps} steps.")

    if script_args.eval_steps == -1:
        script_args.eval_steps = round(len(train_dataset) / script_args.steps_factor / num_devices / script_args.per_device_train_batch_size)
        print(f"Evaluating every {script_args.eval_steps} steps.")
    
    training_args = TrainingArguments(
        output_dir=script_args.model_name_target,
        evaluation_strategy = script_args.evaluation_strategy,
        gradient_checkpointing=script_args.gradient_checkpointing,
        per_device_train_batch_size=script_args.per_device_train_batch_size,
        gradient_accumulation_steps=script_args.gradient_accumulation_steps,
        per_device_eval_batch_size=script_args.per_device_eval_batch_size,
        learning_rate=script_args.learning_rate,
        logging_steps=script_args.logging_steps,
        num_train_epochs=script_args.num_train_epochs,
        max_steps=script_args.max_steps,
        report_to=script_args.log_with,
        save_steps=script_args.save_steps,
        eval_steps=script_args.eval_steps,
        lr_scheduler_type=script_args.lr_scheduler_type,
        warmup_steps=script_args.num_warmup_steps,
        optim=script_args.optimizer_type,
        bf16=script_args.bf16,
        fp16=script_args.fp16,
        remove_unused_columns=False,
        ddp_find_unused_parameters=False,
        run_name=f"sft-{script_args.model_name_target}",
        push_to_hub=script_args.push_to_hub,
        hub_strategy=script_args.hub_strategy,
        hub_private_repo=script_args.hub_private_repo,
        #greater_is_better=False,
        #metric_for_best_model="eval_loss",
        #load_best_model_at_end=True,
    )
    
    trainer = SFTTrainer(
        model=base_model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
        packing=True,
        max_seq_length=script_args.seq_length,
        tokenizer=tokenizer,
        args=training_args,
        #callbacks=[EarlyStoppingCallback(), ],
    )
    trainer.train()
    
    if script_args.push_to_hub:
        trainer.push_to_hub(script_args.model_name)
    else:
        trainer.save_model(script_args.output_dir)
        output_dir = os.path.join(script_args.output_dir, "final_checkpoint")
        trainer.model.save_pretrained(output_dir)

if __name__ == "__main__":
    main()