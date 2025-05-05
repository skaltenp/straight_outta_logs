import os
from dataclasses import dataclass, field
from typing import Optional
from tqdm import tqdm

import numpy as np

import torch
from accelerate import Accelerator
from datasets import load_dataset, concatenate_datasets
from transformers import HfArgumentParser, set_seed, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig
from trl import  SFTConfig, SFTTrainer
from trl.trainer import ConstantLengthDataset
import pm4py
import wandb


def chars_token_ratio(dataset, tokenizer, configs, nb_examples=400):
    """
    Estimate the average number of characters per token in the dataset.
    """
    total_characters, total_tokens = 0, 0
    for _, example in tqdm(zip(range(nb_examples), iter(dataset)), total=nb_examples):
        text = prepare_sample_text(example, tokenizer, configs)
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


def prepare_sample_text(example, tokenizer, configs, prefix_length=5, start=None, end=None):
    """Prepare the text from a sample of the dataset."""
    thread = example["event_list"]
    if start != None and end != None:
        thread = thread[start:end]
    text = ""
    for i in range(0, len(thread), prefix_length):
        text_sample = f"{configs[example['file']]}{tokenizer.eos_token}\n"
        for message in thread[i: i + prefix_length]:
            text_sample += f"{message}{tokenizer.eos_token}\n"
        text += text_sample + "\n"
    return text

# def prepare_sample_text(example, tokenizer, remove_indent=False, start=None, end=None):
#     """Prepare the text from a sample of the dataset."""
#     thread = example["event_list"]
#     if start != None and end != None:
#         thread = thread[start:end]
#     text = ""
#     for message in thread:
#         text += f"{message}{tokenizer.eos_token}\n"
#     return text


def create_datasets(tokenizer, args, configs):
    dataset = load_dataset(
        args.dataset_name,
        name=args.fold_name,
        token=True,
        num_proc=args.num_workers,
        download_mode='force_redownload'
    )
    test_dataset = dataset["test"] # DON'T use this for validation
    train_dataset = dataset["train"].train_test_split(test_size=0.2, seed=args.random_seed)
    valid_dataset = train_dataset["test"]
    train_dataset = train_dataset["train"]

    print(train_dataset)

    chars_per_token = max(chars_token_ratio(train_dataset, tokenizer, configs), 3)
    print(f"The character to token ratio of the dataset is: {chars_per_token:.2f}")

    print(f"Size of the train set: {len(train_dataset)}. Size of the validation set: {len(valid_dataset)}")
    return train_dataset, valid_dataset


@dataclass
class ScriptArguments:
    model_name: Optional[str] = field(default="Qwen/Qwen3-0.6B-Base", metadata={"help": "the model name"})
    report_to: Optional[str] = field(default="wandb", metadata={"help": "use 'wandb' to log with wandb"})
    random_seed: Optional[int] = field(default=42, metadata={"help": "random seed for model training"})

    dataset_name: Optional[str] = field(default="skaltenp/global_event_logs", metadata={"help": "dataset name"})
    fold_name: Optional[str] = field(default="cv_split0", metadata={"help": "name of the fold"})
    use_fast_tokenizer: Optional[bool] = field(default=True, metadata={"help": "whether to use a fast tokenizer"})
    steps_factor: Optional[int] = field(default=4, metadata={"help": "the number to divide the whole epoch for eval, log, and save steps calculation"})
    logging_strategy: Optional[str] = field(default="steps", metadata={"help": "the logging strategy"})
    logging_steps: Optional[int] = field(default=1, metadata={"help": "the logging frequency"})
    save_strategy: Optional[str] = field(default="steps", metadata={"help": "the save strategy"})
    save_steps: Optional[int] = field(default=0.25, metadata={"help": "the saving frequency"})
    eval_strategy: Optional[str] = field(default="steps", metadata={"help": "the evaluation strategy"})
    eval_steps: Optional[int] = field(default=0.25, metadata={"help": "the eval frequency"})
    num_workers: Optional[int] = field(default=1, metadata={"help": "the workers for loading dataset"})
    seq_length: Optional[int] = field(default=4096, metadata={"help": "the sequence length"})
    max_steps: Optional[int] = field(default=-1, metadata={"help": "the maximum number of sgd steps"})
    num_train_epochs: Optional[int] = field(default=1, metadata={"help": "the number of train epochs"})
    per_device_train_batch_size: Optional[int] = field(default=8, metadata={"help": "the per device train batch size"})
    per_device_eval_batch_size: Optional[int] = field(default=8, metadata={"help": "the per device eval batch size"})
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

    if script_args.model_name_target == "":
        script_args.model_name_target = f"{script_args.model_name.split('/')[-1]}-{script_args.dataset_name.split('/')[-1]}-{script_args.fold_name}"

    configs = {}
    for file in os.listdir(os.path.join("configs", script_args.fold_name)):
        file_path = os.path.join("configs", script_args.fold_name, file)
        with open(file_path, "r") as f:
            file_content = f.read()
            configs[file.replace(".xml", ".xes")] = file_content

    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

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

    train_dataset, eval_dataset = create_datasets(tokenizer, script_args, configs)
    
    training_args = SFTConfig(
        output_dir=script_args.model_name_target,
        eval_strategy = script_args.eval_strategy,
        gradient_checkpointing=script_args.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant":True} if script_args.gradient_checkpointing else None,
        per_device_train_batch_size=script_args.per_device_train_batch_size,
        gradient_accumulation_steps=script_args.gradient_accumulation_steps,
        per_device_eval_batch_size=script_args.per_device_eval_batch_size,
        learning_rate=script_args.learning_rate,
        logging_strategy=script_args.logging_strategy,
        logging_steps=script_args.logging_steps,
        num_train_epochs=script_args.num_train_epochs,
        max_steps=script_args.max_steps,
        report_to=script_args.report_to,
        save_strategy=script_args.save_strategy,
        save_steps=script_args.save_steps,
        eval_steps=script_args.eval_steps,
        lr_scheduler_type=script_args.lr_scheduler_type,
        warmup_steps=script_args.num_warmup_steps,
        optim=script_args.optimizer_type,
        bf16=script_args.bf16,
        fp16=script_args.fp16,
        remove_unused_columns=False,
        ddp_find_unused_parameters=False,
        run_name=f"{script_args.model_name_target}",
        push_to_hub=script_args.push_to_hub,
        hub_strategy=script_args.hub_strategy,
        hub_private_repo=script_args.hub_private_repo,
        max_seq_length=script_args.seq_length,
        packing=True,
    )
    
    trainer = SFTTrainer(
        model=base_model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
        processing_class=tokenizer,
        formatting_func=lambda x: prepare_sample_text(x, tokenizer, configs),
        args=training_args,
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