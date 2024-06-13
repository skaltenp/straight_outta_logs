import os
import shutil
import statistics
import time
from dataclasses import dataclass, field
from typing import Optional
from tqdm import tqdm
import logging

import numpy as np
import pandas as pd

import torch
from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer, BitsAndBytesConfig, set_seed
from peft import AutoPeftModelForCausalLM
from huggingface_hub import login
import wandb
import argparse
import pm4py
login(token="YOURTOKENHERE", add_to_git_credential=True)

def secure_mkdir(directory_path):
    if not os.path.isdir(directory_path):
        os.mkdir(directory_path)

def secure_listdir(path, rm_dirs=[".ipynb_checkpoints", ]):
    path_list = os.listdir(path)
    for rm_dir in rm_dirs:
        if rm_dir in path_list:
            path_list.remove(rm_dir)
    return path_list

def prepare_sample_text(example, tokenizer, remove_indent=False, start=None, end=None, pred=False):
    """Prepare the text from a sample of the dataset."""
    thread = example["event_list"]
    if start != None and end != None:
        thread = thread[start:end]
    text = ""
    for message in thread:
        text += f"{message}{tokenizer.eos_token}\n"
    return text

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path", 
        default="", 
        help="hf model path",
        type=str
    )
    parser.add_argument(
        "--dataset_path", 
        default="",
        help="hf dataset path",
        type=str
    )
    parser.add_argument(
        "--eotag", 
        default="<event>",
        help="event opening tag",
        type=str
    )
    parser.add_argument(
        "--tctag", 
        default="</trace>",
        help="trace closing tag",
        type=str
    )
    parser.add_argument(
        "--lctag", 
        default="</log>",
        help="log closing tag",
        type=str
    )
    parser.add_argument(
        "--random_seed", 
        default=42,
        help="random seed",
        type=int
    )
    parser.add_argument(
        "--print",
        action='store_true'
    )
    parser.add_argument(
        "--device", 
        default="cuda:0",
        help="device", 
        type=str
    )
    args = parser.parse_args()

    results_path = "results"
    secure_mkdir(results_path)
    
    set_seed(args.random_seed)
    model_path = args.model_path
    if "Meta-Llama-3-8B-bpi12-199900595" in model_path or "Meta-Llama-3-8B-bpi12-862061404" in model_path:
        pass
    else:    
        dataset_path = args.dataset_path
        model_name = model_path.split("/")[-1]
    
        logs_path = "logs"
        secure_mkdir(logs_path)
        logger = logging.getLogger(__name__)
        logging.basicConfig(filename=os.path.join(logs_path, f'{model_name}.log'), encoding='utf-8', level=logging.DEBUG)
        
        dataset = load_dataset(dataset_path)
        train_data = dataset["train"].train_test_split(train_size=0.8, shuffle=True, seed=args.random_seed)
        test_data = train_data["test"]
        train_data = train_data["train"].train_test_split(train_size=0.8, shuffle=True, seed=args.random_seed)
        valid_data = train_data["test"]
        train_data = train_data["train"]
        dataset = DatasetDict(
            {
                "train": train_data,
                "valid": valid_data,
                "test": test_data
            }
        )
    
        tokenizer = AutoTokenizer.from_pretrained(
            model_path, 
            use_fast=True, 
        )
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.truncation_side = "left"
    
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        model = AutoPeftModelForCausalLM.from_pretrained(
            model_path, 
            quantization_config=bnb_config,
            device_map=args.device, 
            torch_dtype=torch.float16,
        )
        model.eval()
    
        target_true_path = os.path.join(results_path, f"True_{model_name}.xes")
        target_predicted_path = os.path.join(results_path, f"Pred_{model_name}.xes")
        
        shutil.copy("xes/True_BACKUP.xes", target_true_path)
        shutil.copy("xes/Pred_BACKUP.xes", target_predicted_path)
        counter = 0
    
        test_len = len(dataset["test"])
        breakpoint = len(dataset["test"])
        test_len = min(breakpoint, test_len)
        case_sum = 0
        st_start = time.time()
        for example in dataset["test"]:
            st = time.time()
            res = prepare_sample_text(example, tokenizer)
            inp = ""
            inp += prepare_sample_text(example, tokenizer, remove_indent=False, start=0, end=1, pred=True)
            with open(target_predicted_path, "a") as xes_file:
                inp = inp.replace(tokenizer.bos_token, "")
                inp = inp.replace(tokenizer.eos_token, "")
                xes_file.write(inp)
            for i in range(1, len(example["event_list"])):
                inp = prepare_sample_text(example, tokenizer, remove_indent=False, start=0, end=i, pred=True)
                if args.print:
                    print("##### INPUT #####")
                    print(i)
                    print(inp)
                    print("#" * 25)
                with torch.no_grad():
                    st_only_gen = time.time()
                    inputs = tokenizer(
                        inp, 
                        return_tensors="pt", 
                        max_length=4096, 
                        truncation=True
                    ).to(args.device)
                    outputs = model.generate(
                        **inputs, 
                        max_new_tokens=1024,
                        do_sample=True, 
                        pad_token_id=tokenizer.eos_token_id
                    )
                    logger.info(f"Generation only: {time.time() - st_only_gen}")
                    pred = tokenizer.batch_decode(outputs.detach().cpu().numpy())[0]
                    inp = pred.replace(tokenizer.bos_token, "")
                    inp = inp.replace(tokenizer.eos_token, "")
                    if args.print:
                        print("##### PREDICTION #####")
                        print(inp)
                        print("#" * 25)
                        print("##### TRUE VALUES #####")
                        print(res)
                        print("#" * 25)
                        print("#" * 25)
                        print("#" * 25)
                        print("#" * 25)
                        print("#" * 25)
                        print()
                    with open(target_predicted_path, "a") as xes_file:
                        inp = f"\t\t{args.eotag}" + inp.split(f"{args.eotag}")[-1]
                        inp = inp.replace(tokenizer.bos_token, "")
                        inp = inp.replace(tokenizer.eos_token, "")
                        inp = inp.replace("</trace>", "")
                        inp = inp + "\n"
                        xes_file.write(inp)
            if not inp.rstrip().endswith(f"{args.tctag}"):
                with open(target_predicted_path, "a") as xes_file:
                    xes_file.write(f"\t{args.tctag}\n")
            with open(target_true_path, "a") as xes_file:
                res = res.replace(tokenizer.bos_token, "")
                res = res.replace(tokenizer.eos_token, "")
                xes_file.write(res)
            counter += 1
            case_time = time.time() - st
            case_sum += case_time
            case_avg = case_sum / counter
            time_log = f"Example {counter} processed in {case_time} seconds."
            time_log += f" Example on average in : {case_avg} seconds."
            time_log += f" Time elapsed: {time.time() - st_start}."
            time_log += f" Time remaining (estim.): {round((test_len - counter) * case_avg) / 60}"
            logger.info(time_log)
            if counter == test_len:
                break
        with open(target_predicted_path, "a") as xes_file:
            xes_file.write(f"{args.lctag}")
        with open(target_true_path, "a") as xes_file:
            xes_file.write(f"{args.lctag}")