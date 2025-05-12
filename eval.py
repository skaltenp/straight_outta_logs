import os
import shutil
import time
from typing import Optional
from tqdm import tqdm
import logging

import numpy as np
import pandas as pd

import torch
from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer, BitsAndBytesConfig, set_seed, pipeline
from peft import AutoPeftModelForCausalLM
import argparse

def secure_mkdir(directory_path):
    if not os.path.isdir(directory_path):
        os.mkdir(directory_path)

def secure_listdir(path, rm_dirs=[".ipynb_checkpoints", ]):
    path_list = os.listdir(path)
    for rm_dir in rm_dirs:
        if rm_dir in path_list:
            path_list.remove(rm_dir)
    return path_list

def prepare_sample_text(example, tokenizer, configs, start=None, end=None):
    """Prepare the text from a sample of the dataset."""
    thread = example["event_list"]
    if start != None and end != None:
        thread = thread[start:end]
    text = f"{configs[example['file']]}{tokenizer.eos_token}\n"
    for message in thread:
        text += f"{message}{tokenizer.eos_token}\n"
    return text

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name", 
        default="skaltenp/Qwen3-1.7B-Base-global_event_logs-cv_split0", 
        help="hf model name",
        type=str
    )
    parser.add_argument(
        "--dataset_name", 
        default="skaltenp/global_event_logs",
        help="hf dataset name",
        type=str
    )
    parser.add_argument(
        "--filter", 
        default="",
        help="dataset filter",
        type=str
    )
    parser.add_argument(
        "--fold_name", 
        default="cv_split0",
        help="hf fold name",
        type=str
    )
    parser.add_argument(
        "--eotag", 
        default="<event>",
        help="event opening tag",
        type=str
    )
    parser.add_argument(
        "--ectag", 
        default="</event>",
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
        "--max_input_size", 
        default=3072,
        help="maximum input tokens",
        type=int
    )
    parser.add_argument(
        "--max_event_size", 
        default=1024,
        help="maximum event prediction tokens",
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
    model_name = args.model_name
    
    dataset_name = args.dataset_name.split("/")[-1]
    model_name = model_name.split("/")[-1]

    logs_path = "logs"
    secure_mkdir(logs_path)
    logger = logging.getLogger(__name__)
    logging.basicConfig(filename=os.path.join(logs_path, f'{model_name}.log'), encoding='utf-8', level=logging.DEBUG)
    
    dataset = load_dataset(
        path=args.dataset_name,
        name=args.fold_name,
        token=True,
        #download_mode='force_redownload',
    )

    if args.filter != "":
        dataset = dataset.filter(lambda x: x["file"] == args.filter)
    
    configs = {}
    for file in os.listdir(os.path.join("configs", args.fold_name)):
        file_path = os.path.join("configs", args.fold_name, file)
        with open(file_path, "r") as f:
            file_content = f.read()
            configs[file.replace(".xml", ".xes")] = file_content

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, 
        use_fast=True,
    )
    tokenizer.model_max_length = args.max_input_size
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.truncation_side = "left"

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    model = AutoPeftModelForCausalLM.from_pretrained(
        args.model_name, 
        quantization_config=bnb_config,
        device_map=args.device, 
        torch_dtype=torch.float16,
    )
    model.eval()

    generator = pipeline(
        "text-generation", 
        tokenizer=tokenizer, 
        model=model,
        return_full_text=False, 
        do_sample=False, 
        temperature=None, 
        top_p=None, 
        max_new_tokens=args.max_event_size,
        truncation=True,
        pad_token_id=tokenizer.eos_token_id,
    )

    target_true_path = os.path.join(results_path, f"True_{model_name}.xes")
    target_predicted_path = os.path.join(results_path, f"Pred_{model_name}.xes")
    
    shutil.copy("xes/True_BACKUP.xes", target_true_path)
    shutil.copy("xes/Pred_BACKUP.xes", target_predicted_path)
    counter = 0

    test_len = len(dataset["test"])
    breakpoint = len(dataset["test"])
    test_len = min(breakpoint, test_len)
    case_sum = 0
    event_times = []
    st_start = time.time()
    for example in tqdm(dataset["test"]):
        st = time.time()
        res = prepare_sample_text(example, tokenizer, configs)
        res = res.replace(configs[example["file"]], "")
        inp = ""
        inp += prepare_sample_text(example, tokenizer, configs, start=0, end=1)
        event_counter = 0
        event_times.append([counter, event_counter, 0])
        with open(target_predicted_path, "a") as xes_file:
            try:
                inp = inp.replace(tokenizer.bos_token, "")
            except Exception as e:
                print("Error: Could not replace bos token")
            inp = inp.replace(tokenizer.eos_token, "")
            xes_file.write(inp.replace(configs[example["file"]], ""))
        
        for i in tqdm(range(1, len(example["event_list"]))):
            event_start_time = time.time()
            inp = prepare_sample_text(example, tokenizer, configs, start=0, end=i)
            if args.print:
                print(f"##### INPUT UP TO {i-1} #####")
                print(inp)
            output = generator(
                text_inputs=inp,
                tokenizer=tokenizer,
                return_full_text=False, 
                do_sample=False, 
                temperature=None, 
                top_p=None, 
                max_new_tokens=1024, 
                truncation=True, 
                pad_token_id=tokenizer.eos_token_id, 
                stop_strings=f"{args.ectag}", 
            )[0]
            inp = output["generated_text"] + "\n"
            #inp = inp.split(f"{args.ectag}")[0] + f"{args.ectag}\n"
            if args.print:
                print(f"##### PREDICTION FOR {i} #####")
                print(inp)
                print(f"##### TRUE VALUES FOR {i} #####")
                print(res.split(f"{args.ectag}")[i].replace(f"{tokenizer.eos_token}", "") + f"{args.ectag}")
                print()
                print("-" * 25)
                print("-" * 25)
                print()
            
            inp = inp.replace(configs[example["file"]], "")
            with open(target_predicted_path, "a") as xes_file:
                xes_file.write(inp)
            event_time = time.time() - event_start_time
            event_counter += 1
            event_times.append([counter, event_counter, event_time])
            logger.info(f"Event time: {event_time}")

        if not inp.rstrip().endswith(f"{args.tctag}"):
            with open(target_predicted_path, "a") as xes_file:
                xes_file.write(f"\t{args.tctag}\n")
        with open(target_true_path, "a") as xes_file:
            try:
                res = res.replace(tokenizer.bos_token, "")
            except Exception as e:
                print("Error: Could not replace bos token")
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
        event_times_df = pd.DataFrame(event_times, columns=["Trace", "Event", "Time"])
        event_times_df.to_csv(os.path.join(logs_path, f"event_times_{model_name}.csv"), index=False)
        if counter == test_len:
            break

    with open(target_predicted_path, "a") as xes_file:
        xes_file.write(f"{args.lctag}")
    with open(target_true_path, "a") as xes_file:
        xes_file.write(f"{args.lctag}")