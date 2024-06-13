#!/usr/bin/env bash
# 534895718, 199900595, 862061404, 787846414, 996406378
accelerate launch sft.py --random_seed 534895718 --dataset_name skaltenp/bpi13_closed_problem &&
accelerate launch sft.py --random_seed 199900595 --dataset_name skaltenp/bpi13_closed_problem &&
accelerate launch sft.py --random_seed 862061404 --dataset_name skaltenp/bpi13_closed_problem &&
accelerate launch sft.py --random_seed 787846414 --dataset_name skaltenp/bpi13_closed_problem &&
accelerate launch sft.py --random_seed 996406378 --dataset_name skaltenp/bpi13_closed_problem &&
accelerate launch sft.py --random_seed 534895718 --dataset_name skaltenp/bpi13_incidents &&
accelerate launch sft.py --random_seed 199900595 --dataset_name skaltenp/bpi13_incidents &&
accelerate launch sft.py --random_seed 862061404 --dataset_name skaltenp/bpi13_incidents &&
accelerate launch sft.py --random_seed 787846414 --dataset_name skaltenp/bpi13_incidents &&
accelerate launch sft.py --random_seed 996406378 --dataset_name skaltenp/bpi13_incidents &&
accelerate launch sft.py --random_seed 534895718 --dataset_name skaltenp/sepsis_cases &&
accelerate launch sft.py --random_seed 199900595 --dataset_name skaltenp/sepsis_cases &&
accelerate launch sft.py --random_seed 862061404 --dataset_name skaltenp/sepsis_cases &&
accelerate launch sft.py --random_seed 787846414 --dataset_name skaltenp/sepsis_cases &&
accelerate launch sft.py --random_seed 996406378 --dataset_name skaltenp/sepsis_cases &&
accelerate launch sft.py --random_seed 534895718 --dataset_name skaltenp/helpdesk &&
accelerate launch sft.py --random_seed 199900595 --dataset_name skaltenp/helpdesk &&
accelerate launch sft.py --random_seed 862061404 --dataset_name skaltenp/helpdesk &&
accelerate launch sft.py --random_seed 787846414 --dataset_name skaltenp/helpdesk &&
accelerate launch sft.py --random_seed 996406378 --dataset_name skaltenp/helpdesk &&
accelerate launch sft.py --random_seed 534895718 --dataset_name skaltenp/bpi12 &&
accelerate launch sft.py --random_seed 199900595 --dataset_name skaltenp/bpi12 &&
accelerate launch sft.py --random_seed 862061404 --dataset_name skaltenp/bpi12 &&
accelerate launch sft.py --random_seed 787846414 --dataset_name skaltenp/bpi12 &&
accelerate launch sft.py --random_seed 996406378 --dataset_name skaltenp/bpi12