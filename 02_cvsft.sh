#!/usr/bin/env bash
accelerate launch --config_file gpu0.yaml sft.py --dataset_name skaltenp/sepsis_cases --fold_name cv_split0 &&
accelerate launch --config_file gpu0.yaml sft.py --dataset_name skaltenp/sepsis_cases --fold_name cv_split1 &&
accelerate launch --config_file gpu0.yaml sft.py --dataset_name skaltenp/sepsis_cases --fold_name cv_split2 &&
accelerate launch --config_file gpu0.yaml sft.py --dataset_name skaltenp/sepsis_cases --fold_name cv_split3 &&
accelerate launch --config_file gpu0.yaml sft.py --dataset_name skaltenp/sepsis_cases --fold_name cv_split4