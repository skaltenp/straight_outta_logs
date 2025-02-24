#!/usr/bin/env bash
python eval.py --dataset_name skaltenp/sepsis_cases --fold_name cv_split0 --model_name skaltenp/Meta-Llama-3-8B-sepsis_cases-cv_split0 &&
python eval.py --dataset_name skaltenp/sepsis_cases --fold_name cv_split1 --model_name skaltenp/Meta-Llama-3-8B-sepsis_cases-cv_split1 &&
python eval.py --dataset_name skaltenp/sepsis_cases --fold_name cv_split2 --model_name skaltenp/Meta-Llama-3-8B-sepsis_cases-cv_split2 &&
python eval.py --dataset_name skaltenp/sepsis_cases --fold_name cv_split3 --model_name skaltenp/Meta-Llama-3-8B-sepsis_cases-cv_split3 &&
python eval.py --dataset_name skaltenp/sepsis_cases --fold_name cv_split4 --model_name skaltenp/Meta-Llama-3-8B-sepsis_cases-cv_split4