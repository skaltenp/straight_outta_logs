#!/usr/bin/env bash
accelerate launch sft.py --fold_name cv_split0 &&
accelerate launch sft.py --fold_name cv_split1 &&
accelerate launch sft.py --fold_name cv_split2 &&
accelerate launch sft.py --fold_name cv_split3 &&
accelerate launch sft.py --fold_name cv_split4