# Straight outta Logs: Can LLMs overcome preprocessing in Next Event Prediction?

## 1 Installation

Install torch locally with cuda from [here](https://pytorch.org/get-started/locally/).

```
pip install datasets transformers accelerate bitsandbytes peft trl flash_attn wandb pm4py
```

## 2 Add Information

Add your huggingface token (can be found [here](https://huggingface.co/docs/hub/en/security-tokens)) 
and your wandb token (can be found [here](https://docs.wandb.ai/quickstart)) to the Jupyter-Notebooks and scripts.

## 3 Generate your datasets

Run all cells in _01_data_prep.ipynb_

## 4 Run Fine-Tuning (more of Domain-Adaptive Pretraining)
Run
```
sh 02_cvsft.sh
```
wait for 72 hours (or split the scripts into multiple processes if you have multiple GPUs)

## 5 Run Evaluation
Run
```
sh 03_cveval.sh
```
and wait for 541 hours (or split the scripts into multiple processes if you have multiple GPUs)

## 6 Calculate Measures, Errors and Time
Run all cells in _04_cvmeasures.ipynb_, then checkout the logs folder, for time calculation.

## Other Information
If you have problems with running the scripts contact [sascha.kaltenpoth@uni-paderborn.de](mailto:sascha.kaltenpoth@uni-paderborn.de).
There is a logs_BACKUP and results_backup folder for you to prevent original logs and results beeing overwritten.
The wandb_sft_tracking.csv contains the tracking of the training runs.
If you want to only get the measurements just start with the step 6.