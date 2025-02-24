# Straight outta Logs: Can LLMs overcome preprocessing in Next Event Prediction?

## Installation

Install torch locally with cuda from [here](https://pytorch.org/get-started/locally/).

```
pip install datasets transformers accelerate bitsandbytes peft trl flash_attn wandb pm4py
```

## Add Information

Add your huggingface token (can be found [here](https://huggingface.co/docs/hub/en/security-tokens)) 
and your wandb token (can be found [here](https://docs.wandb.ai/quickstart)) to the Jupyter-Notebooks and scripts.

## Usage

### 1 Prepare Dataset (Optional)
Follow the steps in _01_data_prep.ipynb_ to prepare and push your own datasets, or proceed with the open datasets provided by HuggingFace user _skaltenp_.

### 3 Run Fine-Tuning (more of Domain-Adaptive Pretraining)
Run
```
sh 02_cvsft.sh
```

### 4 Run Evaluation
Run
```
sh 03_cveval.sh
```

### 5 Calculate Measures, Errors and Time
Run all cells in _04_cvmeasures.ipynb_, then checkout the logs folder, for time calculation.

### Other Information
If you have problems with running the scripts or any other code-related questions contact [sascha.kaltenpoth@uni-paderborn.de](mailto:sascha.kaltenpoth@uni-paderborn.de).