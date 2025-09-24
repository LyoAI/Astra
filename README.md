## 🎯Quick Start

### ⚙️Install dependencies

```sh
# step 1: create a virtual environment
conda create -n astra python=3.9

# step 2: activate the virtual environment
conda activate astra

# step 3: install dependencies from requirements.txt
pip install -r requirements.txt
```

### 📦 Prepare datasets

We use the processed datasets uploaded to Huggingface Hub by PiSSA. One can download the datasets using the following code:

```python
from datasets import load_dataset
train_data = load_dataset("fxmeng/pissa-dataset", split="train") # from PiSSA

# MetamathQA dataset
math_types = {"GSM_Rephrased", "GSM_AnsAug", "GSM_SV", "GSM_FOBAR", "MATH_Rephrased", "MATH_AnsAug", "MATH_SV", "MATH_FOBAR"}
train_data = train_data.filter(lambda example: example["type"] in math_types)
train_data.to_json("dataset/metamath/train.json")

# CodeFeedback-Python dataset
code_types = {"python"}
train_data = train_data.filter(lambda example: example["type"] in code_types)
train_data.to_json("dataset/python/train.json")

# Commonsense reasoning
train_data = load_dataset("zwhe99/commonsense_170k", split="train")
train_data.to_json("dataset/commonsense/train.json")
```

### 🔁 Reproduce Results

To reproduce the results, please run the following bash scripts:

```bash
# metamath
bash scripts/metamath/run.sh

# code
bash scripts/code/run.sh

# commonsense
bash scripts/commonsense/run.sh
```

