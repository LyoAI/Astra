<p align="center">
<h1 align="center">Astra: Activation-Space Tail-Eigenvector Low-Rank Adaptation of Large Language Models

<p align="center">
    <a href="https://arxiv.org/abs/2602.19111"><img alt="Paper" src="https://img.shields.io/badge/📄-Paper-orange"></a>
    <a href="https://github.com/LyoAI/Astra/blob/main/LICENSE"><img alt="GitHub license" src="https://img.shields.io/github/license/LyoAI/Astra"></a>
</p>

## 🔍Overview
Parameter-Efficient Fine-Tuning (PEFT) methods, especially LoRA, are widely used for adapting pre-trained models to downstream tasks due to their computational and storage efficiency. However, in the context of LoRA and its variants, the potential of activation subspaces corresponding to tail eigenvectors remains substantially under-exploited, which may lead to suboptimal fine-tuning performance. In this work, we propose Astra (Activation-Space Tail-Eigenvector Low-Rank Adaptation), a novel PEFT method that leverages the tail eigenvectors of the model output activations-estimated from a small task-specific calibration set-to construct task-adaptive low-rank adapters. By constraining updates to the subspace spanned by these tail eigenvectors, Astra achieves faster convergence and improved downstream performance with a significantly reduced parameter budget. Extensive experiments across natural language understanding (NLU) and natural language generation (NLG) tasks demonstrate that Astra consistently outperforms existing PEFT baselines across 16 benchmarks and even surpasses full fine-tuning (FFT) in certain scenarios.


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


