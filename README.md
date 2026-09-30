<p align="center">
<h1 align="center">Astra: Activation-Space Tail-Eigenvector Low-Rank Adaptation of Large Language Models

<p align="center">
    <a href="https://arxiv.org/abs/2602.19111"><img alt="Paper" src="https://img.shields.io/badge/📄-Paper-orange"></a>
    <a href="https://github.com/LyoAI/Astra/blob/main/LICENSE"><img alt="GitHub license" src="https://img.shields.io/github/license/LyoAI/Astra"></a>
</p>

## 🔍Overview
In this work, we propose Astra (Activation-Space Tail-Eigenvector Low-Rank Adaptation), a novel PEFT method that leverages the tail eigenvectors of the model output activations-estimated from a small task-specific calibration set-to construct task-adaptive low-rank adapters. By constraining updates to the subspace spanned by these tail eigenvectors, Astra achieves faster convergence and improved downstream performance with a significantly reduced parameter budget. Extensive experiments across natural language understanding (NLU) and natural language generation (NLG) tasks demonstrate that Astra consistently outperforms existing PEFT baselines across 16 benchmarks and even surpasses full fine-tuning (FFT) in certain scenarios.

<img width="1041" height="395" alt="image" src="https://github.com/user-attachments/assets/a990903e-d922-49d0-8878-5e79b8529181" />

<img width="1048" height="520" alt="image" src="https://github.com/user-attachments/assets/0e7b0aee-25ec-484c-b341-1df287c4d254" />


## 🎯Quick Start

### 🤗 PEFT Integration

Astra is integrated directly in [Hugging Face PEFT](https://github.com/huggingface/peft). The examples in this repository use the PEFT implementation; until the next PEFT release is published, the dependencies pin the PEFT commit that contains Astra.

```python
import torch
from peft import LoraConfig, get_peft_model
from peft.tuners.lora import AstraConfig, preprocess_astra
from transformers import AutoModelForCausalLM, AutoTokenizer

from dataset.loader import get_calibration_dataloader

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf", dtype=torch.bfloat16, device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
calibration_dataloader = get_calibration_dataloader(
    "metamath",
    tokenizer,
    num_samples=256,
    batch_size=1,
    seq_len=512,
    padding="max_length",
    calib_on_inputs=True,
)

def run_model():
    for batch in calibration_dataloader:
        batch = {key: value.to(model.device) for key, value in batch.items()}
        with torch.no_grad():
            model(**batch)

lora_config = LoraConfig(
    init_lora_weights="astra",
    r=128,
    lora_alpha=128,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "down_proj", "up_proj"],
    astra_config=AstraConfig(),
)
preprocess_astra(model, lora_config, run_model=run_model)
peft_model = get_peft_model(model, lora_config)
```

`target_modules` must be passed explicitly because preprocessing runs before `get_peft_model`, which is where PEFT normally infers model-specific defaults.

### ⚙️Install dependencies

```sh
# step 1: create a virtual environment
conda create -n astra python=3.10

# step 2: activate the virtual environment
conda activate astra

# step 3: install dependencies from requirements.txt
pip install -r requirements.txt
```

The current dependency pins PEFT at commit `73f9a1a9`, which includes Astra. Replace this Git dependency with the next PEFT release once available.

### 💾 Save and convert an Astra adapter

Astra preprocessing creates a residual base model and saves the untrained adapter to `astra_init`. The initial adapter is required to convert a trained Astra adapter into a standard LoRA adapter:

```python
# After preprocessing, before training:
peft_model.peft_config["default"].init_lora_weights = True
peft_model.save_pretrained(os.path.join(residual_model_path, "astra_init"))

# After training:
peft_model.save_pretrained(
    lora_output_dir,
    path_initial_model_for_weight_conversion=os.path.join(residual_model_path, "astra_init"),
)
```

The converted adapter can be loaded on top of the original base model and used with standard LoRA tooling.

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




