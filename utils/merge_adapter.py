from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from transformers import HfArgumentParser
from dataclasses import dataclass, field
from typing import Optional
import argparse
import torch
import os
import re

@dataclass
class MergeArguments:
    base_model: Optional[str] = field(default=None)
    adapter_path: Optional[str] = field(default=None)
    output_path: Optional[str] = field(default=None)
    training_checkpoints: Optional[bool] = field(default=False)
    torch_device: Optional[str] = field(default="cuda")

parser = HfArgumentParser(MergeArguments)
args = parser.parse_args_into_dataclasses()[0]

print(
    "params\n",
    "base_model: ", args.base_model, "\n",
    "adapter_path: ", args.adapter_path, "\n",
    "output_path: ", args.output_path, "\n",
    "training_checkpoints: ", args.training_checkpoints, "\n",
    "torch_device: ", args.torch_device, "\n"
)

# find the latest checkpoint directory under adapter_path
if args.training_checkpoints:
    # Find the latest checkpoint directory
    checkpoint_dirs = [d for d in os.listdir(args.adapter_path) 
                      if d.startswith(PREFIX_CHECKPOINT_DIR)]
    if checkpoint_dirs:
        # Extract numbers and find the maximum
        numbers = [int(re.search(r'\d+$', d).group()) for d in checkpoint_dirs]
        latest_checkpoint = f"{PREFIX_CHECKPOINT_DIR}-{max(numbers)}"
        adapter_path = os.path.join(args.adapter_path, latest_checkpoint)
    else:
        raise ValueError(f"No checkpoint directories found in {args.adapter_path}")
else:
    adapter_path = args.adapter_path

model = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=torch.bfloat16, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(args.base_model)
model = PeftModel.from_pretrained(model, adapter_path, local_files_only=True, torch_device=args.torch_device)
model = model.merge_and_unload()
model.save_pretrained(args.output_path)
tokenizer.save_pretrained(args.output_path)