import os
import time
import torch
import torch.nn as nn
import torch.distributed as dist
import argparse
import logging
from peft import get_peft_model
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
from dataset.loader import get_calibration_dataloader
from setproctitle import setproctitle
from astra import preprocess_astra, AstraLayer
from astra_config import AstraConfig, MyLoraConfig


logger = logging.getLogger(__name__)

def setup_logger(log_file=None):
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    if log_file:
        handler = logging.FileHandler(log_file)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

def init_distributed():
    if not torch.distributed.is_initialized():
        logger.info("Initializing distributed environment...")
        import datetime
        try:
            torch.distributed.init_process_group(backend='nccl', timeout=datetime.timedelta(minutes=20))
            logger.info(f"Distributed initialization successful, world size: {torch.distributed.get_world_size()}")
        except Exception as e:
            logger.error(f"Distributed initialization failed: {str(e)}")

def is_main_process():
    """Check if current process is the main process (rank 0)"""
    if not torch.distributed.is_initialized():
        return True
    return torch.distributed.get_rank() == 0

from tqdm import tqdm

setproctitle("Astra Initialization")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

parser = argparse.ArgumentParser(description="Initialize Astra")
# model and lora configuration
parser.add_argument("--model_name", type=str, help="model name for logging")
parser.add_argument("--base_model_path", type=str, required=True, help="The name or path of the base model.")
parser.add_argument("--output_dir", type=str, required=True)
parser.add_argument("--init_weights", type=str, default="astra", help="LoRA initialization")
parser.add_argument("--lora_r", type=int, default=128)
parser.add_argument("--lora_alpha", type=int, default=128)
parser.add_argument("--lora_dropout", type=float, default=0)
parser.add_argument('--target_modules', nargs='+', help='target modules to apply LoRA', required=True)
parser.add_argument("--rank_allocation", action="store_true", help="whether to do rank allocation or not")
parser.add_argument("--rank_pattern", type=str, help="rank pattern cache file")
parser.add_argument("--bits", type=str, default="fp32", choices=["bf16", "fp16", "fp32"])
parser.add_argument("--task_type", choices=["causal_lm", "seq_cls"], default="causal_lm", help="NLU/NLG Tasks")
parser.add_argument("--num_labels", type=int, default=None, help="number of labels used for sequence classification")
# calibration configuration
parser.add_argument("--calibration_dataset_name", help="Calibration dataset for gradient attribution")
parser.add_argument("--num_calibration_samples", type=int, default=64)
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--max_seq_len", type=int, default=512, help="maximum sequence length for each calibration sample")
parser.add_argument("--padding", type=str, help="padding type [True, False, max_length]", default="max_length")
parser.add_argument("--calib_on_inputs", action="store_true", help="whether to calibration(compute loss) on inputs")
parser.add_argument("--device", type=str, help="device where gradient computation on", default="cuda:0")
# astra configuration
parser.add_argument("--cache_file", type=str, default=None, help="cache file to store eigne results")
parser.add_argument("--covariance_file", type=str, default=None, help="cache file to store covariance matrix")
parser.add_argument("--astra_method", type=str, default="IPM", help="The method used for Astra, 'IPM': Instruction-Previewed Mode, currently only support for IPM")
parser.add_argument("--use_float16_for_covariance", action="store_true", help="wether to use float16 to compute covariance matrix")
parser.add_argument("--prune_temporary_fields", action="store_true", help="prune the temporary attributes")
# others
parser.add_argument("--verbose", action="store_true")
parser.add_argument("--log_file", type=str, default=None, help="Log file path for logger output redirection")
script_args = parser.parse_args()

if script_args.log_file:
    setup_logger(script_args.log_file)

# model and tokenizer initialization
if script_args.task_type.lower().strip() == "seq_cls":
    assert script_args.num_labels, "num_labels should be given if task_type is seq_cls"
    model = AutoModelForSequenceClassification.from_pretrained(
        script_args.base_model_path,
        num_labels=script_args.num_labels,
        return_dict=True,
        torch_dtype=(
            torch.float16
            if script_args.bits == "fp16"
            else (torch.bfloat16 if script_args.bits == "bf16" else torch.float32)
        ),
        trust_remote_code=True,
        device_map="cuda"
    )
else:
    model = AutoModelForCausalLM.from_pretrained(
        script_args.base_model_path,
        torch_dtype=(
            torch.float16
            if script_args.bits == "fp16"
            else (torch.bfloat16 if script_args.bits == "bf16" else torch.float32)
        ),
        device_map="cuda"
    )
tokenizer = AutoTokenizer.from_pretrained(script_args.base_model_path)
tokenizer.pad_token_id = tokenizer.eos_token_id

# calibration data
if script_args.padding == "True":
    padding = True
elif script_args.padding == "False":
    padding = False
else:
    padding = "max_length"

if is_main_process():
    logger.info(
        f"base_model_path: {script_args.base_model_path}\n"
        f"output_dir: {script_args.output_dir}\n"
        f"init_weights: {script_args.init_weights}\n"
        f"lora_r: {script_args.lora_r}\n"
        f"lora_alpha: {script_args.lora_alpha}\n"
        f"lora_dropout: {script_args.lora_dropout}\n"
        f"target_modules: {script_args.target_modules}\n"
        f"calibration_dataset_name: {script_args.calibration_dataset_name}\n"
        f"num_calibration_samples: {script_args.num_calibration_samples}\n"
        f"batch_size: {script_args.batch_size}\n"
        f"max_seq_len: {script_args.max_seq_len}\n"
        f"padding: {script_args.padding}\n"
        f"calib_on_inputs: {script_args.calib_on_inputs}\n"
        f"rank_allocation: {script_args.rank_allocation}\n"
        f"device: {script_args.device}\n"
        f"log_file: {script_args.log_file}\n"
        f"task_type: {script_args.task_type}\n"
        f"num_labels: {script_args.num_labels}\n"
        f"bits: {script_args.bits}\n"
        f"model: {model}\n"
    )
    logger.info("+" * 100)

calibration_dataloader = get_calibration_dataloader(
    dataset_name=script_args.calibration_dataset_name,
    tokenizer=tokenizer,
    num_samples=script_args.num_calibration_samples,
    batch_size=script_args.batch_size,
    seq_len=script_args.max_seq_len,
    padding=padding,
    calib_on_inputs=script_args.calib_on_inputs
)

astra_config = AstraConfig(
    cache_file=script_args.cache_file,
    covariance_file=script_args.covariance_file,
    verbose=script_args.verbose,
    astra_method=script_args.astra_method,
    use_float16_for_covariance=script_args.use_float16_for_covariance,
    prune_temporary_fields=script_args.prune_temporary_fields,
    rank_allocation=script_args.rank_allocation,
    rank_pattern=script_args.rank_pattern
)

lora_config = MyLoraConfig(
    r=script_args.lora_r,
    lora_alpha=script_args.lora_alpha,
    init_lora_weights=True if "lora" in script_args.init_weights.lower().strip() else script_args.init_weights.lower().strip(),
    lora_dropout=script_args.lora_dropout,
    target_modules=script_args.target_modules,
    task_type="CAUSAL_LM",
    astra_config=astra_config
)

if script_args.task_type.lower().strip() == "seq_cls":
    lora_config.task_type = "SEQ_CLS"

def MyRun():
    iterator = tqdm(calibration_dataloader, desc=f"Covariance Collection", total=len(calibration_dataloader), leave=True)
    for batch_idx, batch in enumerate(iterator):
        if "attention_mask" in batch:
            attention_mask = batch["attention_mask"].to(device=model.device)
        else:
            attention_mask = None
        
        if "labels" in batch:
            labels = batch["labels"].to(device=model.device)
        else:
            labels = None

        input_ids = batch["input_ids"].to(device=model.device)
        model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

preprocess_astra(model, lora_config, run_model=MyRun, log_file=script_args.log_file)

custom_module_mapping = {nn.Linear: AstraLayer}
lora_config._register_custom_module(custom_module_mapping)
lora_config.init_lora_weights = "astra"

peft_model = get_peft_model(model=model, peft_config=lora_config)
peft_model.print_trainable_parameters()

os.makedirs(script_args.output_dir, exist_ok=True)
os.makedirs(os.path.join(script_args.output_dir, f"{script_args.init_weights.lower().strip()}_init"), exist_ok=True)

if is_main_process():
    logger.info("Starting to save PEFT modules...")
    start_time = time.time()

# Save PEFT modules:
peft_model.peft_config["default"].init_lora_weights = True
peft_model.save_pretrained(os.path.join(script_args.output_dir, f"{script_args.init_weights.lower().strip()}_init"))

# Save residual model:
peft_model = peft_model.unload()
peft_model.save_pretrained(script_args.output_dir)
# Save the tokenizer:
tokenizer.save_pretrained(script_args.output_dir)

if is_main_process():
    logger.info("Saving PEFT modules completed in %s seconds", time.time() - start_time)
    logger.info("Done! Save the model and tokenizer to %s", script_args.output_dir)
