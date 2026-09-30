import argparse
import logging
import os
from typing import Optional

import torch
from peft import LoraConfig, get_peft_model
from peft.tuners.lora import AstraConfig, preprocess_astra
from setproctitle import setproctitle
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

from dataset.loader import get_calibration_dataloader


logger = logging.getLogger(__name__)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
setproctitle("Astra Initialization")


def setup_logger(log_file: Optional[str] = None) -> None:
    logger.setLevel(logging.INFO)
    handlers = [logging.StreamHandler()]
    if log_file:
        os.makedirs(os.path.dirname(os.path.abspath(log_file)), exist_ok=True)
        handlers.append(logging.FileHandler(log_file))

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    for handler in handlers:
        handler.setFormatter(formatter)
        logger.addHandler(handler)


def parse_args():
    parser = argparse.ArgumentParser(description="Initialize Astra")
    parser.add_argument("--model_name", type=str, help="Model name used in logs")
    parser.add_argument("--base_model_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--lora_r", type=int, default=128)
    parser.add_argument("--lora_alpha", type=int, default=128)
    parser.add_argument("--lora_dropout", type=float, default=0)
    parser.add_argument("--target_modules", nargs="+", required=True)
    parser.add_argument(
        "--bits", type=str, default="fp32", choices=["bf16", "fp16", "fp32"]
    )
    parser.add_argument(
        "--task_type", choices=["causal_lm", "seq_cls"], default="causal_lm"
    )
    parser.add_argument("--num_labels", type=int, default=None)
    parser.add_argument(
        "--calibration_dataset", dest="calibration_dataset_name", required=True
    )
    parser.add_argument("--num_calibration_samples", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_seq_len", type=int, default=512)
    parser.add_argument("--padding", default="max_length")
    parser.add_argument("--calib_on_inputs", action="store_true")
    parser.add_argument("--cache_file", type=str, default=None)
    parser.add_argument("--covariance_file", type=str, default=None)
    parser.add_argument("--use_float16_for_covariance", action="store_true")
    parser.add_argument(
        "--prune_temporary_fields",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--log_file", type=str, default=None)
    return parser.parse_args()


def load_model(script_args):
    dtype = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }[script_args.bits]

    if script_args.task_type == "seq_cls":
        if script_args.num_labels is None:
            raise ValueError("--num_labels is required when --task_type is seq_cls")
        return AutoModelForSequenceClassification.from_pretrained(
            script_args.base_model_path,
            num_labels=script_args.num_labels,
            torch_dtype=dtype,
            trust_remote_code=True,
            device_map="auto",
        )

    return AutoModelForCausalLM.from_pretrained(
        script_args.base_model_path,
        torch_dtype=dtype,
        trust_remote_code=True,
        device_map="auto",
    )


@torch.no_grad()
def run_calibration_model(model, calibration_dataloader):
    model.eval()
    for batch in tqdm(calibration_dataloader, desc="Collecting covariance matrices"):
        input_ids = batch["input_ids"].to(model.device)
        attention_mask = batch.get("attention_mask")
        attention_mask = (
            attention_mask.to(model.device) if attention_mask is not None else None
        )
        labels = batch.get("labels")
        labels = labels.to(model.device) if labels is not None else None
        model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)


def main():
    script_args = parse_args()
    setup_logger(script_args.log_file)

    model = load_model(script_args)
    tokenizer = AutoTokenizer.from_pretrained(
        script_args.base_model_path, trust_remote_code=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    padding = (
        True
        if script_args.padding == "True"
        else False
        if script_args.padding == "False"
        else script_args.padding
    )
    calibration_dataloader = get_calibration_dataloader(
        dataset_name=script_args.calibration_dataset_name,
        tokenizer=tokenizer,
        num_samples=script_args.num_calibration_samples,
        batch_size=script_args.batch_size,
        seq_len=script_args.max_seq_len,
        padding=padding,
        calib_on_inputs=script_args.calib_on_inputs,
    )

    astra_config = AstraConfig(
        cache_file=script_args.cache_file,
        covariance_file=script_args.covariance_file,
        verbose=script_args.verbose,
        use_float16_for_covariance=script_args.use_float16_for_covariance,
        prune_temporary_fields=script_args.prune_temporary_fields,
    )
    lora_config = LoraConfig(
        r=script_args.lora_r,
        lora_alpha=script_args.lora_alpha,
        init_lora_weights="astra",
        lora_dropout=script_args.lora_dropout,
        target_modules=script_args.target_modules,
        task_type="SEQ_CLS" if script_args.task_type == "seq_cls" else "CAUSAL_LM",
        astra_config=astra_config,
    )

    logger.info("Building Astra initialization with %s", script_args.base_model_path)
    preprocess_astra(
        model,
        lora_config,
        run_model=lambda: run_calibration_model(model, calibration_dataloader),
    )
    peft_model = get_peft_model(model, lora_config)
    peft_model.print_trainable_parameters()

    os.makedirs(script_args.output_dir, exist_ok=True)
    initial_adapter_dir = os.path.join(script_args.output_dir, "astra_init")
    peft_model.peft_config["default"].init_lora_weights = True
    peft_model.save_pretrained(initial_adapter_dir)

    residual_model = peft_model.unload()
    residual_model.save_pretrained(script_args.output_dir)
    tokenizer.save_pretrained(script_args.output_dir)
    logger.info(
        "Saved residual model to %s and initial adapter to %s",
        script_args.output_dir,
        initial_adapter_dir,
    )


if __name__ == "__main__":
    main()
