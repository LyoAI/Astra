import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import copy
import time
import random
import logging
import datasets
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.distributed
import transformers
import matplotlib.pyplot as plt
from transformers import Trainer, BitsAndBytesConfig
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Optional, Dict, Sequence, List, Union
from dataclasses import dataclass, field
from datasets import load_dataset, concatenate_datasets
from peft import prepare_model_for_kbit_training, PeftModel
from setproctitle import setproctitle
setproctitle("Astra")

IGNORE_INDEX = -100
logger = logging.getLogger(__name__)

def setup_logger(log_file=None):
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    if log_file:
        handler = logging.FileHandler(log_file)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

def is_main_process():
    """Check if current process is the main process (rank 0)"""
    if not torch.distributed.is_initialized():
        return True
    return torch.distributed.get_rank() == 0


PROMPT = (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    )


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    # Base model or residual model setting
    model_name_or_path: Optional[str] = field(default="meta-llama/Llama-2-7b-hf")

    # gradient attribution setting
    calibration_data: Optional[str] = field(
        default="wikitext2",
        metadata={"help": ("Calibration dataset used for gradient attribution")}
    )
    num_calibration_samples: Optional[int] = field(
        default=64,
        metadata={"help": ("Number of calibration data used for gradient attribution")}
    )
    calibration_max_seq_length: Optional[int] = field(
        default=512,
        metadata={"help": ("Max sequence length of calibration data")}
    )

    # Lora or Astra setting
    full_finetune : Optional[bool] = field(default=False)
    adapter_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": ("Pre-initialized Astra adapter path; when this is not None, the following arguments are ignored.")}
    )
    init_weights: str = field(
        default="astra",
        metadata={"help": ("True -> LoRA; `astra` -> Astra;")}
    )
    target_modules : List[str] = field(
        default_factory=lambda: ["q_proj","v_proj","k_proj","o_proj","gate_proj","down_proj","up_proj"],
        metadata={"help": ("Target Modules to apply LoRA or Astra")}
    )
    lora_rank : Optional[int] = field(default=8)
    lora_alpha : Optional[float] = field(default=32.)
    lora_dropout : Optional[float] = field(
        default=0.,
        metadata={"help": ("Must be set to 0 when using Astra.")}
    )
    use_dora : Optional[bool] = field(default=False)

    # Quantization setting
    bits: int = field(
        default=16,
        metadata={"help": "How many bits to use."}
    )
    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )

    # DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    sub_task: List[str] = field(default=None)
    dataset_split: str = field(default="train", metadata={"help": "(`['train', 'test', 'eval']`):"})
    dataset_field: List[str] = field(default=None, metadata={"help": "Fields of dataset input and output."})
    shuffle_dataset : Optional[bool] = field(default=False)

    # TrainingArguments
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."}
    )
    merge : Optional[bool] = field(
        default=False,
        metadata={"help": "Merge the adapter to the residual model or LoRA to the base model"}
    )
    
    # Logging arguments
    log_file: Optional[str] = field(
        default=None,
        metadata={"help": "Path to save the training log file."}
    )
    log_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Directory to save the training logs and plots."}
    )
    # Other arguments
    stage_merged: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to merge the adapter to the residual model during training."}
    )
    merge_interval: Optional[int] = field(
        default=None,
        metadata={"help": "Interval to merge the adapter to the residual model during training."}
    )


class MergePeftModelCallback(transformers.TrainerCallback):
    def __init__(self, script_args):
        self.script_args = script_args
        self.merge_interval = script_args.merge_interval

    def save_model(self, args, state, kwargs):
        if args.local_rank == 0:
            logger.info(f"Available kwargs keys: {kwargs.keys()}")
            logger.info('Saving PEFT checkpoint...')
        if state.best_model_checkpoint is not None:
            checkpoint_folder = os.path.join(state.best_model_checkpoint, "adapter_model")
        else:
            checkpoint_folder = os.path.join(args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}")

        peft_model_path = os.path.join(checkpoint_folder, "adapter_model")
        kwargs["model"].save_pretrained(peft_model_path)

    def on_save(self, args, state, control, **kwargs):
        self.save_model(args, state, kwargs)
        return control

    def on_step_end(self, args, state, control, **kwargs):
        current_step = state.global_step
        if (self.merge_interval is not None and
            current_step >= self.merge_interval and
            current_step % self.merge_interval == 0):

            if is_main_process():
                logger.info(f"⏳ Starting merge at step {current_step}")

            model = kwargs['model']
            if isinstance(model, PeftModel):
                merged_model = model.merge_and_unload()
                tmp_merge_dir = os.path.join(self.script_args.output_dir, f"tmp_merge_{current_step}")
                os.makedirs(tmp_merge_dir, exist_ok=True)
                merged_model.save_pretrained(tmp_merge_dir)
                if is_main_process():
                    logger.info(f"💾 Merged model saved to {tmp_merge_dir}")
                
                if torch.distributed.is_initialized():
                    torch.distributed.barrier() 

                model = build_model(self.script_args, checkpoint_dir=None, use_in_merge_callback_path=tmp_merge_dir)
                kwargs["model"] = model
                #TODO: Reinitialize optimizer and lr_scheduler
                self.save_model(args, state, kwargs)
                if is_main_process():
                    logger.info(f"✨ New adapter initialized and has been saved")
            
            def touch(fname, times=None):
                with open(fname, 'a'):
                    os.utime(fname, times)
            touch(os.path.join(args.output_dir, 'rebuild'))

            if torch.distributed.is_initialized():
                torch.distributed.barrier()   

class SavePeftModelCallback(transformers.TrainerCallback):
    def save_model(self, args, state, kwargs):
        if args.local_rank == 0:
            logger.info(f"Available kwargs keys: {kwargs.keys()}")
            logger.info('Saving PEFT checkpoint...')
        if state.best_model_checkpoint is not None:
            checkpoint_folder = os.path.join(state.best_model_checkpoint, "adapter_model")
        else:
            checkpoint_folder = os.path.join(args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}")

        peft_model_path = os.path.join(checkpoint_folder, "adapter_model")
        kwargs["model"].save_pretrained(peft_model_path)

    def on_save(self, args, state, control, **kwargs):
        self.save_model(args, state, kwargs)
        return control

    def on_train_end(self, args, state, control, **kwargs):
        def touch(fname, times=None):
            with open(fname, 'a'):
                os.utime(fname, times)
        touch(os.path.join(args.output_dir, 'completed'))
        self.save_model(args, state, kwargs)


def get_last_checkpoint(checkpoint_dir):
    if os.path.isdir(checkpoint_dir):
        is_completed = os.path.exists(os.path.join(checkpoint_dir, 'completed'))
        if is_completed: return None # already finished
        max_step = 0
        for filename in os.listdir(checkpoint_dir):
            if os.path.isdir(os.path.join(checkpoint_dir, filename)) and filename.startswith(PREFIX_CHECKPOINT_DIR):
                max_step = max(max_step, int(filename.replace(PREFIX_CHECKPOINT_DIR + '-', '')))
        if max_step == 0: return None
        latest_ckpt_dir = os.path.join(checkpoint_dir, f'{PREFIX_CHECKPOINT_DIR}-{max_step}')
        logger.info(f"Found a previous checkpoint at: {checkpoint_dir}")
        return latest_ckpt_dir
    return None # first training


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [tokenizer(text, max_length=tokenizer.model_max_length, truncation=True) for text in strings]
    input_ids = labels = [np.array(tokenized.input_ids) for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [len(tokenized.input_ids) for tokenized in tokenized_list]

    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def preprocess(
    sources: Sequence[str],
    targets: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """Preprocess the data by tokenizing."""
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (examples, sources)]
    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX
    return dict(input_ids=input_ids, labels=labels)


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""
    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = [torch.tensor(x) for x in input_ids]
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = [torch.tensor(x) for x in labels]
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )


def train_tokenize_function(examples, tokenizer, query, response):
    sources = [PROMPT.format_map(dict(instruction=instruction)) for instruction in examples[query]]
    targets = [f"{output}\n{tokenizer.eos_token}" for output in examples[response]]
    data_dict = preprocess(sources, targets, tokenizer)
    return data_dict


def build_model(script_args, checkpoint_dir, tokenizer: Optional[transformers.PreTrainedTokenizer] = None, use_in_merge_callback_path: Optional[str] = None):
    if script_args.full_finetune:
        assert script_args.bits in [16, 32]
    compute_dtype = (torch.bfloat16 if script_args.bf16 else torch.float32)

    # Base Model Initialization
    model = AutoModelForCausalLM.from_pretrained(
        script_args.model_name_or_path if not use_in_merge_callback_path else use_in_merge_callback_path,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=script_args.bits == 4,
            load_in_8bit=script_args.bits == 8,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=script_args.double_quant,
            bnb_4bit_quant_type=script_args.quant_type,
        ) if script_args.bits in [4, 8] else None,
        torch_dtype=compute_dtype,
        trust_remote_code=True
    )
    setattr(model, 'model_parallel', True)
    setattr(model, 'is_parallelizable', True)
    # Tokenizer
    
    if not script_args.full_finetune:
        if script_args.bits < 16:
            model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=script_args.gradient_checkpointing)

        if checkpoint_dir is not None:
            if script_args.local_rank == 0:
                logger.info(f"Loading adapters from {checkpoint_dir}.")
            # os.path.join(checkpoint_dir, 'adapter_model')
            model = PeftModel.from_pretrained(model, checkpoint_dir, is_trainable=True)
        elif script_args.adapter_name_or_path is not None and not use_in_merge_callback_path:
            if script_args.local_rank == 0:
                logger.info(f"Initilize adapters from {script_args.model_name_or_path}/{script_args.adapter_name_or_path}.")
            model = PeftModel.from_pretrained(model, script_args.model_name_or_path, subfolder = script_args.adapter_name_or_path, is_trainable=True)
        else:
            raise NotImplementedError("Please initialize adapters before training")
    else:
        for param in model.parameters():
            param.requires_grad = True

    for name, module in model.named_modules():
        if 'norm' in name or 'gate' in name:
            module = module.to(torch.float32)

    return model


def train():
    parser = transformers.HfArgumentParser(TrainingArguments)
    script_args = parser.parse_args_into_dataclasses()[0]
    log_level = script_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()
        
    # Setup logger with log file if specified
    if script_args.log_file:
        setup_logger(script_args.log_file)
        
    if script_args.local_rank == 0:
        logger.info('='*100)
        logger.info(script_args)
    
    tokenizer = AutoTokenizer.from_pretrained(
        script_args.model_name_or_path,
        model_max_length=script_args.model_max_length,
        padding_side="right",
        use_fast=True,
        trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if script_args.local_rank == 0:
        logger.info("Load tokenizer from {} over.".format(script_args.model_name_or_path))
    
    resume_from_checkpoint_dir = get_last_checkpoint(script_args.output_dir)
    model = build_model(script_args, resume_from_checkpoint_dir)
    if script_args.local_rank == 0 and not script_args.full_finetune:
        logger.info("+" * 100)
        trainable_params, all_params = model.get_nb_trainable_parameters()
        logger.info(f"trainable params: {trainable_params:,} || all params: {all_params-trainable_params:,} || trainable%: {100 * trainable_params / (all_params - trainable_params)}")
        logger.info("+" * 100)
    elif script_args.local_rank == 0 and script_args.full_finetune:
        def count_trainable_parameters(model):
            return sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info("+" * 100)
        trainable_params = count_trainable_parameters(model)
        all_params = sum(p.numel() for p in model.parameters())
        logger.info(f"trainable params: {trainable_params:,} || all params: {all_params:,} || trainable%: {100 * trainable_params / (all_params)}")
        logger.info("+" * 100)

    all_training_dataset = []
    for task in script_args.sub_task:
        if ":" in task: # e.g. math:500, gsm8k:100
            cur_task, num_split = task.split(":")
            cur_split = f"{script_args.dataset_split}[:{num_split}]"
        else:
            cur_task, cur_split = task, script_args.dataset_split

        ds = load_dataset(script_args.data_path, data_dir=cur_task, split=cur_split)
        if script_args.local_rank == 0:
            logger.info(f"{script_args.data_path}/{cur_task}/{cur_split}/{ds.num_rows}")
            for k,v in ds[0].items():
                logger.info("-"*100)
                logger.info(f"{k}:\t{v}")
            logger.info("+"*100)
        all_training_dataset.append(ds)
        
    raw_train_datasets = concatenate_datasets(all_training_dataset)
    if script_args.shuffle_dataset:
        if script_args.local_rank == 0:
            logger.info(f"Shuffle dataset with seed={script_args.seed}")
        raw_train_datasets = raw_train_datasets.shuffle(seed=script_args.seed)

    if script_args.local_rank > 0: 
        torch.distributed.barrier()
        
    train_dataset = raw_train_datasets.map(
        train_tokenize_function,
        batched=True,
        batch_size=3000,
        num_proc=32,
        remove_columns=raw_train_datasets.column_names,
        load_from_cache_file=True,
        desc="Running tokenizer on train dataset",
        fn_kwargs={"tokenizer": tokenizer, "query": script_args.dataset_field[0], "response": script_args.dataset_field[1]}
    )

        
    if script_args.local_rank == 0:
        torch.distributed.barrier()
        logger.info(str(model))
        logger.info(f"Training dataset samples: {len(train_dataset)}")
        for index in random.sample(range(len(train_dataset)), 3):
            logger.info(f"Sample {index} of the training set: {train_dataset[index]['input_ids']}, {train_dataset[index]['labels']}.")
            logger.info(f"Sample {index} of the training set: {tokenizer.decode(list(train_dataset[index]['input_ids']))}.")

    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    data_module = dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)

    trainer = Trainer(model=model, tokenizer=tokenizer, args=script_args, **data_module)
    if not script_args.full_finetune:
        trainer.add_callback(SavePeftModelCallback)
    if script_args.stage_merged:
        merge_callback = MergePeftModelCallback(script_args=script_args)
        trainer.add_callback(merge_callback)
    trainer.train(resume_from_checkpoint = resume_from_checkpoint_dir)
    trainer.save_state()
    
    if not script_args.full_finetune and script_args.merge:
        model = model.merge_and_unload()
        model.save_pretrained(script_args.output_dir)
        tokenizer.save_pretrained(script_args.output_dir)
    if script_args.full_finetune:
        safe_save_model_for_hf_trainer(trainer=trainer, output_dir=script_args.output_dir)
    
        # Plot and save training loss
    if script_args.log_dir and script_args.local_rank == 0:
        # Plot training loss
        log_history = trainer.state.log_history
        steps = [log["step"] for log in log_history if "loss" in log]
        losses = [log["loss"] for log in log_history if "loss" in log]
        
        plt.figure(figsize=(10, 6))
        plt.plot(steps, losses, label="Training Loss")
        plt.xlabel("Steps")
        plt.ylabel("Loss")
        plt.legend()
        plt.title(f"Training Loss Curve - {script_args.init_weights}")
        
        # Save plot
        os.makedirs(os.path.join("figure", script_args.init_weights), exist_ok=True)
        sub_task = [task.split(":")[0] for task in script_args.sub_task] #['math', 'gsm8k']
        plot_path = os.path.join("figure", script_args.init_weights, f"training-loss-{'-'.join(sub_task)}.png")
        plt.savefig(plot_path, dpi=500)
        plt.close()
        
        # Save log history to Excel
        log_df = pd.DataFrame(log_history)
        excel_path = f"{script_args.log_dir}-training_logs.xlsx"
        log_df.to_excel(excel_path, index=False)
        
        logger.info(f"Training logs and plots saved to {script_args.log_dir}")
    
    if script_args.local_rank == 0:
        logger.info("Training complete.")


if __name__ == "__main__":
    try:
        train()
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        raise e
