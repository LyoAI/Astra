import os
import random
import torch
from datasets import load_dataset
from typing import Optional, Literal, Union, List
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from transformers import DataCollatorForSeq2Seq


def get_calibration_dataloader(
    dataset_name: Literal['metamath', 'code', 'commonsense', 'wikitext2', 'alpaca'],
    tokenizer,
    num_samples: Optional[int] = 64,
    seq_len: Optional[float] = 512,
    padding: Optional[Union[str, bool]] = 'max_length',
    batch_size: Optional[int] = 1,
    seed: Optional[int] = 42,
    calib_on_inputs: Optional[bool] = None,
    add_eos_token: Optional[bool] = False
):
    random.seed(seed)
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer, return_tensors='pt', padding=True
    )
    class TrainDataset(Dataset):
        def __init__(self, input_tensors) -> None:
            self.inputs = input_tensors
            self.targets = input_tensors.clone()
        
        def __len__(self):
            return len(self.inputs)
        
        def __getitem__(self, index):
            result = {}
            result["input_ids"] = self.inputs[index, :-1]
            result["labels"] = self.targets[index, 1:]
            return result

    def tokenize(prompt, add_eos_token: bool =True):
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=seq_len,
            padding=padding,
            return_tensors=None,
        )
        if (
            result["input_ids"][-1] != tokenizer.eos_token_id
            and len(result["input_ids"]) < seq_len
            and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)

        result["labels"] = result["input_ids"].copy()[1:]
        result["input_ids"] = result["input_ids"][:-1]
        result["attention_mask"] = result["attention_mask"][:-1]
        return result

    def process_pretrain_data(train_data, tokenizer, seq_len, field_name):
        train_ids = tokenizer("\n\n".join(train_data[field_name]), return_tensors='pt').input_ids[0]
        train_ids_batch = []
        nsamples = train_ids.numel() // seq_len

        for i in range(nsamples):
            batch = train_ids[(i * seq_len):((i + 1) * seq_len)]
            train_ids_batch.append(batch)
        train_ids_batch = torch.stack(train_ids_batch)
        return TrainDataset(input_tensors=train_ids_batch)
    
    def process_task_data(train_data):
        data_point = train_data["full_text"]
        tokenized_full_prompt = tokenize(data_point, add_eos_token)
        if not calib_on_inputs:
            user_prompt = train_data["user_prompt"]
            tokenized_user_prompt = tokenize(user_prompt, add_eos_token)
            user_prompt_len = len(tokenized_user_prompt["input_ids"])

            if add_eos_token:
                user_prompt_len -= 1
            
            tokenized_full_prompt["labels"] = [-100] * user_prompt_len + tokenized_full_prompt["labels"][user_prompt_len:]

        return tokenized_full_prompt
    
    def preprocess(sample):
        example = {}
        example["full_text"] = (
            sample["instruction"]
            + "\n### Response:"
            + sample["output"]
        )
        example["user_prompt"] = (  
            sample["instruction"]
            + "\n### Response:"
        )
        return example

    if 'metamath' in dataset_name:
        train_data = load_dataset("fxmeng/pissa-dataset", split="train") # huggingface dataset
        selected_types = {"GSM_Rephrased", "GSM_AnsAug", "GSM_SV", "GSM_FOBAR", "MATH_Rephrased", "MATH_AnsAug", "MATH_SV", "MATH_FOBAR"}
        train_data = train_data.filter(lambda example: example["type"] in selected_types)
        random_indices = random.sample(range(len(train_data)), num_samples)
        train_data = train_data.select(random_indices)
        train_data = train_data.map(preprocess)
        train_data = train_data.map(process_task_data)
        train_dataset = train_data.select_columns(["input_ids", "attention_mask", "labels"])
    
    elif "code" in dataset_name:
        train_data = load_dataset("fxmeng/pissa-dataset", split="train") # huggingface dataset
        selected_types = {"python"}
        train_data = train_data.filter(lambda example: example["type"] in selected_types)
        random_indices = random.sample(range(len(train_data)), num_samples)
        train_data = train_data.select(random_indices)
        train_data = train_data.map(preprocess)
        train_data = train_data.map(process_task_data)        
        train_dataset = train_data.select_columns(["input_ids", "attention_mask", "labels"])

    elif 'commonsense' in dataset_name:
        train_data = load_dataset("zwhe99/commonsense_170k", split="train")
        random_indices = random.sample(range(len(train_data)), num_samples)
        train_data = train_data.select(random_indices)
        train_data = train_data.map(preprocess)
        train_data = train_data.map(process_task_data)
        train_dataset = train_data.select_columns(["input_ids", "attention_mask", "labels"])

    elif 'wikitext2' in dataset_name:
        train_data = load_dataset(
            'wikitext',
            'wikitext-2-raw-v1',
            split='train'
        )
        random_indices = random.sample(range(len(train_data)), num_samples)
        train_data = train_data.select(random_indices)
        train_dataset = process_pretrain_data(train_data, tokenizer, seq_len, 'text')
        data_collator = None
    
    elif 'alpaca' in dataset_name:
        train_data = load_dataset("yahma/alpaca-cleaned", split="train")
        random_indices = random.sample(range(len(train_data)), num_samples)
        train_data = train_data.select(random_indices)
        def preprocess_alpaca(sample):
            example = {}
            if sample["input"] is None:
                example["full_text"] = (
                    "Instruction: "
                    + sample["instruction"]
                    + "\n### Response:"
                    + sample["output"]
                )
                example["user_prompt"] = (
                    "Instruction: "
                    + sample["instruction"]
                    + "\n### Response:"
                )
            else:
                example["full_text"] = (
                    "Instruction: "
                    + sample["instruction"]
                    + "\n### Input:"
                    + sample["input"]
                    + "\n### Response:"
                    + sample["output"]
                )
                example["user_prompt"] = (
                    "Instruction: "
                    + sample["instruction"]
                    + "\n### Input:"
                    + sample["input"]
                    + "\n### Response:"
                )
            return example
        train_data = train_data.map(preprocess_alpaca)
        train_data = train_data.map(process_task_data)
        train_dataset = train_data.select_columns(["input_ids", "attention_mask", "labels"])
    elif 'nq_open' in dataset_name:
        train_data = load_dataset("google-research-datasets/nq_open", split="train")
        random_indices = random.sample(range(len(train_data)), num_samples)
        train_data = train_data.select(random_indices)
        def preprocess_nq_open(sample):
            answers = sample.get("answer", [])
            if isinstance(answers, list):
                answers_text = ", ".join(answers)
            else:
                answers_text = str(answers)
            example = {}
            example["full_text"] = (
                "Question: "
                + sample["question"]
                + "\n### Response:"
                + answers_text
            )
            example["user_prompt"] = (  
                "Question: "
                + sample["question"]
                + "\n### Response:"
            )
            return example
        train_data = train_data.map(preprocess_nq_open)
        train_data = train_data.map(process_task_data)
        train_dataset = train_data.select_columns(["input_ids", "attention_mask", "labels"])
    elif dataset_name in ["mnli", "sst2", "mrpc", "cola", "qnli", "qqp", "rte", "stsb"]:
        train_data = load_dataset("glue", dataset_name, split="train")
        random_indices = random.sample(range(len(train_data)), num_samples)
        train_data = train_data.select(random_indices)
        train_data = train_data.add_column("dataset_name", [dataset_name] * len(train_data))
        def preprocess_glue(examples):
            if examples["dataset_name"] == 'sst2' or examples["dataset_name"] == 'cola':
                outputs = tokenizer(examples["sentence"], truncation=True, max_length=seq_len)
            elif examples["dataset_name"] == 'qnli':
                outputs = tokenizer(examples["question"], examples["sentence"], truncation=True, max_length=seq_len)
            elif examples["dataset_name"] == 'qqp':
                outputs = tokenizer(examples["question1"], examples["question2"], truncation=True, max_length=seq_len)
            elif examples["dataset_name"] == 'mnli':
                outputs = tokenizer(examples["premise"], examples["hypothesis"], truncation=True, max_length=seq_len)
            else:
                outputs = tokenizer(examples["sentence1"], examples["sentence2"], truncation=True, max_length=seq_len)
            return outputs
        train_data = train_data.map(preprocess_glue)
        train_data = train_data.rename_column("label", "labels")
        train_dataset = train_data.select_columns(["input_ids", "attention_mask", "labels"])
    else:
        raise NotImplementedError
    
    print("=>Done Loading Data!")

    if dataset_name in ["mnli", "sst2", "mrpc", "cola", "qnli", "qqp", "rte", "stsb"]:
        def collate_fn(examples):
            return tokenizer.pad(examples, padding="longest", return_tensors="pt")
        return DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True)
    else:
        return DataLoader(train_dataset, batch_size=batch_size, collate_fn=data_collator, shuffle=True)