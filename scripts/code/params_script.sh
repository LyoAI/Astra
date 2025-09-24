#!/bin/bash

# Base model or residual model setting
export MODEL_NAME_OR_PATH="meta-llama/Llama-2-7b-hf" # Qwen/Qwen2-0.5B meta-llama/Llama-2-7b-hf meta-llama/Meta-Llama-3-8B
export MODEL_NAME="Llama2-7B"
export FULL_FINETUNE=false
export INIT_WEIGHTS="astra" # Full_FT astra
export TARGET_MODULES="q_proj v_proj k_proj o_proj gate_proj down_proj up_proj"
export LORA_RANK=128
export LORA_ALPHA=128
export LORA_DROPOUT=0
export USE_DORA=false
export HF_ENDPOINT=https://hf-mirror.com
export RES_MODEL="outputs/${INIT_WEIGHTS}-${MODEL_NAME}-r${LORA_RANK}"
export OUTPUT_PATH="outputs/code-${INIT_WEIGHTS}-${MODEL_NAME}-r${LORA_RANK}"

#astra configurations
export CACHE_FILE="cache/Eigen-${INIT_WEIGHTS}-${MODEL_NAME}-r${LORA_RANK}.pt"
export COVARIANCE_FILE="cache/Covariance-${INIT_WEIGHTS}-${MODEL_NAME}-r${LORA_RANK}.pt"
export ASTRA_METHOD="IPM" # "KPM"
export USE_FLOAT16_FOR_COVARIANCE=true
export PRUNE_TEMPORARY_FIELDS=true

# Quantization setting
export BITS=bf16
export DOUBLE_QUANT=true
export QUANT_TYPE="nf4"

# DataArguments
export DATA_PATH="dataset"
export SUB_TASK="python:100000"
export DATASET_SPLIT="train"
export DATASET_FIELD="instruction output"
export SHUFFLE_DATASET=false

# TrainingArguments
export OPTIM="adamw_torch"
export MODEL_MAX_LENGTH=512
export MERGE=true
export GRADIENT_CHECKPOINTING=true
export BF16=true
export SEED=42
export STAGE_MERGED=false
export MERGE_INTERVAL=2

# DeepSpeed configuration
export DEEPSPEED_CONFIG="configs/ds_config_zero2_no_offload.json"

# Batch size configuration
export PER_DEVICE_TRAIN_BATCH_SIZE=8
export GRADIENT_ACCUMULATION_STEPS=4

# Learning rate configuration
export LEARNING_RATE=2e-5 # 3e-4, 2e-5
export WEIGHT_DECAY=0.0
export WARMUP_RATIO=0.03
export LR_SCHEDULER_TYPE="cosine"  # or "linear"

# Training steps configuration
export NUM_TRAIN_EPOCHS=1
export SAVE_STRATEGY="steps"
export SAVE_STEPS=1000
export SAVE_TOTAL_LIMIT=1
export LOGGING_STEPS=1

# Device configuration
export MASTER_PORT=29501
export LOCALHOST="localhost:0,1,2,3"
export NUM_GPUS=$(echo $LOCALHOST | tr -cd ',' | wc -c)
export NUM_GPUS=$((NUM_GPUS + 1))
export WORLD_SIZE=$NUM_GPUS
export BATCH_SIZE=$((PER_DEVICE_TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS * NUM_GPUS))
export AVAILABLE_DEVICES="0"
export EVAL_DEVICE="0,1,2,3"

# Logging arguments
export LOG_FILE="logs/code/${INIT_WEIGHTS}-${ASTRA_METHOD}-${MODEL_NAME}-r${LORA_RANK}.log"
export LOG_DIR="logs/log_history/code/${INIT_WEIGHTS}-${ASTRA_METHOD}-${MODEL_NAME}-r${LORA_RANK}"
export REPORT_TO="wandb"

# Calibration parameters
export CALIBRATION_DATASET="code"
export NUM_CALIBRATION_SAMPLES=256
export CALIBRATION_MAX_SEQ_LENGTH=512
export CALIBRATION_DEVICE="cuda:0"
export RANK_ALLOCATION=false
export RANK_PATTERN_PATH="cache/rank_pattern.pt"

# Helper functions
setup_logging() {
    mkdir -p "logs/log_history/code"
    mkdir -p "logs/code"
    if [ "$INIT_WEIGHTS" == "Full_FT" ]; then
        LOG_DIR="logs/log_history/code/${INIT_WEIGHTS}_${MODEL_NAME}"
        LOG_FILE="logs/code/${INIT_WEIGHTS}_${MODEL_NAME}.log"
    else
        LOG_DIR="logs/log_history/code/${INIT_WEIGHTS}-${ASTRA_METHOD}-${MODEL_NAME}-r${LORA_RANK}"
        LOG_FILE="logs/code/${INIT_WEIGHTS}-${ASTRA_METHOD}-${MODEL_NAME}-r${LORA_RANK}.log"
    fi
}

# Helper functions
print_params() {
    echo "++++++++++Base model parameters++++++++++"
    echo "MODEL_NAME_OR_PATH: $MODEL_NAME_OR_PATH"
    echo "MODEL_NAME: $MODEL_NAME"
    echo "FULL_FINETUNE: $FULL_FINETUNE"
    echo "INIT_WEIGHTS: $INIT_WEIGHTS"
    echo "TARGET_MODULES: $TARGET_MODULES"
    echo "LORA_RANK: $LORA_RANK"
    echo "LORA_ALPHA: $LORA_ALPHA"
    echo "LORA_DROPOUT: $LORA_DROPOUT"
    echo "USE_DORA: $USE_DORA"
    echo "RES_MODEL: $RES_MODEL"
    echo "ADAPTER_PATH: $ADAPTER_PATH"
    echo "OUTPUT_PATH: $OUTPUT_PATH"
    
    echo "++++++++++Astra configurations++++++++++"
    echo "CACHE_FILE: $CACHE_FILE"
    echo "COVARIANCE_FILE: $COVARIANCE_FILE"
    echo "USE_FLOAT16_FOR_COVARIANCE: $USE_FLOAT16_FOR_COVARIANCE"
    echo "PRUNE_TEMPORARY_FIELDS: $PRUNE_TEMPORARY_FIELDS"
    
    echo "++++++++++Quantization parameters++++++++++"
    echo "BITS: $BITS"
    echo "DOUBLE_QUANT: $DOUBLE_QUANT"
    echo "QUANT_TYPE: $QUANT_TYPE"
    
    echo "++++++++++Data parameters++++++++++"
    echo "DATA_PATH: $DATA_PATH"
    echo "SUB_TASK: $SUB_TASK"
    echo "DATASET_SPLIT: $DATASET_SPLIT"
    echo "DATASET_FIELD: $DATASET_FIELD"
    echo "SHUFFLE_DATASET: $SHUFFLE_DATASET"
    
    echo "++++++++++Training parameters++++++++++"
    echo "OPTIM: $OPTIM"
    echo "MODEL_MAX_LENGTH: $MODEL_MAX_LENGTH"
    echo "MERGE: $MERGE"
    echo "GRADIENT_CHECKPOINTING: $GRADIENT_CHECKPOINTING"
    echo "BF16: $BF16"
    echo "SEED: $SEED"
    echo "STAGE_MERGED: $STAGE_MERGED"
    echo "MERGE_INTERVAL: $MERGE_INTERVAL"
    echo "DEEPSPEED_CONFIG: $DEEPSPEED_CONFIG"
    echo "PER_DEVICE_TRAIN_BATCH_SIZE: $PER_DEVICE_TRAIN_BATCH_SIZE"
    echo "GRADIENT_ACCUMULATION_STEPS: $GRADIENT_ACCUMULATION_STEPS"
    echo "BATCH_SIZE: $BATCH_SIZE"
    echo "LEARNING_RATE: $LEARNING_RATE"
    echo "WEIGHT_DECAY: $WEIGHT_DECAY"
    echo "WARMUP_RATIO: $WARMUP_RATIO"
    echo "LR_SCHEDULER_TYPE: $LR_SCHEDULER_TYPE"
    echo "NUM_TRAIN_EPOCHS: $NUM_TRAIN_EPOCHS"
    echo "SAVE_STRATEGY: $SAVE_STRATEGY"
    echo "SAVE_STEPS: $SAVE_STEPS"
    echo "SAVE_TOTAL_LIMIT: $SAVE_TOTAL_LIMIT"
    echo "LOGGING_STEPS: $LOGGING_STEPS"
    
    echo "++++++++++Device configuration++++++++++"
    echo "MASTER_PORT: $MASTER_PORT"
    echo "LOCALHOST: $LOCALHOST"
    echo "NUM_GPUS: $NUM_GPUS"
    echo "WORLD_SIZE: $WORLD_SIZE"
    echo "AVAILABLE_DEVICES: $AVAILABLE_DEVICES"
    
    echo "++++++++++Logging parameters++++++++++"
    echo "LOG_FILE: $LOG_FILE"
    echo "LOG_DIR: $LOG_DIR"
    echo "REPORT_TO: $REPORT_TO"
    
    echo "++++++++++Calibration parameters++++++++++"
    echo "CALIBRATION_DATASET: $CALIBRATION_DATASET"
    echo "NUM_CALIBRATION_SAMPLES: $NUM_CALIBRATION_SAMPLES"
    echo "CALIBRATION_MAX_SEQ_LENGTH: $CALIBRATION_MAX_SEQ_LENGTH"
    echo "CALIBRATION_DEVICE: $CALIBRATION_DEVICE"
    echo "RANK_ALLOCATION: $RANK_ALLOCATION"
    echo "RANK_PATTERN_PATH: $RANK_PATTERN_PATH"
}
