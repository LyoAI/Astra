#!/bin/bash
source scripts/code/params_script.sh

#SAVE LOG HISTORY
setup_logging

# Print parameters
print_params

if [ -e $RES_MODEL ]; then
    echo "Use pre-initialized residual model."
else
    echo "Perform initialization"
    CUDA_VISIBLE_DEVICES=$AVAILABLE_DEVICES python init_astra.py \
        --model_name $MODEL_NAME \
        --base_model_path $MODEL_NAME_OR_PATH \
        --bits $BITS \
        --output_dir $RES_MODEL \
        --init_weights $INIT_WEIGHTS \
        --lora_r $LORA_RANK \
        --lora_alpha $LORA_ALPHA \
        --lora_dropout $LORA_DROPOUT \
        --target_modules $TARGET_MODULES \
        --calibration_dataset $CALIBRATION_DATASET \
        --num_calibration_samples $NUM_CALIBRATION_SAMPLES \
        --batch_size 1 \
        --max_seq_len $CALIBRATION_MAX_SEQ_LENGTH \
        --padding max_length \
        --verbose \
        --log_file $LOG_FILE \
        --calib_on_inputs \
        $([ "$RANK_ALLOCATION" = true ] && echo "--rank_allocation") \
        --rank_pattern $RANK_PATTERN_PATH \
        --cache_file $CACHE_FILE \
        --covariance_file $COVARIANCE_FILE \
        --astra_method $ASTRA_METHOD \
        $([ "$USE_FLOAT16_FOR_COVARIANCE" = true ] && echo "--use_float16_for_covariance") \
        $([ "$PRUNE_TEMPORARY_FIELDS" = true ] && echo "--prune_temporary_fields")
fi

# batch size = per_device_train_batch_size * gradient_accumulation_steps * num_gpus = 128  --warmup_ratio 0.03 \
deepspeed --master_port=$MASTER_PORT --include=$LOCALHOST train.py \
    --deepspeed $DEEPSPEED_CONFIG \
    --model_name_or_path $RES_MODEL \
    --full_finetune $FULL_FINETUNE \
    --bf16 \
    --init_weights $INIT_WEIGHTS \
    --lora_rank $LORA_RANK \
    --lora_alpha $LORA_ALPHA \
    --lora_dropout $LORA_DROPOUT \
    --adapter_name_or_path "$(echo ${INIT_WEIGHTS} | tr '[:upper:]' '[:lower:]')_init" \
    --data_path $DATA_PATH \
    --sub_task $SUB_TASK \
    --dataset_split $DATASET_SPLIT \
    --dataset_field $DATASET_FIELD \
    --output_dir $OUTPUT_PATH \
    --num_train_epochs $NUM_TRAIN_EPOCHS \
    --model_max_length $MODEL_MAX_LENGTH \
    --per_device_train_batch_size $PER_DEVICE_TRAIN_BATCH_SIZE \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --save_strategy $SAVE_STRATEGY \
    --save_steps $SAVE_STEPS \
    --save_total_limit $SAVE_TOTAL_LIMIT \
    --learning_rate $LEARNING_RATE \
    --weight_decay $WEIGHT_DECAY \
    --warmup_ratio $WARMUP_RATIO \
    --logging_steps $LOGGING_STEPS \
    --lr_scheduler_type $LR_SCHEDULER_TYPE \
    --report_to "wandb" \
    --merge $MERGE \
    --log_file $LOG_FILE \
    --log_dir $LOG_DIR \

# Run code generation test
if ! $MERGE; then
    echo "Merge model manually"
    python utils/merge_adapter.py \
        --base_model $RES_MODEL \
        --adapter_path $OUTPUT_PATH \
        --output_path $OUTPUT_PATH \
        --training_checkpoints
fi

CUDA_VISIBLE_DEVICES=$EVAL_DEVICE python utils/gen_vllm.py --model $OUTPUT_PATH --sub_task python --output_file $OUTPUT_PATH/python_response.jsonl --tensor_parallel_size $NUM_GPUS --log_file $LOG_FILE --batch_size 400
CUDA_VISIBLE_DEVICES=$EVAL_DEVICE python utils/code_process.py --path $OUTPUT_PATH/python_response.jsonl
evalplus.evaluate --dataset humaneval --samples $OUTPUT_PATH/humaneval.jsonl --log_file $LOG_FILE
evalplus.evaluate --dataset mbpp --samples $OUTPUT_PATH/mbpp.jsonl --log_file $LOG_FILE