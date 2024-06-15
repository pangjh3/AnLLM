
# Multi-nodes are also supported
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=eth1
export NCCL_IB_GID_INDEX=3
export NCCL_IB_SL=3
export NCCL_NET_GDR_READ=1

export MASTER_ADDR="${CHIEF_IP:=localhost}"
export MASTER_PORT="${MASTER_PORT:=29500}"

export HF_HOME=/jianhuipang_qy3/hf_cache
export TRANSFORMERS_CACHE=/jianhuipang_qy3/hf_cache

train_path=/newmodels/run_allms.py
model_path=/jianhuipang/opensourcellms/llama2/Llama-2-7b-hf
model_path=/jianhuipang/opensourcellms/llama2/Llama-2-13b-hf

# deepspeedpath=/jianhuipang/LLMs4MT/train/deepspeed_config_zero2.json
deepspeedpath=/jianhuipang/llama2_sft/train/deepspeed_config_bf16.json

datanamepath=/jianhuipang/datasets/RedPajama-Data-1T-Sample
echo $datanamepath

model_save=/jianhuipang_qy3/models/checkpoints_ct/punc/allm-juhao-13b

if [ ! -d "$model_save" ]; then
    mkdir -p "$model_save"
fi

    # --training_data_num_lines $size  \
# HOST_NUM will be 1
HOST_NUM=4
torchrun --nnodes $HOST_NUM --node_rank $INDEX --nproc_per_node 8 \
    --master_addr $MASTER_ADDR --master_port $MASTER_PORT  \
    ${train_path} \
    --deepspeed $deepspeedpath \
    --anchor_symbols "." \
    --model_name_or_path ${model_path} \
    --dataset_name $datanamepath \
    --preprocessing_num_workers 32 \
    --dataloader_num_workers 8 \
    --dataloader_pin_memory True \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --num_train_epochs 1 \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_steps 20 \
    --lr_scheduler_type "constant_with_warmup" \
    --logging_steps 1 \
    --block_size 4096 \
    --do_train \
    --evaluation_strategy "no" \
    --validation_split_percentage 1 \
    --bf16 True \
    --bf16_full_eval True \
    --ddp_timeout 72000 \
    --seed 34 \
    --overwrite_output_dir \
    --gradient_checkpointing True \
    --output_dir ${model_save} \
    2>&1 | tee -a ${model_save}/log.txt

# Use streaming for large datasets and specify the max_steps
#    --streaming \
#    --max_steps 2500 \


# done

