#!/bin/bash
DIR=`pwd`
export CUDA_VISIBLE_DEVICES=3
TASK_NAME=mnli  #mnli sst2 stsb mnli qqp rte cola mrpc qnli
STAGE=one_stage
LRATE=1e-4
EPOCH=9
WARMUP_EPOCH=1
BATCH_SIZE_PER_GPU=64
NAME="ceva_optimize_9"
SAVE_PATH=./out/${NAME}/
mkdir -p ${SAVE_PATH}


CONFIG="config/ceva_config.json"
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% if users provide *NO* models, use the following script %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% the following command will first download huggingface models and then compress %%%%%%%
MODEL=yoshitomo-matsubara/bert-base-uncased-${TASK_NAME} ## for both student and teacher
run_cmd="python -m torch.distributed.launch --nproc_per_node=1 \
  --master_port 12345 \
  run_glue_no_trainer.py \
  --seed 42 \
  --distill_method ${STAGE} \
  --model_name_or_path ${MODEL} \
  --task_name $TASK_NAME \
  --max_length 128 \
  --pad_to_max_length \
  --per_device_train_batch_size ${BATCH_SIZE_PER_GPU} \
  --per_device_eval_batch_size 64 \
  --learning_rate $LRATE \
  --num_train_epochs ${EPOCH}\
  --num_warmup_epochs ${WARMUP_EPOCH}  \
  --eval_step 1000 \
  --deepspeed_config config/ceva_config.json \
  --deepspeed \
  --save_best_model --clean_best_model \
  --gradient_accumulation_steps 1 \
  --output_dir ${SAVE_PATH} | tee -a  ${SAVE_PATH}/train.log"

echo ${run_cmd}
eval ${run_cmd}
set +x