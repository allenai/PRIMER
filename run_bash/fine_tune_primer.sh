#!/bin/bash

cd ../script

DATA_NAME="wcep"


# PRIMER model for WCEP
MODEL_NAME="PRIMER_wcep"
MODEL_PATH="PRIMER_wcep"

# NUM_TRAIN_DATA=100

CUDA_LAUNCH_BLOCKING=1 python primer_main_fs.py  \
                --gpus 1  \
                --mode train \
                --lr 3e-5 \
                --label_smoothing 0.1 \
                --accum_data_per_step 10 \
                --warmup_steps 100 \
                --total_steps 1000 \
                --batch_size 2 \
                --model_path ../models/$MODEL_NAME/  \
                --dataset_name ${DATA_NAME} \
                --primer_path ../${MODEL_PATH} \
                --num_workers 8 \
                --progress_bar_refresh_rate 50 \
                --rand_seed 42 \
                --saveTopK 5 \
                --test_imediate \
                --test_batch_size 8 \
                --grad_ckpt \
        > ../finetune_${DATA_NAME}_${MODEL_NAME}_${RAND_SEED}.out 2>&1