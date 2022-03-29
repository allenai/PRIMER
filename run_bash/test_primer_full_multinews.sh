#!/bin/bash

cd ../script

# multi-news dataset
DATA_NAME="multi_news"


# Fully supervised PRIMER model on multinews
MODEL_NAME="PRIMER_multinews"
MODEL_PATH="PRIMER_multinews"
nohup python primer_main.py  \
                --batch_size 16 \
                --gpus 1  \
                --mode test \
                --model_path ../models/$MODEL_NAME/  \
                --dataset_name ${DATA_NAME} \
                --primer_path ../${MODEL_PATH} \
                --num_workers 0 \
                --progress_bar_refresh_rate 50 \
                --beam_size 5 \
        > ../test_${DATA_NAME}_${MODEL_NAME}.out 2>&1
