#!/bin/bash

cd ../script

# Multi-news dataset
DATA_NAME="multi_news"
# Multi-xscience dataset
DATA_NAME="multi_x_science_sum"
# Wikisum dataset
DATA_NAME="wikisum"
# wcep dataset
DATA_NAME="wcep"
# arxiv dataset
DATA_NAME="arxiv"


# original PRIMER model
MODEL_NAME="PRIMER"
MODEL_PATH="PRIMER_model"
nohup python primer_main.py  \
                --batch_size 16 \
                --gpus 1  \
                --mode test \
                --model_path ../models/$MODEL_NAME/  \
                --dataset_name ${DATA_NAME} \
                --primer_path ../${MODEL_PATH} \
                --num_workers 0 \
                --progress_bar_refresh_rate 50 \
        > ../test_${DATA_NAME}_${MODEL_NAME}.out &

