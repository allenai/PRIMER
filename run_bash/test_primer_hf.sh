#!/bin/bash

cd ../script

# Multi-news dataset
DATA_NAME="multi_news"
LENGTH_LIMIT=256
# Multi-xscience dataset
DATA_NAME="multi_x_science_sum"
LENGTH_LIMIT=128
# Wikisum dataset
DATA_NAME="wikisum"
LENGTH_LIMIT=128
# wcep dataset
DATA_NAME="wcep"
LENGTH_LIMIT=50
# arxiv dataset
DATA_NAME="arxiv"
LENGTH_LIMIT=300

# original PRIMER model
MODEL_NAME="PRIMER"
MODEL_PATH="PRIMER_model"
nohup python primer_hf_main.py  \
                --batch_size 8 \
                --gpus 1  \
                --mode test \
                --model_path ../PRIMER_hf  \
                --dataset_name ${DATA_NAME} \
                --primer_path ../PRIMER_hf \
                --num_workers 0 \
                --progress_bar_refresh_rate 5 \
                --max_length_tgt ${LENGTH_LIMIT} \
        > ../test_${DATA_NAME}_${MODEL_NAME}.out &
