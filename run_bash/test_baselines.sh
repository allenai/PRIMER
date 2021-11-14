#!/bin/bash

cd ../script

# BART MODEL
# MODEL_NAME="facebook/bart-large"
# MODEL_SAVE_NAME="bart_large_1024_1024"
# LENGTH_TGT=1024
# LENGTH_INPUT=1024

# PEGASUS
# MODEL_NAME="google/pegasus-large"
# MODEL_SAVE_NAME="pegasus_large_512_256"
# LENGTH_TGT=256
# LENGTH_INPUT=512

# LED
MODEL_NAME="allenai/led-large-16384"
MODEL_SAVE_NAME="led_large_orig"
LENGTH_TGT=1024
LENGTH_INPUT=4096

# Multi-news dataset
DATA_NAME="multi_news"
# Multi-xscience dataset
# DATA_NAME="multi_x_science_sum"
# Wikisum dataset
# DATA_NAME="wikisum"
# wcep dataset
# DATA_NAME="wcep"
# arxiv dataset
# DATA_NAME="arxiv"

nohup python pegasus_main.py  \
                        --batch_size 4 \
                        --gpus 1  \
                        --mode test \
                        --model_name ${MODEL_NAME} \
                        --model_path ../models/${MODEL_SAVE_NAME}/ \
                        --dataset_name ${DATA_NAME} \
                        --pretrained_model_path ../pretrained_models \
                        --num_workers 0 \
                        --progress_bar_refresh_rate 50 \
                        --data_path ${DATA_PATH} \
                        --max_length_tgt ${LENGTH_TGT} \
                        --max_length_input ${LENGTH_INPUT} \
                > ../test_${DATA_NAME}_${MODEL_SAVE_NAME}.out &

