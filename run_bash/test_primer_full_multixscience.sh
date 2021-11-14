#!/bin/bash

cd ../script

# Multi-xscience dataset
DATA_NAME="multi_x_science_sum"


# Fully supervised PRIMER model on multi_x_science
MODEL_NAME="PRIMER_multixscience"
MODEL_PATH="PRIMER_multixscience"
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
        > ../test_${DATA_NAME}_${MODEL_NAME}.out &
