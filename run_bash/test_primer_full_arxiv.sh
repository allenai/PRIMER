#!/bin/bash

cd ../script

# arxiv dataset
DATA_NAME="arxiv"


# Fully supervised PRIMER model on arxiv
MODEL_NAME="PRIMER_arxiv"
MODEL_PATH="PRIMER_arxiv"
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

