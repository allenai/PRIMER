#!/bin/bash

cd ../script

# wcep dataset
DATA_NAME="wcep"


# Fully supervised PRIMER model on wcep
MODEL_NAME="PRIMER_wcep"
MODEL_PATH="PRIMER_wcep"
nohup python primer_main.py  \
                --batch_size 4 \
                --gpus 1  \
                --mode test \
                --model_path ../models/$MODEL_NAME/  \
                --resume_ckpt /content/drive/MyDrive/MDS_PRIMER/primer/models/PRIMER_wcep/ckpt_bs3/ckpt.ckpt \
                --dataset_name ${DATA_NAME} \
                --primer_path ../${MODEL_PATH} \
                --num_workers 0 \
                --progress_bar_refresh_rate 1 \
                --beam_size 5 \
        > ../test_${DATA_NAME}_${MODEL_NAME}.out 2>&1
