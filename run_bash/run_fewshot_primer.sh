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

#fewshot training for 10 examples
NUM_TRAIN_DATA=10

for RAND_SEED in 1111 1234 5555 6789 7362
do
nohup python primer_main.py  \
                --gpus 1  \
                --mode train \
                --lr 3e-5 \
                --label_smoothing 0.1 \
                --accum_data_per_step 10 \
                --warmup_steps 20 \
                --total_steps 200 \
                --batch_size 2 \
                --model_path ../models/$MODEL_NAME/  \
                --dataset_name ${DATA_NAME} \
                --primer_path ../${MODEL_PATH} \
                --num_workers 0 \
                --progress_bar_refresh_rate 50 \
                --rand_seed ${RAND_SEED} \
                --saveTopK 3 \
                --test_imediate \
                --test_batch_size 8 \
                --fewshot \
                --grad_ckpt \
        > ../fewshot_${DATA_NAME}_${MODEL_NAME}_${RAND_SEED}.out &
done



#fewshot training for 100 examples
NUM_TRAIN_DATA=100

for RAND_SEED in 1111 1234 5555 6789 7362
do
nohup python primer_main.py  \
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
                --num_workers 0 \
                --progress_bar_refresh_rate 50 \
                --rand_seed ${RAND_SEED} \
                --saveTopK 3 \
                --test_imediate \
                --test_batch_size 8 \
                --fewshot \
                --grad_ckpt \
        > ../fewshot_${DATA_NAME}_${MODEL_NAME}_${RAND_SEED}.out &
done
