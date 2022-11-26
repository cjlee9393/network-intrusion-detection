#!/bin/bash

FILE_NAME="code_cleaning"
BATCH_SIZE=64
SEQ_LEN=120
BINARY=1
MULTI_TO_BINARY=1
MANY_TRAIN=0

VALID_RATIO=0.1
HIDDEN_SIZE=100
NUM_LAYERS=1
LR=0.001
MAX_PATIENCE=5
MAX_EPOCH=10000
TRY_ENTRIES=5
DATASET="unsw"
MODEL="rnn"
PATH=''
OPTIMIZER=0
DROPOUT=0.5

/usr/bin/python3.8 model.py \
    --file_name=$FILE_NAME \
    --batch_size=$BATCH_SIZE \
    --seq_len=$SEQ_LEN \
    --binary=$BINARY \
    --multi_to_binary=$MULTI_TO_BINARY \
    --many_train=$MANY_TRAIN \
    --valid_ratio=$VALID_RATIO \
    --hidden_size=$HIDDEN_SIZE \
    --num_layers=$NUM_LAYERS \
    --lr=$LR \
    --max_patience=$MAX_PATIENCE \
    --max_epoch=$MAX_EPOCH \
    --try_entries=$TRY_ENTRIES \
    --dataset=$DATASET \
    --model=$MODEL \
    --path=$PATH \
    --optimizer=$OPTIMIZER \
    --dropout=$DROPOUT