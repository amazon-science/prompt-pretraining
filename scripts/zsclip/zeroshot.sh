#!/bin/bash

#cd ../..

# custom config
DATA=~/efs/data/
TRAINER=ZeroshotCLIP
DATASET=$1
CFG=$2  # rn50, rn101, vit_b32 or vit_b16 vit_l14

python train.py \
--root ${DATA} \
--trainer ${TRAINER} \
--dataset-config-file configs/datasets/${DATASET}.yaml \
--config-file configs/trainers/CoOp/${CFG}.yaml \
--output-dir output/${TRAINER}/${CFG}/${DATASET} \
--eval-only

# CUDA_VISIBLE_DEVICES=0 sh scripts/zsclip/zeroshot.sh imagenet_21k rn50
# CUDA_VISIBLE_DEVICES=0 sh scripts/zsclip/zeroshot.sh oxford_pets vit_l14