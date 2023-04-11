#!/bin/bash

#cd ../..

# custom config
DATA=~/efs/data/
TRAINER=CoCoOp

DATASET=$1
SEED=$2

CFG=vit_b16_c4_ep10_batch1_ctxv1
SHOTS=16


DIR=output/evaluation/${TRAINER}/${CFG}_${SHOTS}shots/${DATASET}/seed${SEED}
if [ -d "$DIR" ]; then
    echo "Results are available in ${DIR}. Skip this job"
else
    echo "Run this job and save the output to ${DIR}"

    python train.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
    --output-dir ${DIR} \
    --model-dir output/imagenet/${TRAINER}/${CFG}_${SHOTS}shots/seed${SEED} \
    --load-epoch 10 \
    --eval-only
fi

# CUDA_VISIBLE_DEVICES=0 sh scripts/cocoop/xd_test.sh oxford_pets 1
# CUDA_VISIBLE_DEVICES=1 sh scripts/cocoop/xd_test.sh oxford_flowers 1
# CUDA_VISIBLE_DEVICES=2 sh scripts/cocoop/xd_test.sh food101 1
# CUDA_VISIBLE_DEVICES=3 sh scripts/cocoop/xd_test.sh sun397 1
# CUDA_VISIBLE_DEVICES=4 sh scripts/cocoop/xd_test.sh stanford_cars 1
# CUDA_VISIBLE_DEVICES=5 sh scripts/cocoop/xd_test.sh ucf101 1