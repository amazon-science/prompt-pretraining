#!/bin/bash

# custom config
DATA=~/efs/data/
TRAINER=MLP

DATASET=$1
CFG=$2  # config file
SHOTS=$3  # number of shots (1, 2, 4, 8, 16)


for SEED in 42
do
    DIR=output/${DATASET}/${TRAINER}/${CFG}_${SHOTS}shots/seed${SEED}
    if [ -d "$DIR" ]; then
        echo "Results are available in ${DIR}. Skip this job"
    else
        echo "Run this job and save the output to ${DIR}"
        python -m torch.distributed.launch --nproc_per_node 8 --master_port 12345 train.py \
        --world-size 8 \
        --root ${DATA} \
        --seed ${SEED} \
        --trainer ${TRAINER} \
        --dataset-config-file configs/datasets/${DATASET}.yaml \
        --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
        --output-dir ${DIR} \
        DATASET.NUM_SHOTS ${SHOTS} \
        TEST.FINAL_MODEL best_val
    fi
done


# sh scripts/mlp/main.sh imagenet_21k rn50_ep20 16
# sh scripts/mlp/main.sh imagenet_21k vit_b32_ep20 16
# sh scripts/mlp/main.sh imagenet_21k vit_b16_ep20 16