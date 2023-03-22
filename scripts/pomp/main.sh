#!/bin/bash

# custom config
DATA=~/efs/data/
TRAINER=POMP

DATASET=$1
CFG=$2  # config file
CTP=$3  # class token position (end or middle)
NCTX=$4  # number of context tokens
SHOTS=$5  # number of shots (1, 2, 4, 8, 16)
CSC=$6  # class-specific context (False or True)
UNC=$7  # number of update class


for SEED in 42
do
    DIR=output/${DATASET}/${TRAINER}/${CFG}_unc${UNC}_${SHOTS}shots/nctx${NCTX}_csc${CSC}_ctp${CTP}/seed${SEED}
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
        TRAINER.POMP.EPT True \
        TRAINER.POMP.N_CTX ${NCTX} \
        TRAINER.POMP.CSC ${CSC} \
        TRAINER.POMP.CLASS_TOKEN_POSITION ${CTP} \
        TRAINER.POMP.UPDATE_CLASS_NUM ${UNC} \
        DATASET.NUM_SHOTS ${SHOTS} \
        TEST.FINAL_MODEL best_val
    fi
done



# sh scripts/pomp/main.sh imagenet_21k vit_b16_ep20_randaug2 end 16 16 False 1000
# sh scripts/pomp/main.sh imagenet_21k vit_b16_ep20_randaug2 end 4 -1 False 1000
# sh scripts/pomp/main.sh imagenet_21k vit_b16_ep20 end 16 16 False 1000

# sh scripts/pomp/main.sh imagenet vit_b16_ep50 end 16 16 False 100