#!/bin/bash

#cd ../..

# custom config
DATA=~/efs/data/
TRAINER=CoOp
SHOTS=16
NCTX=4
CSC=False
CTP=end

DATASET=$1
CFG=$2

for SEED in 1
do
    python train.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
    --output-dir output/evaluation/${TRAINER}/${CFG}_${SHOTS}shots/nctx${NCTX}_csc${CSC}_ctp${CTP}/${DATASET}/seed${SEED} \
    --model-dir output/imagenet/${TRAINER}/${CFG}_${SHOTS}shots/nctx${NCTX}_csc${CSC}_ctp${CTP}/seed${SEED} \
    --load-epoch 50 \
    --eval-only \
    TRAINER.COOP.N_CTX ${NCTX} \
    TRAINER.COOP.CSC ${CSC} \
    TRAINER.COOP.CLASS_TOKEN_POSITION ${CTP}
done

# CUDA_VISIBLE_DEVICES=4 sh scripts/coop/eval.sh oxford_pets vit_b16_ep50
# CUDA_VISIBLE_DEVICES=4 sh scripts/coop/eval.sh oxford_flowers vit_b16_ep50
# CUDA_VISIBLE_DEVICES=4 sh scripts/coop/eval.sh food101 vit_b16_ep50
# CUDA_VISIBLE_DEVICES=4 sh scripts/coop/eval.sh sun397 vit_b16_ep50
# CUDA_VISIBLE_DEVICES=4 sh scripts/coop/eval.sh stanford_cars vit_b16_ep50
# CUDA_VISIBLE_DEVICES=4 sh scripts/coop/eval.sh ucf101 vit_b16_ep50