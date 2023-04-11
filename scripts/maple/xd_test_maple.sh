#!/bin/bash

#cd ../..

# custom config
#DATA=~/efs/imagenet/
DATA=~/efs/CoOp/data/
TRAINER=MaPLe

DATASET=$1
SEED=$2

CFG=vit_b16_c2_ep5_batch4_2ctx_cross_datasets
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
    --load-epoch 2 \
    --eval-only
fi

# CUDA_VISIBLE_DEVICES=0 sh scripts/maple/xd_test_maple.sh oxford_pets 1
# CUDA_VISIBLE_DEVICES=1 sh scripts/maple/xd_test_maple.sh oxford_flowers 1
# CUDA_VISIBLE_DEVICES=2 sh scripts/maple/xd_test_maple.sh food101 1
# CUDA_VISIBLE_DEVICES=3 sh scripts/maple/xd_test_maple.sh sun397 1
# CUDA_VISIBLE_DEVICES=4 sh scripts/maple/xd_test_maple.sh stanford_cars 1
# CUDA_VISIBLE_DEVICES=5 sh scripts/maple/xd_test_maple.sh ucf101 1
# CUDA_VISIBLE_DEVICES=6 sh scripts/maple/xd_test_maple.sh eurosat 1
# CUDA_VISIBLE_DEVICES=7 sh scripts/maple/xd_test_maple.sh fgvc_aircraft 1
# CUDA_VISIBLE_DEVICES=4 sh scripts/maple/xd_test_maple.sh caltech101 1
# CUDA_VISIBLE_DEVICES=4 sh scripts/maple/xd_test_maple.sh dtd 1
# CUDA_VISIBLE_DEVICES=7 sh scripts/maple/xd_test_maple.sh imagenet_a 1
# CUDA_VISIBLE_DEVICES=7 sh scripts/maple/xd_test_maple.sh imagenet_r 1
# CUDA_VISIBLE_DEVICES=7 sh scripts/maple/xd_test_maple.sh imagenet_sketch 1
# CUDA_VISIBLE_DEVICES=7 sh scripts/maple/xd_test_maple.sh imagenetv2 1