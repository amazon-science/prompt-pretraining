#!/bin/bash

# custom config
DATA=~/efs/data/
TRAINER=POMP
DATASET=$1
SEED=$2
UNC=$3  # number of update class
EPOCH=$4

CFG=vit_b16_ep5_randaug2
SHOTS=16
NCTX=4

if [ ! $EPOCH ]; then
  DIR=output/evaluation/${TRAINER}/${CFG}_unc${UNC}_${SHOTS}shots_nctx${NCTX}_best_val_${DATASET}/seed${SEED}
  if [ -d "$DIR" ]; then
      echo "Oops! The results exist at ${DIR} (so skip this job)"
  else
      python -m torch.distributed.launch --nproc_per_node 8 --master_port 12346 train.py \
      --world-size 8 \
      --root ${DATA} \
      --seed ${SEED} \
      --trainer ${TRAINER} \
      --dataset-config-file configs/datasets/${DATASET}.yaml \
      --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
      --output-dir ${DIR} \
      --model-dir output/imagenet_21k/${TRAINER}/${CFG}_unc${UNC}_${SHOTS}shots/nctx${NCTX}_cscFalse_ctpend/seed${SEED} \
      --eval-only \
      TRAINER.POMP.N_CTX ${NCTX}
  fi
else
  DIR=output/evaluation/${TRAINER}/${CFG}_unc${UNC}_${SHOTS}shots_nctx${NCTX}_epoch${EPOCH}_${DATASET}/seed${SEED}
  if [ -d "$DIR" ]; then
      echo "Oops! The results exist at ${DIR} (so skip this job)"
  else
      python -m torch.distributed.launch --nproc_per_node 8 --master_port 12346 train.py \
      --world-size 8 \
      --root ${DATA} \
      --seed ${SEED} \
      --trainer ${TRAINER} \
      --dataset-config-file configs/datasets/${DATASET}.yaml \
      --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
      --output-dir ${DIR} \
      --model-dir output/imagenet_21k/${TRAINER}/${CFG}_unc${UNC}_${SHOTS}shots/nctx${NCTX}_cscFalse_ctpend/seed${SEED} \
      --load-epoch ${EPOCH} \
      --eval-only \
      TRAINER.POMP.N_CTX ${NCTX}
  fi
fi
