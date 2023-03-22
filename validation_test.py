import os

DATA = "~/efs/data/"
TRAINER = "POMP"
DATASET = "imagenet_21k"
CFG = "vit_b16_ep5_randaug2"  # config file
CTP = "end"  # class token position (end or middle)
NCTX = 4  # number of context tokens
SHOTS = 16  # number of shots (1, 2, 4, 8, 16)
CSC = False  # class-specific context (False or True)
UNC = 1000  # number of update class
SEED = 42
# ID = 'nw8y9kjk'
EPOCH = 5
DIR = f'output/{DATASET}/{TRAINER}/{CFG}_unc{UNC}_{SHOTS}shots/nctx{NCTX}_cscFalse_ctpend/seed{SEED}'
with open(os.path.join(DIR, 'wandb_id.txt'), 'r') as f:
    ID = f.readlines()[0].strip()

# command = f"python train.py \
command = f"python -m torch.distributed.launch --nproc_per_node 8 --master_port 12345 train.py \
      --world-size 8 \
      --root {DATA} \
      --seed {SEED} \
      --trainer {TRAINER} \
      --dataset-config-file configs/datasets/{DATASET}.yaml \
      --config-file configs/trainers/{TRAINER}/{CFG}.yaml \
      --output-dir {DIR} \
      --model-dir {DIR} \
      --load-epoch {EPOCH} \
      --validation-test \
      --wandb_id {ID} \
      DATASET.NUM_SHOTS 1 \
      TRAINER.POMP.N_CTX {NCTX}"
os.system(command)

