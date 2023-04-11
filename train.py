'''
Adapted from https://github.com/KaiyangZhou/CoOp
'''
import argparse
import torch
import torch.distributed as dist
from dassl.utils import setup_logger, set_random_seed, collect_env_info
from dassl.config import get_cfg_default
from dassl.engine import build_trainer
import os
from utils import print_args, is_main_process
# custom
import datasets.oxford_pets
import datasets.oxford_flowers
import datasets.fgvc_aircraft
import datasets.dtd
import datasets.eurosat
import datasets.stanford_cars
import datasets.food101
import datasets.sun397
import datasets.caltech101
import datasets.ucf101
import datasets.imagenet
import datasets.imagenet_21k
import datasets.imagenet_sketch
import datasets.imagenetv2
import datasets.imagenet_a
import datasets.imagenet_r

import trainers.pomp
import trainers.clip_mlp
import trainers.coop
import trainers.cocoop
import trainers.zsclip
import trainers.maple
import trainers.vpt


def reset_cfg(cfg, args):
    if args.root:
        cfg.DATASET.ROOT = args.root

    if args.output_dir:
        cfg.OUTPUT_DIR = args.output_dir

    if args.resume:
        cfg.RESUME = args.resume

    if args.seed:
        cfg.SEED = args.seed

    if args.source_domains:
        cfg.DATASET.SOURCE_DOMAINS = args.source_domains

    if args.target_domains:
        cfg.DATASET.TARGET_DOMAINS = args.target_domains

    if args.transforms:
        cfg.INPUT.TRANSFORMS = args.transforms

    if args.trainer:
        cfg.TRAINER.NAME = args.trainer

    if args.backbone:
        cfg.MODEL.BACKBONE.NAME = args.backbone

    if args.head:
        cfg.MODEL.HEAD.NAME = args.head

    cfg.LOCAL_RANK = args.local_rank
    cfg.WANDB_ID = args.wandb_id


def extend_cfg(cfg):
    """
    Add new config variables.

    E.g.
        from yacs.config import CfgNode as CN
        cfg.TRAINER.MY_MODEL = CN()
        cfg.TRAINER.MY_MODEL.PARAM_A = 1.
        cfg.TRAINER.MY_MODEL.PARAM_B = 0.5
        cfg.TRAINER.MY_MODEL.PARAM_C = False
    """
    from yacs.config import CfgNode as CN

    cfg.DATASET.SUBSAMPLE_CLASSES = "all"  # all, base or new

    # Config for POMP
    cfg.TRAINER.POMP = CN()
    cfg.TRAINER.POMP.N_CTX = 4  # number of context vectors
    cfg.TRAINER.POMP.CSC = False  # class-specific context
    cfg.TRAINER.POMP.CTX_INIT = ""  # initialization words
    cfg.TRAINER.POMP.CLASS_TOKEN_POSITION = "end"  # 'middle' or 'end' or 'front'
    cfg.TRAINER.POMP.UPDATE_CLASS_NUM = 128
    cfg.TRAINER.POMP.PREC = "fp16"  # fp16, fp32, amp
    cfg.TRAINER.POMP.EPT = False
    cfg.TRAINER.POMP.LOCAL_CORRECTION = True
    cfg.TRAINER.POMP.NEG_SAMPLING_MODE = 'random'

    # Config for COOP
    cfg.TRAINER.COOP = CN()
    cfg.TRAINER.COOP.N_CTX = 16  # number of context vectors
    cfg.TRAINER.COOP.CSC = False  # class-specific context
    cfg.TRAINER.COOP.CTX_INIT = ""  # initialization words
    cfg.TRAINER.COOP.PREC = "fp16"  # fp16, fp32, amp
    cfg.TRAINER.COOP.CLASS_TOKEN_POSITION = "end"  # 'middle' or 'end' or 'front'
    cfg.TRAINER.COOP.PRETRAINED_WEOGHT = ''

    # Config for COCOOP
    cfg.TRAINER.COCOOP = CN()
    cfg.TRAINER.COCOOP.N_CTX = 16  # number of context vectors
    cfg.TRAINER.COCOOP.CTX_INIT = ""  # initialization words
    cfg.TRAINER.COCOOP.PREC = "fp16"  # fp16, fp32, amp

    # Config for MaPLe
    cfg.TRAINER.MAPLE = CN()
    cfg.TRAINER.MAPLE.N_CTX = 2  # number of context vectors
    cfg.TRAINER.MAPLE.CTX_INIT = "a photo of a"  # initialization words
    cfg.TRAINER.MAPLE.PREC = "fp16"  # fp16, fp32, amp
    cfg.TRAINER.MAPLE.PROMPT_DEPTH = 9 # Max 12, minimum 0, for 1 it will act as shallow MaPLe (J=1)

    # Config for only vision side prompting
    cfg.TRAINER.VPT = CN()
    cfg.TRAINER.VPT.N_CTX_VISION = 2  # number of context vectors at the vision branch
    cfg.TRAINER.VPT.CTX_INIT = "a photo of a"  # initialization words
    cfg.TRAINER.VPT.PREC = "fp16"  # fp16, fp32, amp
    cfg.TRAINER.VPT.PROMPT_DEPTH_VISION = 1  # if set to 1, will represent shallow vision prompting only

    # Config for MLP & Linear
    cfg.TRAINER.MLP = CN()
    cfg.TRAINER.MLP.PREC = "fp16"  # fp16, fp32, amp

    cfg.TRAINER.U = ""  # pre-computed PCA bases

def setup_cfg(args):
    cfg = get_cfg_default()
    extend_cfg(cfg)

    # 1. From the dataset config file
    if args.dataset_config_file:
        cfg.merge_from_file(args.dataset_config_file)

    # 2. From the method config file
    if args.config_file:
        cfg.merge_from_file(args.config_file)

    # 3. From input arguments
    reset_cfg(cfg, args)

    # 4. From optional input arguments
    cfg.merge_from_list(args.opts)

    cfg.freeze()

    return cfg


def main(args):
    cfg = setup_cfg(args)
    setup_logger(cfg.OUTPUT_DIR)

    if torch.cuda.is_available() and cfg.USE_CUDA:
        torch.backends.cudnn.benchmark = True

    if args.world_size > 0:
        dist.init_process_group(backend="nccl", world_size=args.world_size, rank=args.local_rank)
        torch.cuda.set_device(args.local_rank)

    if is_main_process():
        if cfg.SEED >= 0:
            print("Setting fixed seed: {}".format(cfg.SEED))
            set_random_seed(cfg.SEED)
        print_args(args, cfg)
        print("Collecting env info ...")
        print("** System info **\n{}\n".format(collect_env_info()))

    trainer = build_trainer(cfg)

    if args.eval_only:
        trainer.load_model(args.model_dir, epoch=args.load_epoch)
        trainer.test()
        return

    if args.validation_test:
        val_accs = []
        for epoch in range(1, args.load_epoch + 1):
            trainer.load_model(args.model_dir, epoch=epoch)
            val_acc = trainer.test(split='val')
            val_accs.append(val_acc)
            torch.cuda.empty_cache()

        if args.trainer == 'POMP':
            learner = 'prompt_learner'
        else:
            learner = 'mlp_learner'

        best_checkpoint_id = val_accs.index(max(val_accs)) + 1
        best_checkpoint_file = "model.pth.tar-" + str(best_checkpoint_id)
        best_checkpoint_path = os.path.join(args.model_dir, learner, best_checkpoint_file)
        rename_path = os.path.join(args.model_dir, learner, 'model-best.pth.tar')
        if is_main_process():
            os.system(f"mv {best_checkpoint_path} {rename_path}")
        dist.barrier()

        trainer.load_model(args.model_dir, epoch=None)
        trainer.test(split='test')

        for epoch in range(1, args.load_epoch):  # keep the ckpt of the last epoch
            if epoch != best_checkpoint_id:
                checkpoint_file = "model.pth.tar-" + str(epoch)
                checkpoint_path = os.path.join(args.model_dir, learner, checkpoint_file)
                if is_main_process():
                    os.system(f"rm -rf {checkpoint_path}")
        return
    if not args.no_train:
        if args.model_dir != '':
            trainer.load_model(args.model_dir, epoch=args.load_epoch)
        trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--world-size", type=int, default=0)
    parser.add_argument("--wandb_id", type=str, default=None)
    parser.add_argument("--root", type=str, default="", help="path to dataset")
    parser.add_argument("--output-dir", type=str, default="", help="output directory")
    parser.add_argument(
        "--resume",
        type=str,
        default="",
        help="checkpoint directory (from which the training resumes)",
    )
    parser.add_argument(
        "--seed", type=int, default=-1, help="only positive value enables a fixed seed"
    )
    parser.add_argument(
        "--source-domains", type=str, nargs="+", help="source domains for DA/DG"
    )
    parser.add_argument(
        "--target-domains", type=str, nargs="+", help="target domains for DA/DG"
    )
    parser.add_argument(
        "--transforms", type=str, nargs="+", help="data augmentation methods"
    )
    parser.add_argument(
        "--config-file", type=str, default="", help="path to config file"
    )
    parser.add_argument(
        "--dataset-config-file",
        type=str,
        default="",
        help="path to config file for dataset setup",
    )
    parser.add_argument("--trainer", type=str, default="", help="name of trainer")
    parser.add_argument("--backbone", type=str, default="", help="name of CNN backbone")
    parser.add_argument("--head", type=str, default="", help="name of head")
    parser.add_argument("--eval-only", action="store_true", help="evaluation only")
    parser.add_argument("--validation-test", action="store_true", help="validation and test")
    parser.add_argument(
        "--model-dir",
        type=str,
        default="",
        help="load model from this directory for eval-only mode",
    )
    parser.add_argument(
        "--load-epoch", type=int, help="load model weights at this epoch for evaluation"
    )
    parser.add_argument(
        "--no-train", action="store_true", help="do not call trainer.train()"
    )
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="modify config options using the command-line",
    )
    args = parser.parse_args()
    main(args)
