'''
Adapted from https://github.com/KaiyangZhou/CoOp
'''
import os.path as osp
import time
import datetime
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast
from dassl.data.data_manager import DataManager
from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, load_checkpoint, MetricMeter, AverageMeter, save_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler
from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from tqdm import tqdm
import torch.distributed as dist
import wandb

_tokenizer = _Tokenizer()


def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    design_details = {"trainer": 'MLP',
                      "vision_depth": 0,
                      "language_depth": 0, "vision_ctx": 0,
                      "language_ctx": 0}
    model = clip.build_model(state_dict or model.state_dict(), design_details)

    return model


class MLPLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        image_proj_dim = clip_model.text_projection.size(-1)
        hidden_dim = 1000
        n_class = len(classnames)
        # self.fc1 = nn.Linear(image_proj_dim, hidden_dim).half()
        # self.relu = torch.nn.ReLU()
        # self.fc2 = nn.Linear(hidden_dim, n_class).half()
        self.fc = nn.Linear(image_proj_dim, n_class).half()

    def forward(self, x):
        # hidden = self.fc1(x)
        # relu = self.relu(hidden)
        # output = self.fc2(relu)
        output = self.fc(x)
        return output


class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.mlp_learner = MLPLearner(cfg, classnames, clip_model)
        self.dtype = clip_model.dtype

        self.image_encoder = clip_model.visual

        self.logit_scale = clip_model.logit_scale

    def forward(self, image):
        image_features = self.image_encoder(image.type(self.dtype))
        # image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        logits = self.mlp_learner(image_features)

        return logits


@TRAINER_REGISTRY.register()
class MLP(TrainerX):
    """Context Optimization (MLP).

    Learning to Prompt for Vision-Language Models
    https://arxiv.org/abs/2109.01134
    """

    def check_cfg(self, cfg):
        assert cfg.TRAINER.MLP.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        train_classnames = self.dm.dataset.classnames
        test_lab2cname, test_classnames = self.dm.dataset.get_lab2cname(self.dm.dataset.test)
        self.test_lab2cname = test_lab2cname

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)

        if cfg.TRAINER.MLP.PREC == "fp32" or cfg.TRAINER.MLP.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()

        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, test_classnames, clip_model)

        print("Turning off gradients in both the image and the text encoder")
        for name, param in self.model.named_parameters():
            if "mlp_learner" not in name:
                param.requires_grad_(False)

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model.mlp_learner, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)

        # Note that multi-gpu training could be slow because CLIP's size is
        # big, which slows down the copy operation in DataParallel
        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            # self.model = nn.DataParallel(self.model)
            if 'ViT' in cfg.MODEL.BACKBONE.NAME:
                self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model)  # reduce accuracy for RN50
            self.model = nn.parallel.DistributedDataParallel(self.model, device_ids=[cfg.LOCAL_RANK])

            # NOTE: only give mlp_learner to the optimizer
            self.optim = build_optimizer(self.model.module.mlp_learner, cfg.OPTIM)
            self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
            self.register_model("mlp_learner", self.model.module.mlp_learner, self.optim, self.sched)
        else:
            # NOTE: only give mlp_learner to the optimizer
            self.optim = build_optimizer(self.model.mlp_learner, cfg.OPTIM)
            self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
            self.register_model("mlp_learner", self.model.mlp_learner, self.optim, self.sched)

        self.scaler = GradScaler() if cfg.TRAINER.MLP.PREC == "amp" else None

        if dist.is_initialized():
            if dist.get_rank() == 0:
                self.record_node = True
            else:
                self.record_node = False
        else:
            self.record_node = True

        if self.record_node is True:
            wandb.init(project='MLP-%s' % cfg.DATASET.NAME, resume="allow", id=cfg.WANDB_ID,
                       name=(('-'.join(cfg.OUTPUT_DIR.split('/')[2:])).replace('MLP_', '')))
        # dist.barrier()

    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)

        prec = self.cfg.TRAINER.MLP.PREC

        if prec == "amp":
            with autocast():
                output = self.model(image)
                loss = F.cross_entropy(output, label)
            self.optim.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()
        else:
            output = self.model(image)
            loss = F.cross_entropy(output, label)
            self.model_backward_and_update(loss)

        loss_summary = {
            "loss": loss.item(),
            "acc": compute_accuracy(output, label)[0].item(),
        }

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    @torch.no_grad()
    def test(self, split=None):
        def distributed_concat(tensor, num_total_examples):
            output_tensors = [tensor.clone() for _ in range(torch.distributed.get_world_size())]
            torch.distributed.all_gather(output_tensors, tensor)
            concat = torch.cat(output_tensors, dim=0)
            # truncate the dummy elements added by SequentialDistributedSampler
            return concat[:num_total_examples]

        """A generic testing pipeline."""
        self.set_model_mode('eval')
        self.evaluator.reset()
        self.evaluator._lab2cname = self.test_lab2cname

        if split is None:
            split = self.cfg.TEST.SPLIT

        if split == 'val' and self.val_loader is not None:
            data_loader = self.val_loader
            print('Do evaluation on {} set'.format(split))
        else:
            data_loader = self.test_loader
            print('Do evaluation on test set')

        # for batch_idx, batch in tqdm(enumerate(data_loader)):
        #     input, label = self.parse_batch_test(batch)
        #     output = self.model_inference(input)
        #
        #     self.evaluator.process(output, label)

        ### for ddp inference
        outputs, labels = [], []
        for batch_idx, batch in tqdm(enumerate(data_loader)):
            input, label = self.parse_batch_test(batch)
            output = self.model_inference(input)
            outputs.append(output)
            labels.append(label)
        output = distributed_concat(torch.cat(outputs, dim=0), len(self.dm.dataset.test))
        label = distributed_concat(torch.cat(labels, dim=0), len(self.dm.dataset.test))
        self.evaluator.process(output, label)

        results = self.evaluator.evaluate()

        if self.record_node is True:
            for k, v in results.items():
                tag = '{}/{}'.format(split, k)
                self.write_scalar(tag, v, self.epoch)
                wandb.log({tag: v})
        dist.barrier()
        torch.cuda.empty_cache()

        return list(results.values())[0]

    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        if isinstance(input, list):  # with data augmentation
            input = torch.cat(input)
            label = label.repeat(self.cfg.DATALOADER.K_TRANSFORMS)
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label

    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()

        # By default, the best model is loaded
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))

            # checkpoint = load_checkpoint(model_path)
            map_location = 'cpu'
            checkpoint = torch.load(model_path, map_location=map_location)

            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            # Ignore fixed token vectors
            if "token_prefix" in state_dict:
                del state_dict["token_prefix"]

            if "token_suffix" in state_dict:
                del state_dict["token_suffix"]

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)

    def before_epoch(self):
        self.train_loader_x.sampler.set_epoch(self.epoch)

    def run_epoch(self):
        self.set_model_mode('train')
        losses = MetricMeter()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        self.num_batches = len(self.train_loader_x)

        end = time.time()
        for self.batch_idx, batch in enumerate(self.train_loader_x):
            data_time.update(time.time() - end)
            loss_summary = self.forward_backward(batch)
            batch_time.update(time.time() - end)
            losses.update(loss_summary)

            if (self.batch_idx + 1) % self.cfg.TRAIN.PRINT_FREQ == 0:
                nb_this_epoch = self.num_batches - (self.batch_idx + 1)
                nb_future_epochs = (
                    self.max_epoch - (self.epoch + 1)
                ) * self.num_batches
                eta_seconds = batch_time.avg * (nb_this_epoch+nb_future_epochs)
                eta = str(datetime.timedelta(seconds=int(eta_seconds)))
                print(
                    'epoch [{0}/{1}][{2}/{3}]\t'
                    'time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    'eta {eta}\t'
                    '{losses}\t'
                    'lr {lr}'.format(
                        self.epoch + 1,
                        self.max_epoch,
                        self.batch_idx + 1,
                        self.num_batches,
                        batch_time=batch_time,
                        data_time=data_time,
                        eta=eta,
                        losses=losses,
                        lr=self.get_current_lr()
                    )
                )

            n_iter = self.epoch * self.num_batches + self.batch_idx

            if self.record_node is True:
                for name, meter in losses.meters.items():
                    self.write_scalar('train/' + name, meter.avg, n_iter)
                    wandb.log({'train/' + name: meter.avg}, step=n_iter)
                self.write_scalar('train/lr', self.get_current_lr(), n_iter)
                wandb.log({'train/lr': self.get_current_lr()}, step=n_iter)
            dist.barrier()

            end = time.time()

    def save_model(self, epoch, directory, is_best=False, val_result=None, model_name=''):
        names = self.get_model_names()

        if self.record_node is True:
            for name in names:
                model_dict = self._models[name].state_dict()

                optim_dict = None
                if self._optims[name] is not None:
                    optim_dict = self._optims[name].state_dict()

                sched_dict = None
                if self._scheds[name] is not None:
                    sched_dict = self._scheds[name].state_dict()

                save_checkpoint(
                    {
                        'state_dict': model_dict,
                        'epoch': epoch + 1,
                        'optimizer': optim_dict,
                        'scheduler': sched_dict,
                        "val_result": val_result
                    },
                    osp.join(directory, name),
                    is_best=is_best,
                    model_name=model_name
                )
        dist.barrier()
