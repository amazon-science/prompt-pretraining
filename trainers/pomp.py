'''
Adapted from https://github.com/KaiyangZhou/CoOp
'''
import os.path as osp
import random
import time
import datetime
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast
import torch.distributed as dist
from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, load_checkpoint, MetricMeter, AverageMeter, save_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler
import wandb
from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from clip.simple_tokenizer import default_bpe, bytes_to_unicode
from collections import OrderedDict, defaultdict
import numpy as np
from tqdm import tqdm
from align_uniform import align_loss, uniform_loss
from torch.utils.data.distributed import DistributedSampler
from utils import is_main_process


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

    design_details = {"trainer": 'POMP',
                      "vision_depth": 0,
                      "language_depth": 0, "vision_ctx": 0,
                      "language_ctx": 0}
    model = clip.build_model(state_dict or model.state_dict(), design_details)

    return model


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


class PromptLearner(nn.Module):
    def __init__(self, cfg, train_classnames, test_classnames, clip_model):
        super().__init__()
        train_n_cls = len(train_classnames)
        test_n_cls = len(test_classnames)
        n_ctx = cfg.TRAINER.POMP.N_CTX
        ctx_init = cfg.TRAINER.POMP.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        if ctx_init:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init)
            # n_ctx = torch.nonzero(prompt).size(0)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1: 1 + n_ctx, :]
            prompt_prefix = ctx_init
        else:
            # random initialization
            if cfg.TRAINER.POMP.CSC:
                print("Initializing class-specific contexts")
                ctx_vectors = torch.empty(train_n_cls, n_ctx, ctx_dim, dtype=dtype)
            else:
                print("Initializing a generic context")
                ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        if is_main_process():
            print(f'Initial context: "{prompt_prefix}"')
            print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)  # to be optimized by gradient

        train_classnames = [name.replace("_", " ") for name in train_classnames]
        train_name_lens = [len(_tokenizer.encode(name)) for name in train_classnames]
        train_prompts = [prompt_prefix + " " + name + "." for name in train_classnames]

        test_classnames = [name.replace("_", " ") for name in test_classnames]
        test_name_lens = [len(_tokenizer.encode(name)) for name in test_classnames]
        test_prompts = [prompt_prefix + " " + name + "." for name in test_classnames]

        train_tokenized_prompts = torch.cat([clip.tokenize(p) for p in train_prompts])
        test_tokenized_prompts = torch.cat([clip.tokenize(p) for p in test_prompts])
        with torch.no_grad():
            train_embedding = clip_model.token_embedding(train_tokenized_prompts).type(dtype)
            test_embedding = clip_model.token_embedding(test_tokenized_prompts).type(dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("train_token_prefix", train_embedding[:, :1, :])  # SOS
        self.register_buffer("train_token_suffix", train_embedding[:, 1 + n_ctx:, :])  # CLS, EOS
        self.register_buffer("test_token_prefix", test_embedding[:, :1, :])  # SOS
        self.register_buffer("test_token_suffix", test_embedding[:, 1 + n_ctx:, :])  # CLS, EOS

        self.train_n_cls = train_n_cls
        self.test_n_cls = test_n_cls
        self.n_upd_cls = cfg.TRAINER.POMP.UPDATE_CLASS_NUM
        self.n_ctx = n_ctx
        self.train_tokenized_prompts = train_tokenized_prompts  # torch.Tensor
        self.test_tokenized_prompts = test_tokenized_prompts  # torch.Tensor
        self.train_name_lens = train_name_lens
        self.class_token_position = cfg.TRAINER.POMP.CLASS_TOKEN_POSITION

    def forward(self, idx=None):
        ctx = self.ctx
        if self.training:
            prefix = self.train_token_prefix
            suffix = self.train_token_suffix
            if idx is not None:
                prefix, suffix = prefix[idx], suffix[idx]

            if ctx.dim() == 2:
                if idx is not None:
                    n = idx.size(0)
                    ctx = ctx.unsqueeze(0).expand(n, -1, -1)
                else:
                    ctx = ctx.unsqueeze(0).expand(self.train_n_cls, -1, -1)
            else:  # class-specific contexts
                if idx is not None:
                    ctx = ctx[idx]
        else:
            prefix = self.test_token_prefix
            suffix = self.test_token_suffix
            if ctx.dim() == 2:
                ctx = ctx.unsqueeze(0).expand(self.test_n_cls, -1, -1)

        if self.class_token_position == "end":
            prompts = torch.cat(
                [
                    prefix,  # (n_cls, 1, dim)
                    ctx,  # (n_cls, n_ctx, dim)
                    suffix,  # (n_cls, *, dim)
                ],
                dim=1,
            )
        elif self.class_token_position == "middle":
            half_n_ctx = self.n_ctx // 2
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i: i + 1, :, :]
                class_i = suffix[i: i + 1, :name_len, :]
                suffix_i = suffix[i: i + 1, name_len:, :]
                ctx_i_half1 = ctx[i: i + 1, :half_n_ctx, :]
                ctx_i_half2 = ctx[i: i + 1, half_n_ctx:, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        ctx_i_half1,  # (1, n_ctx//2, dim)
                        class_i,  # (1, name_len, dim)
                        ctx_i_half2,  # (1, n_ctx//2, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)
        elif self.class_token_position == "front":
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i: i + 1, :, :]
                class_i = suffix[i: i + 1, :name_len, :]
                suffix_i = suffix[i: i + 1, name_len:, :]
                ctx_i = ctx[i: i + 1, :, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        class_i,  # (1, name_len, dim)
                        ctx_i,  # (1, n_ctx, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)
        else:
            raise ValueError

        return prompts


class CustomCLIP(nn.Module):
    def __init__(self, cfg, train_classnames, test_classnames, clip_model):
        super().__init__()
        self.cfg = cfg
        self.prompt_learner = PromptLearner(cfg, train_classnames, test_classnames, clip_model)
        self.train_tokenized_prompts = self.prompt_learner.train_tokenized_prompts
        self.test_tokenized_prompts = self.prompt_learner.test_tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.text_features = None

    def forward(self, image, label_ib=None):
        logit_scale = self.logit_scale.exp()

        # compute query features
        image_features = self.image_encoder(image.type(self.dtype))
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        if self.prompt_learner.training:
            if self.cfg.TRAINER.POMP.EPT:
                assert label_ib is not None
                # negative sampling, ib: in batch, oob: out of batch
                label_all = torch.arange(0, self.prompt_learner.train_n_cls).to(label_ib)
                num_oob = self.prompt_learner.n_upd_cls - 1
                label_oob, prob = self.negative_sampling(label_all, image_features.clone().detach(), num_oob, self.cfg.TRAINER.POMP.NEG_SAMPLING_MODE)

                label_upd = torch.cat([label_ib, label_oob])

                prompts_upd = self.prompt_learner(label_upd)
                tokenized_prompts_upd = self.train_tokenized_prompts[label_upd]
                text_features_upd = self.text_encoder(prompts_upd, tokenized_prompts_upd)
                text_features_upd = text_features_upd / text_features_upd.norm(dim=-1, keepdim=True)

                text_features_pos = text_features_upd[:label_ib.size(0)]
                logits_pos = logit_scale * (image_features * text_features_pos).sum(dim=-1, keepdims=True)

                text_features_neg = text_features_upd[-label_oob.size(0):]
                logits_neg = logit_scale * image_features @ text_features_neg.t()
                bsz = image.size(0)
                reject_labels = label_ib.unsqueeze(1) == label_oob.unsqueeze(0).expand(bsz, -1)
                logits_neg -= 1e6 * reject_labels

                if self.cfg.TRAINER.POMP.LOCAL_CORRECTION:
                    if self.cfg.TRAINER.POMP.NEG_SAMPLING_MODE == 'random':
                        logits_neg += torch.log10(torch.tensor(self.prompt_learner.train_n_cls - 1).to(logits_neg))
                        true_label_oob_num = torch.tensor([[num_oob]] * bsz).to(logits_neg) - \
                                             reject_labels.sum(-1, keepdims=True).to(logits_neg)
                        logits_neg -= torch.log10(true_label_oob_num)
                    else:
                        logits_neg -= torch.log10(prob).to(logits_neg)

                logits = torch.cat([logits_pos, logits_neg], dim=1)
            else:
                prompts = self.prompt_learner()
                tokenized_prompts = self.test_tokenized_prompts
                text_features = self.text_encoder(prompts, tokenized_prompts)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)

                logits = logit_scale * image_features @ text_features.t()
        else:  # val and test
            logits = logit_scale * image_features @ self.text_features.t()
        return logits

    @torch.no_grad()
    def negative_sampling(self, neg_index, image_features, sampled_neg_num, mode='random'):
        if mode == 'random':
            perm = torch.randperm(neg_index.size(0))
            k_neg_index = perm[:sampled_neg_num]
            sampled_neg_index = neg_index[k_neg_index]
            prob = [sampled_neg_num / neg_index.size(0)] * sampled_neg_num
            prob = torch.tensor(prob)
        else:
            neg_text_features = self.memory_bank.clone().detach()[neg_index]
            # prompts = self.prompt_learner()
            # neg_text_features = self.text_encoder(prompts, self.test_tokenized_prompts)[neg_index]
            neg_text_features = neg_text_features / neg_text_features.norm(dim=-1, keepdim=True)
            neg_logits = self.logit_scale.exp() * image_features @ neg_text_features.t()
            neg_logits = neg_logits.mean(dim=0)
            if mode == 'easy':
                neg_logits = -neg_logits
            prob = neg_logits.cpu().tolist()
            temp = 1
            prob = np.exp([p / temp for p in prob]) / sum(np.exp([p / temp for p in prob]))  # softmax
            topk_neg_index = np.random.choice(list(range(neg_index.size(0))), size=sampled_neg_num, p=prob)
            topk_neg_index = torch.tensor(topk_neg_index)
            # _, topk_neg_index = neg_logits.topk(sampled_neg_num)  # index to neg_index
            sampled_neg_index = neg_index[topk_neg_index]
            prob = torch.tensor(prob[topk_neg_index])
        return sampled_neg_index, prob


@TRAINER_REGISTRY.register()
class POMP(TrainerX):
    """Context Optimization (POMP).
    Learning to Prompt for Vision-Language Models
    https://arxiv.org/abs/2109.01134
    """

    def check_cfg(self, cfg):
        assert cfg.TRAINER.POMP.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        train_classnames = self.dm.dataset.classnames
        test_lab2cname, test_classnames = self.dm.dataset.get_lab2cname(self.dm.dataset.test)
        self.test_lab2cname = test_lab2cname

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)

        if cfg.TRAINER.POMP.PREC == "fp32" or cfg.TRAINER.POMP.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()

        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, train_classnames, test_classnames, clip_model)

        print("Turning off gradients in both the image and the text encoder")
        for name, param in self.model.named_parameters():
            if "prompt_learner" not in name:
                param.requires_grad_(False)

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model.prompt_learner, cfg.MODEL.INIT_WEIGHTS)

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

            # NOTE: only give prompt_learner to the optimizer
            self.optim = build_optimizer(self.model.module.prompt_learner, cfg.OPTIM)
            self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
            self.register_model("prompt_learner", self.model.module.prompt_learner, self.optim, self.sched)
        else:
            # NOTE: only give prompt_learner to the optimizer
            self.optim = build_optimizer(self.model.prompt_learner, cfg.OPTIM)
            self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
            self.register_model("prompt_learner", self.model.prompt_learner, self.optim, self.sched)

        self.scaler = GradScaler() if cfg.TRAINER.POMP.PREC == "amp" else None

        if is_main_process():
            wandb.init(project='POMP-%s' % cfg.DATASET.NAME, resume="allow", id=cfg.WANDB_ID,
                       name=(('-'.join(cfg.OUTPUT_DIR.split('/')[2:])).replace('GPT_', '')))
            with open(osp.join(cfg.OUTPUT_DIR, 'wandb_id.txt'), 'w') as f:
                f.write(wandb.run.id)
        # dist.barrier()

    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)

        prec = self.cfg.TRAINER.POMP.PREC

        if prec == "amp":
            with autocast():
                output = self.model(image, label)
                label = torch.zeros_like(label).to(label)
                loss = F.cross_entropy(output, label)
            self.optim.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()
        else:
            output = self.model(image, label)
            if self.cfg.TRAINER.POMP.EPT:
                new_label = torch.zeros_like(label).to(label)
            else:
                new_label = label
            loss = F.cross_entropy(output, new_label)
            # loss = self.loss_fn(output, label)
            self.model_backward_and_update(loss)

        loss_summary = {
            "loss": loss.item(),
            "acc": compute_accuracy(output, new_label)[0].item(),
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

        if isinstance(self.model, nn.parallel.DistributedDataParallel):
            prompts = self.model.module.prompt_learner()
            tokenized_prompts = self.model.module.test_tokenized_prompts
            text_features = self.model.module.text_encoder(prompts, tokenized_prompts)
            self.model.module.text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        else:
            prompts = self.model.prompt_learner()
            tokenized_prompts = self.model.test_tokenized_prompts
            text_features = self.model.text_encoder(prompts, tokenized_prompts)
            self.model.text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # for batch_idx, batch in tqdm(enumerate(data_loader)):
        #     input, label = self.parse_batch_test(batch)
        #     output = self.model_inference(input)
        #
        #     self.evaluator.process(output, label)

        # for ddp inference
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

        if is_main_process():
            for k, v in results.items():
                tag = '{}/{}'.format(split, k)
                self.write_scalar(tag, v, self.epoch)
                wandb.log({tag: v})
        dist.barrier()

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

            map_location = 'cpu'
            checkpoint = torch.load(model_path, map_location=map_location)

            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            # Ignore fixed token vectors
            if "token_prefix" in state_dict:
                del state_dict["token_prefix"]
            if "token_suffix" in state_dict:
                del state_dict["token_suffix"]
            if "train_token_prefix" in state_dict:
                del state_dict["train_token_prefix"]
            if "train_token_suffix" in state_dict:
                del state_dict["train_token_suffix"]
            if "test_token_prefix" in state_dict:
                del state_dict["test_token_prefix"]
            if "test_token_suffix" in state_dict:
                del state_dict["test_token_suffix"]

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)

    def before_epoch(self):
        if isinstance(self.train_loader_x.sampler, DistributedSampler):
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
                eta_seconds = batch_time.avg * (nb_this_epoch + nb_future_epochs)
                eta = str(datetime.timedelta(seconds=int(eta_seconds)))
                if is_main_process():
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

            if is_main_process():
                for name, meter in losses.meters.items():
                    self.write_scalar('train/' + name, meter.avg, n_iter)
                    wandb.log({'train/' + name: meter.avg}, step=n_iter)
                self.write_scalar('train/lr', self.get_current_lr(), n_iter)
                wandb.log({'train/lr': self.get_current_lr()}, step=n_iter)
            dist.barrier()

            end = time.time()

    def save_model(self, epoch, directory, is_best=False, val_result=None, model_name=''):
        names = self.get_model_names()

        if is_main_process():
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
