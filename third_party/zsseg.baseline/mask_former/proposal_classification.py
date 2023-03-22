# Copyright (c) Facebook, Inc. and its affiliates.
import logging

import torch
import numpy as np
from detectron2.config import configurable
from detectron2.data import MetadataCatalog
from detectron2.modeling import META_ARCH_REGISTRY
from detectron2.utils.logger import log_every_n, log_first_n
from detectron2.utils.events import get_event_storage
from detectron2.utils.visualizer import Visualizer
from torch import nn
from torch.nn import functional as F

from .modeling.clip_adapter import (
    ClipAdapter,
    PredefinedPromptExtractor,
    ImageNetPromptExtractor,
    VILDPromptExtractor,
    LearnablePromptExtractor,
    POMPPromptExtractor,
)
from .modeling.clip_adapter.utils import CLIP


@META_ARCH_REGISTRY.register()
class ProposalClipClassifier(nn.Module):
    @configurable
    def __init__(self, clip_adapter):
        super().__init__()
        self.clip_adapter = clip_adapter
        # store text features
        self.text_features = dict()
        self.register_buffer(
            "pixel_mean", torch.Tensor(CLIP.PIXEL_MEAN).view(1, -1, 1, 1), False
        )
        self.register_buffer(
            "pixel_std", torch.Tensor(CLIP.PIXEL_STD).view(1, -1, 1, 1), False
        )
        names = []
        for name, param in self.named_parameters():
            if param.requires_grad:
                names.append(name)
        log_first_n(logging.INFO, names)

    @classmethod
    def from_config(cls, cfg):

        if cfg.MODEL.CLIP_ADAPTER.PROMPT_LEARNER == "predefined":
            prompt_learner = PredefinedPromptExtractor(
                cfg.MODEL.CLIP_ADAPTER.PREDEFINED_PROMPT_TEMPLATES
            )
        elif cfg.MODEL.CLIP_ADAPTER.PROMPT_LEARNER == "imagenet":
            prompt_learner = ImageNetPromptExtractor()
        elif cfg.MODEL.CLIP_ADAPTER.PROMPT_LEARNER == "vild":
            prompt_learner = VILDPromptExtractor()
        elif cfg.MODEL.CLIP_ADAPTER.PROMPT_LEARNER == "learnable":
            prompt_learner = LearnablePromptExtractor(
                prompt_dim=cfg.MODEL.CLIP_ADAPTER.PROMPT_DIM,
                prompt_shape=cfg.MODEL.CLIP_ADAPTER.PROMPT_SHAPE,
            )
        elif cfg.MODEL.CLIP_ADAPTER.PROMPT_LEARNER == "pomp":
            prompt_learner = POMPPromptExtractor(
                prompt_dim=cfg.MODEL.CLIP_ADAPTER.PROMPT_DIM,
                prompt_shape=cfg.MODEL.CLIP_ADAPTER.PROMPT_SHAPE,
            )
            if cfg.MODEL.CLIP_ADAPTER.PROMPT_CHECKPOINT != "":
                checkpoint = torch.load(cfg.MODEL.CLIP_ADAPTER.PROMPT_CHECKPOINT, map_location="cpu")["state_dict"]
                missing, unexpected = prompt_learner.load_state_dict(
                    {
                        k: v
                        for k, v in checkpoint.items()
                        if "ctx" in k
                    },
                    strict=False,
                )
                log_first_n(
                    logging.INFO,
                    "Load Prompt Learner from {}".format(cfg.MODEL.CLIP_ADAPTER.PROMPT_CHECKPOINT),
                    1,
                )
                log_first_n(logging.WARN, "Missing {}".format(missing), 1)
                log_first_n(logging.WARN, "Unexpected {}".format(unexpected), 1)
        else:
            raise NotImplementedError(
                "Prompt learner {} is not supported".format(
                    cfg.MODEL.CLIP_ADAPTER.PROMPT_LEARNER
                )
            )
        clip_adapter = ClipAdapter(
            cfg.MODEL.CLIP_ADAPTER.CLIP_MODEL_NAME, prompt_learner
        )
        return {
            "clip_adapter": clip_adapter,
        }

    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper`.
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:
                   * "image": Tensor, image in (C, H, W) format.
                   * "instances": per-region ground truth
                   * Other information that's included in the original dicts, such as:
                     "height", "width" (int): the output resolution of the model (may be different
                     from input resolution), used in inference.
        """
        dataset_name = [x["meta"]["dataset_name"] for x in batched_inputs]
        assert len(set(dataset_name)) == 1
        dataset_name = dataset_name[0]
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = torch.stack(images)
        masks = (
            torch.stack([x["instances"].gt_masks[0] for x in batched_inputs])
            .to(self.device)
            .type(images.dtype)
        )

        class_names = [
            c.strip() for c in MetadataCatalog.get(dataset_name).stuff_classes
        ]
        # normalize
        images = (images / 255.0 - self.pixel_mean) / self.pixel_std
        images = (
            images * masks[:, None, ...] + (1 - masks[:, None, ...]) * self.pixel_mean
        )
        logits = self.clip_adapter(images, class_names)
        metadata = MetadataCatalog.get(dataset_name)

        if self.training:
            target = torch.cat([x["instances"].gt_classes for x in batched_inputs])
            loss_cls = F.cross_entropy(logits, target.to(self.device))
            storage = get_event_storage()
            if storage.iter % 100 == 0:
                vis = Visualizer(
                    batched_inputs[0]["image"].permute(1, 2, 0).cpu().numpy().copy(),
                    metadata,
                )

                vis_mask = target.new_ones(batched_inputs[0]["image"].shape[1:]) * 255
                vis_mask[batched_inputs[0]["instances"].gt_masks[0]] = batched_inputs[
                    0
                ]["instances"].gt_classes[0]
                vis.draw_sem_seg(vis_mask)
                pvis = Visualizer(
                    batched_inputs[0]["image"].permute(1, 2, 0).cpu().numpy().copy(),
                    metadata,
                )
                vis_mask = target.new_ones(batched_inputs[0]["image"].shape[1:]) * 255
                vis_mask[batched_inputs[0]["instances"].gt_masks[0]] = (
                    logits[0].argmax().detach().cpu()
                )
                pvis.draw_sem_seg(vis_mask)
                storage.put_image(
                    "train_data",
                    np.concatenate(
                        [vis.get_output().get_image(), pvis.get_output().get_image()],
                        axis=1,
                    ),
                )
                storage.put_scalar(
                    "train_acc",
                    100.0
                    * (logits.detach().argmax(dim=1).cpu() == target).sum()
                    / len(target),
                )
            return {"loss_cls": loss_cls}
        else:
            sim = [{"classification": logit[None].detach()} for logit in logits]
            return sim

    @property
    def device(self):
        return self.pixel_mean.device
