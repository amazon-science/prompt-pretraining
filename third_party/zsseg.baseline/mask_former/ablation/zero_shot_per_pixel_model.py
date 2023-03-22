# Copyright (c) Facebook, Inc. and its affiliates.
import os
import logging

import torch
from torch import nn
from torch.nn import functional as F


from detectron2.config import configurable
from detectron2.data import MetadataCatalog
from detectron2.modeling import META_ARCH_REGISTRY, build_backbone, build_sem_seg_head
from detectron2.modeling.postprocessing import sem_seg_postprocess
from detectron2.structures import ImageList
from detectron2.utils.logger import log_first_n
from detectron2.modeling import SemanticSegmentor
from ..modeling.clip_adapter import (
    PerPixelClipAdapter,
    PredefinedPromptExtractor,
    ImageNetPromptExtractor,
    VILDPromptExtractor,
    LearnablePromptExtractor,
    POMPPromptExtractor,
)


@META_ARCH_REGISTRY.register()
class ZeroShotPerPixelModel(SemanticSegmentor):
    @configurable
    def __init__(
        self,
        clip_adapter,
        clip_ensemble_weight,
        **kwargs,
    ):
        super(ZeroShotPerPixelModel, self).__init__(**kwargs)
        self.clip_adapter = clip_adapter
        self.clip_ensemble_weight = clip_ensemble_weight
        # for debug
        names = []
        for name, param in self.named_parameters():
            if param.requires_grad:
                names.append(name)
        while len(names) > 0:
            if len(names) > 20:
                log_first_n(logging.INFO, names[:20], n=100)
                names = names[20:]
            else:
                log_first_n(logging.INFO, names)
                names = []

    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        sem_seg_head = build_sem_seg_head(cfg, backbone.output_shape())
        prompt_learner = build_prompt_learner(cfg.MODEL.CLIP_ADAPTER)
        clip_adapter = PerPixelClipAdapter(
            cfg.MODEL.CLIP_ADAPTER.CLIP_MODEL_NAME,
            prompt_learner,
        )
        return {
            "clip_adapter": clip_adapter,
            "backbone": backbone,
            "sem_seg_head": sem_seg_head,
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            "clip_ensemble_weight": cfg.MODEL.CLIP_ADAPTER.CLIP_ENSEMBLE_WEIGHT,
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
        Returns:
            list[dict]:
                each dict has the results for one image. The dict contains the following keys:

                * "sem_seg":
                    A Tensor that represents the
                    per-pixel segmentation prediced by the head.
                    The prediction has shape KxHxW that represents the logits of
                    each class for each pixel.
                * "panoptic_seg":
                    A tuple that represent panoptic output
                    panoptic_seg (Tensor): of shape (height, width) where the values are ids for each segment.
                    segments_info (list[dict]): Describe each segment in `panoptic_seg`.
                        Each dict contains keys "id", "category_id", "isthing".
        """
        dataset_name = [x["meta"]["dataset_name"] for x in batched_inputs]
        assert len(set(dataset_name)) == 1
        dataset_name = dataset_name[0]
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)

        features = self.backbone(images.tensor)

        if "sem_seg" in batched_inputs[0]:
            targets = [x["sem_seg"].to(self.device) for x in batched_inputs]
            targets = ImageList.from_tensors(
                targets, self.backbone.size_divisibility, self.sem_seg_head.ignore_value
            ).tensor
        else:
            targets = None
        class_names = [
            c.strip() for c in MetadataCatalog.get(dataset_name).stuff_classes
        ]
        text_features = self.clip_adapter.get_text_features(class_names)
        targets = {
            "sem_seg_target": targets,
            "text_features": text_features,
            "cosine_sim_func": self.clip_adapter.get_sim_logits,
        }
        results, losses = self.sem_seg_head(features, targets)

        if self.training:
            return losses

        processed_results = []
        for result, input_per_image, image_size in zip(
            results, batched_inputs, images.image_sizes
        ):
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])
            r = sem_seg_postprocess(result, image_size, height, width).permute(1, 2, 0)
            r = r / r.norm(dim=-1, keepdim=True)
            r = (
                self.clip_adapter.get_sim_logits(text_features, r)
                .permute(2, 0, 1)
                .softmax(dim=0)
            )  # Cls,H,W
            if self.clip_ensemble_weight <= 1.0:
                dataset_name = input_per_image["meta"]["dataset_name"]
                class_names = self.get_class_name_list(dataset_name)
                image = input_per_image["image"]  # C,H,W
                clip_act_map = (
                    self.clip_adapter(image.unsqueeze(0), class_names)
                    .softmax(dim=-1)
                    .permute(0, 3, 1, 2)
                    .squeeze(0)
                )  # b,cls,gh,gw
                clip_act_map = sem_seg_postprocess(
                    clip_act_map, image_size, height, width
                )
                if self.clip_ensemble_weight < 0:
                    r = clip_act_map
                else:
                    if hasattr(MetadataCatalog.get(dataset_name), "trainable_flag"):
                        trained_mask = torch.Tensor(
                            MetadataCatalog.get(dataset_name).trainable_flag
                        ).to(clip_act_map.device)[:, None, None]
                    else:
                        trained_mask = 0
                    r = trained_mask * torch.pow(
                        r, self.clip_ensemble_weight
                    ) * torch.pow(clip_act_map, 1 - self.clip_ensemble_weight) + (
                        1 - trained_mask
                    ) * torch.pow(
                        r, 1 - self.clip_ensemble_weight
                    ) * torch.pow(
                        clip_act_map, self.clip_ensemble_weight
                    )
            processed_results.append({"sem_seg": r})
        return processed_results

    def get_class_name_list(self, dataset_name):
        class_names = [
            c.strip() for c in MetadataCatalog.get(dataset_name).stuff_classes
        ]
        return class_names
