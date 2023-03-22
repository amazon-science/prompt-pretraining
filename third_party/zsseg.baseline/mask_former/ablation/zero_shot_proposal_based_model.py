# Copyright (c) Facebook, Inc. and its affiliates.
import logging
import os
import torch
from torch import nn
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.data import MetadataCatalog
from detectron2.modeling import META_ARCH_REGISTRY
from detectron2.modeling.postprocessing import sem_seg_postprocess
from detectron2.structures import ImageList
from detectron2.utils.logger import log_first_n
from ..modeling.clip_adapter import (
    ClipAdapter,
    MaskFormerClipAdapter,
    PredefinedPromptExtractor,
    ImageNetPromptExtractor,
    VILDPromptExtractor,
    LearnablePromptExtractor,
    POMPPromptExtractor,
    build_prompt_learner,
)


@META_ARCH_REGISTRY.register()
class ZeroShotProposalBasedSegmentor(nn.Module):
    """
    Main class for zero shot mask classification semantic segmentation architectures.
    """

    @configurable
    def __init__(
        self,
        *,
        clip_adapter: nn.Module,
        clip_ensemble: bool,
        clip_ensemble_weight: float,
    ):
        super().__init__()
        self.clip_adapter: ClipAdapter = clip_adapter
        self.clip_ensemble: bool = clip_ensemble
        self.clip_ensemble_weight: float = clip_ensemble_weight

    @classmethod
    def from_config(cls, cfg):
        init_kwargs = {}
        prompt_learner = build_prompt_learner(cfg.MODEL.CLIP_ADAPTER)
        clip_adapter = MaskFormerClipAdapter(
            cfg.MODEL.CLIP_ADAPTER.CLIP_MODEL_NAME,
            prompt_learner,
            mask_fill=cfg.MODEL.CLIP_ADAPTER.MASK_FILL,
            mask_expand_ratio=cfg.MODEL.CLIP_ADAPTER.MASK_EXPAND_RATIO,
            mask_thr=cfg.MODEL.CLIP_ADAPTER.MASK_THR,
            mask_matting=cfg.MODEL.CLIP_ADAPTER.MASK_MATTING,
            region_resized=cfg.MODEL.CLIP_ADAPTER.REGION_RESIZED,
        )
        init_kwargs["clip_adapter"] = clip_adapter
        init_kwargs["clip_ensemble"] = cfg.MODEL.CLIP_ADAPTER.CLIP_ENSEMBLE
        init_kwargs[
            "clip_ensemble_weight"
        ] = cfg.MODEL.CLIP_ADAPTER.CLIP_ENSEMBLE_WEIGHT

        return init_kwargs

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
        images = ImageList.from_tensors(images, self.size_divisibility)
        class_names = self.get_class_name_list(dataset_name)
        if self.training:
            raise NotImplementedError()
        else:
            processed_results = []
            for input_per_image, image_size in zip(batched_inputs, images.image_sizes):
                height = image_size[0]
                width = image_size[1]

                image = input_per_image["image"].to(self.device)
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])
                image = sem_seg_postprocess(image, image_size, height, width)
                # semantic segmentation inference
                mask_proposals = self.load_mask_proposals(
                    os.path.basename(input_per_image["file_name"])
                )
                r = self.semantic_inference(
                    mask_proposals, image, class_names, dataset_name
                )
                processed_results.append({"sem_seg": r})
            return processed_results

    def semantic_inference(self, mask_pred, image, class_names, dataset_name):
        # get the classification result from clip model
        clip_cls, valid_flag = self.clip_adapter(
            image, class_names, mask_pred, normalize=True
        )
        # softmax before index or after?
        clip_cls = F.softmax(clip_cls[:, :-1], dim=-1)
        mask_pred = mask_pred[valid_flag]
        semseg = torch.einsum("qc,qhw->chw", clip_cls, mask_pred)
        return semseg

    def get_class_name_list(self, dataset_name):
        class_names = [
            c.strip() for c in MetadataCatalog.get(dataset_name).stuff_classes
        ]
        return class_names

    def load_mask_proposals(self, image_filename):
        file_name = ".".join(image_filename.split(".")[:-1])
        proposals = torch.from_numpy(
            np.load(os.path.join(self.proposal_dir, file_name + ".npy"))
        ).to(self.device)
        proposals = F.one_hot(proposals).permute(2, 0, 1)
        return proposals
