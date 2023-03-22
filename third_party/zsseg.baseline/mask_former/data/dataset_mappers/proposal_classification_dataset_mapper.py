from detectron2.data.dataset_mapper import DatasetMapper

# Copyright (c) Facebook, Inc. and its affiliates.
import copy
import logging

import numpy as np
import torch
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.structures import BoxMode
from ..augmentations import CropImageWithBox, RandomResizedCrop, CenterCrop


class ProposalClasificationDatasetMapper(DatasetMapper):
    @configurable
    def __init__(
        self,
        is_train: bool,
        *,
        augmentations,
        image_format: str,
    ):
        """
        NOTE: this interface is experimental.

        Args:
            is_train: whether it's used in training or inference
            augmentations: a list of augmentations or deterministic transforms to apply
            image_format: an image format supported by :func:`detection_utils.read_image`.
            use_instance_mask: whether to process instance segmentation annotations, if available
            use_keypoint: whether to process keypoint annotations if available
            instance_mask_format: one of "polygon" or "bitmask". Process instance segmentation
                masks into this format.
            keypoint_hflip_indices: see :func:`detection_utils.create_keypoint_hflip_indices`
            precomputed_proposal_topk: if given, will load pre-computed
                proposals from dataset_dict and keep the top k proposals for each image.
            recompute_boxes: whether to overwrite bounding box annotations
                by computing tight bounding boxes from instance mask annotations.
        """
        # fmt: off
        self.is_train               = is_train
        self.augmentations          = T.AugmentationList(augmentations)
        self.image_format           = image_format
        # fmt: on
        logger = logging.getLogger(__name__)
        mode = "training" if is_train else "inference"
        logger.info(f"[DatasetMapper] Augmentations used in {mode}: {augmentations}")

    @classmethod
    def from_config(cls, cfg, is_train: bool = True):
        if is_train:
            augs = [
                CropImageWithBox(cfg.INPUT.EXPAND_RATIO, mode=cfg.INPUT.EXPAND_MODE)
            ]
        else:
            augs = [CropImageWithBox(cfg.INPUT.EXPAND_RATIO[0])]
        if is_train:
            augs.append(RandomResizedCrop(cfg.INPUT.MIN_SIZE_TRAIN))
            augs.append(T.RandomFlip())
        else:
            min_size = cfg.INPUT.MIN_SIZE_TEST
            augs.append(T.ResizeShortestEdge(min_size, sample_style="choice"))
            augs.append(CenterCrop(min_size, seg_ignore_label=0))

        ret = {
            "is_train": is_train,
            "augmentations": augs,
            "image_format": cfg.INPUT.FORMAT,
        }
        return ret

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        # USER: Write your own image loading if it's not from a file
        image = utils.read_image(dataset_dict["file_name"], format=self.image_format)
        utils.check_image_size(dataset_dict, image)
        assert len(dataset_dict["annotations"]) == 1
        obj = dataset_dict["annotations"][0]

        aug_input = T.AugInput(
            image,
            boxes=BoxMode.convert(
                np.asarray(obj["bbox"]).reshape((1, 4)),
                obj["bbox_mode"],
                BoxMode.XYXY_ABS,
            ),
        )
        transforms = self.augmentations(aug_input)
        image = aug_input.image
        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = torch.as_tensor(
            np.ascontiguousarray(image.transpose(2, 0, 1))
        )
        dataset_dict["category_id"] = torch.Tensor([obj["category_id"]]).long()

        dataset_dict.pop("annotations", None)
        dataset_dict.pop("sem_seg_file_name", None)

        return dataset_dict
