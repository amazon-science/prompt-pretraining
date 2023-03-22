from typing import List
import torch
from torch import nn
from torch.nn import functional as F
from detectron2.structures import BitMasks
from .utils import build_clip_model, crop_with_mask, CLIP
from .text_prompt import PromptExtractor


class ClipAdapter(nn.Module):
    def __init__(self, clip_model_name: str, prompt_learner: PromptExtractor):
        super().__init__()
        self.clip_model = build_clip_model(clip_model_name)
        self.prompt_learner = prompt_learner
        self.prompt_learner.init_buffer(self.clip_model)
        self.text_feature_buffer = {}

    def forward(self, image: torch.Tensor, text: List[str], **kwargs):
        image = self._preprocess_image(image, **kwargs)
        text_feature = self.get_text_features(text)  # k,feat_dim
        image_features = self.get_image_features(image)
        return self.get_sim_logits(text_feature, image_features)

    def _preprocess_image(self, image: torch.Tensor):
        return image

    def _get_text_features(self, noun_list: List[str]):
        if not self.prompt_learner.with_trainable_params:

            left_noun_list = [
                noun for noun in noun_list if noun not in self.text_feature_buffer
            ]
            if len(left_noun_list) > 0:
                left_text_features = self.prompt_learner(
                    left_noun_list, self.clip_model
                )
                self.text_feature_buffer.update(
                    {
                        noun: text_feature
                        for noun, text_feature in zip(
                            left_noun_list, left_text_features
                        )
                    }
                )
            return torch.stack([self.text_feature_buffer[noun] for noun in noun_list])
        else:
            text_features = self.prompt_learner(noun_list, self.clip_model)
            self.text_feature_buffer.update(
                {
                    noun: text_feature.detach()
                    for noun, text_feature in zip(noun_list, text_features)
                }
            )
            return text_features

    def get_text_features(self, noun_list: List[str]):
        return self._get_text_features(noun_list)

    def get_image_features(self, image: torch.Tensor):
        image_features = self.clip_model.visual(image)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        return image_features

    def get_sim_logits(
        self,
        text_features: torch.Tensor,
        image_features: torch.Tensor,
        temperature: float = 100,
    ):
        return temperature * image_features @ text_features.T

    def normalize_feature(self, feat: torch.Tensor):
        return feat / feat.norm(dim=-1, keepdim=True)


class MaskFormerClipAdapter(ClipAdapter):
    def __init__(
        self,
        clip_model_name: str,
        prompt_learner: PromptExtractor,
        mask_fill: str = "mean",
        mask_expand_ratio: float = 1.0,
        mask_thr: float = 0.5,
        mask_matting: bool = False,
        region_resized: bool = True,
    ):
        super().__init__(clip_model_name, prompt_learner)
        self.non_object_embedding = nn.Parameter(
            torch.empty(1, self.clip_model.text_projection.shape[-1])
        )
        nn.init.normal_(
            self.non_object_embedding.data,
            std=self.clip_model.transformer.width ** -0.5,
        )
        # for test
        self.mask_fill = mask_fill
        if self.mask_fill == "zero":
            self.mask_fill = (0.0, 0.0, 0.0)
        elif self.mask_fill == "mean":
            self.mask_fill = [255.0 * c for c in CLIP.PIXEL_MEAN]
        else:
            raise NotImplementedError(
                "Unknown mask_fill method: {}".format(self.mask_fill)
            )
        self.mask_expand_ratio = mask_expand_ratio
        self.mask_thr = mask_thr
        self.mask_matting = mask_matting
        self.region_resized = region_resized

        self.register_buffer(
            "pixel_mean", torch.Tensor(CLIP.PIXEL_MEAN).reshape(1, 3, 1, 1) * 255.0
        )
        self.register_buffer(
            "pixel_std", torch.Tensor(CLIP.PIXEL_STD).reshape(1, 3, 1, 1) * 255.0
        )

    def forward(
        self,
        image: torch.Tensor,
        text: List[str],
        mask: torch.Tensor,
        normalize: bool = True,
    ):
        image, valid_flag = self._preprocess_image(image, mask, normalize=normalize)
        if image is None:
            return None, valid_flag
        if isinstance(image, list):
            image_features = torch.cat(
                [self.get_image_features(image_i) for image_i in image], dim=0
            )
        else:
            image_features = self.get_image_features(image)
        text_feature = self.get_text_features(text)  # k,feat_dim
        return self.get_sim_logits(text_feature, image_features), valid_flag

    def _preprocess_image(
        self, image: torch.Tensor, mask: torch.Tensor, normalize: bool = True
    ):
        """crop, mask and normalize the image

        Args:
            image ([type]): [C,H,W]
            mask ([type]): [K,H,W
            normalize (bool, optional): [description]. Defaults to True.
        """
        dtype = mask.dtype
        bin_mask = mask > self.mask_thr
        valid = bin_mask.sum(dim=(-1, -2)) > 0
        bin_mask = bin_mask[valid]
        mask = mask[valid]
        if not self.mask_matting:
            mask = bin_mask
        bin_mask = BitMasks(bin_mask)
        bboxes = bin_mask.get_bounding_boxes()
        # crop,mask
        regions = [
            crop_with_mask(
                image.type(dtype),
                single_mask.type(dtype),
                bbox,
                fill=self.mask_fill,
                expand_ratio=self.mask_expand_ratio,
            )[None, ...]
            for bbox, single_mask in zip(bboxes, mask)
        ]
        if len(regions) == 0:
            return None, valid
        if normalize:
            regions = [(r - self.pixel_mean) / self.pixel_std for r in regions]
        # resize
        if self.region_resized:
            regions = [
                F.interpolate(r, size=(224, 224), mode="bicubic") for r in regions
            ]
            regions = torch.cat(regions)
        return regions, valid

    def get_text_features(self, noun_list: List[str]):
        object_text_features = self._get_text_features(noun_list)
        non_object_text_features = (
            self.non_object_embedding
            / self.non_object_embedding.norm(dim=-1, keepdim=True)
        )
        return torch.cat([object_text_features, non_object_text_features], dim=0)


class PerPixelClipAdapter(ClipAdapter):
    def __init__(self, *args, **kwargs):
        super(PerPixelClipAdapter, self).__init__(*args, **kwargs)
        self.register_buffer(
            "pixel_mean", torch.Tensor(CLIP.PIXEL_MEAN).reshape(1, 3, 1, 1) * 255.0
        )
        self.register_buffer(
            "pixel_std", torch.Tensor(CLIP.PIXEL_STD).reshape(1, 3, 1, 1) * 255.0
        )

    def _preprocess_image(self, image: torch.Tensor):
        return (image.to(self.pixel_mean.device) - self.pixel_mean) / self.pixel_std

    def get_image_features(self, image: torch.Tensor, per_pixel: bool = False):
        if per_pixel:
            image_features = self.clip_model.visual(image, return_cls=False)  # b,h,w,c
        else:
            image_features = self.clip_model.visual(image)[:, None, None, :].expand(
                image.shape[0], 2, 2, -1
            )  # b,c
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        return image_features

    def forward(
        self, image: torch.Tensor, text: List[str], per_pixel: bool = True, **kwargs
    ):
        image = self._preprocess_image(image, **kwargs)
        text_feature = self.get_text_features(text)  # k,feat_dim
        image_features = self.get_image_features(image)
        return self.get_sim_logits(text_feature, image_features)
