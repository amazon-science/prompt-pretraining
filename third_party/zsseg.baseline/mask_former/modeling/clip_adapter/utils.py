from typing import Tuple
import numpy as np
import torch
import clip
from detectron2.utils.comm import get_local_rank, synchronize


class CLIP:
    PIXEL_MEAN = (0.48145466, 0.4578275, 0.40821073)
    PIXEL_STD = (0.26862954, 0.26130258, 0.27577711)
    IMAGENET_PROMPT = [
        "a bad photo of a {}.",
        "a photo of many {}.",
        "a sculpture of a {}.",
        "a photo of the hard to see {}.",
        "a low resolution photo of the {}.",
        "a rendering of a {}.",
        "graffiti of a {}.",
        "a bad photo of the {}.",
        "a cropped photo of the {}.",
        "a tattoo of a {}.",
        "the embroidered {}.",
        "a photo of a hard to see {}.",
        "a bright photo of a {}.",
        "a photo of a clean {}.",
        "a photo of a dirty {}.",
        "a dark photo of the {}.",
        "a drawing of a {}.",
        "a photo of my {}.",
        "the plastic {}.",
        "a photo of the cool {}.",
        "a close-up photo of a {}.",
        "a black and white photo of the {}.",
        "a painting of the {}.",
        "a painting of a {}.",
        "a pixelated photo of the {}.",
        "a sculpture of the {}.",
        "a bright photo of the {}.",
        "a cropped photo of a {}.",
        "a plastic {}.",
        "a photo of the dirty {}.",
        "a jpeg corrupted photo of a {}.",
        "a blurry photo of the {}.",
        "a photo of the {}.",
        "a good photo of the {}.",
        "a rendering of the {}.",
        "a {} in a video game.",
        "a photo of one {}.",
        "a doodle of a {}.",
        "a close-up photo of the {}.",
        "a photo of a {}.",
        "the origami {}.",
        "the {} in a video game.",
        "a sketch of a {}.",
        "a doodle of the {}.",
        "a origami {}.",
        "a low resolution photo of a {}.",
        "the toy {}.",
        "a rendition of the {}.",
        "a photo of the clean {}.",
        "a photo of a large {}.",
        "a rendition of a {}.",
        "a photo of a nice {}.",
        "a photo of a weird {}.",
        "a blurry photo of a {}.",
        "a cartoon {}.",
        "art of a {}.",
        "a sketch of the {}.",
        "a embroidered {}.",
        "a pixelated photo of a {}.",
        "itap of the {}.",
        "a jpeg corrupted photo of the {}.",
        "a good photo of a {}.",
        "a plushie {}.",
        "a photo of the nice {}.",
        "a photo of the small {}.",
        "a photo of the weird {}.",
        "the cartoon {}.",
        "art of the {}.",
        "a drawing of the {}.",
        "a photo of the large {}.",
        "a black and white photo of a {}.",
        "the plushie {}.",
        "a dark photo of a {}.",
        "itap of a {}.",
        "graffiti of the {}.",
        "a toy {}.",
        "itap of my {}.",
        "a photo of a cool {}.",
        "a photo of a small {}.",
        "a tattoo of the {}.",
    ]
    VILD_PROMPT = [
        "a photo of a {}.",
        "This is a photo of a {}",
        "There is a {} in the scene",
        "There is the {} in the scene",
        "a photo of a {} in the scene",
        "a photo of a small {}.",
        "a photo of a medium {}.",
        "a photo of a large {}.",
        "This is a photo of a small {}.",
        "This is a photo of a medium {}.",
        "This is a photo of a large {}.",
        "There is a small {} in the scene.",
        "There is a medium {} in the scene.",
        "There is a large {} in the scene.",
    ]


def expand_box(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    expand_ratio: float = 1.0,
    max_h: int = None,
    max_w: int = None,
):
    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)
    w = x2 - x1
    h = y2 - y1
    w = w * expand_ratio
    h = h * expand_ratio
    box = [cx - 0.5 * w, cy - 0.5 * h, cx + 0.5 * w, cy + 0.5 * h]
    if max_h is not None:
        box[1] = max(0, box[1])
        box[3] = min(max_h - 1, box[3])
    if max_w is not None:
        box[0] = max(0, box[0])
        box[2] = min(max_w - 1, box[2])
    return [int(b) for b in box]


def mask2box(mask: torch.Tensor):
    # use naive way
    row = torch.nonzero(mask.sum(dim=0))[:, 0]
    if len(row) == 0:
        return None
    x1 = row.min()
    x2 = row.max()
    col = np.nonzero(mask.sum(dim=1))[:, 0]
    y1 = col.min()
    y2 = col.max()
    return x1, y1, x2 + 1, y2 + 1


def crop_with_mask(
    image: torch.Tensor,
    mask: torch.Tensor,
    bbox: torch.Tensor,
    fill: Tuple[float, float, float] = (0, 0, 0),
    expand_ratio: float = 1.0,
):
    l, t, r, b = expand_box(*bbox, expand_ratio)
    _, h, w = image.shape
    l = max(l, 0)
    t = max(t, 0)
    r = min(r, w)
    b = min(b, h)
    new_image = torch.cat(
        [image.new_full((1, b - t, r - l), fill_value=val) for val in fill]
    )
    return (
        image[:, t:b, l:r] * mask[None, t:b, l:r]
        + (1 - mask[None, t:b, l:r]) * new_image
    )


def build_clip_model(model: str, frozen: bool = True):
    rank = get_local_rank()
    if rank == 0:
        # download on rank 0 only
        model, _ = clip.load(model, device="cpu")
    synchronize()
    if rank != 0:
        model, _ = clip.load(model, device="cpu")
    synchronize()
    if frozen:
        for param in model.parameters():
            param.requires_grad = False
    return model
