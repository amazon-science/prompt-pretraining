import os

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_sem_seg

PCONTEXT_SEM_SEG_CATEGORIES = ['aeroplane', 'bag', 'bed', 'bedclothes', 'bench', 'bicycle', 'bird', 'boat', 'book', 'bottle', 'building', 'bus', 'cabinet', 'car', 'cat', 'ceiling', 'chair', 'cloth', 'computer', 'cow', 'cup', 'curtain', 'dog', 'door', 'fence', 'floor', 'flower', 'food', 'grass', 'ground', 'horse', 'keyboard', 'light', 'motorbike', 'mountain', 'mouse', 'person', 'plate', 'platform', 'pottedplant', 'road', 'rock', 'sheep', 'shelves', 'sidewalk', 'sign', 'sky', 'snow', 'sofa', 'diningtable', 'track', 'train', 'tree', 'truck', 'tvmonitor', 'wall', 'water', 'window', 'wood']


def register_all_ade20k_full(root):
    root = os.path.join(root, "pcontext")
    for name, dirname in [("train", "train"), ("val", "val")]:
        image_dir = os.path.join(root, dirname, "image")
        gt_dir = os.path.join(root, dirname, "label")
        name = f"pcontext_sem_seg_{name}"
        DatasetCatalog.register(
            name, lambda x=image_dir, y=gt_dir: load_sem_seg(y, x, gt_ext="png", image_ext="jpg")
        )
        MetadataCatalog.get(name).set(
            stuff_classes=PCONTEXT_SEM_SEG_CATEGORIES[:],
            image_root=image_dir,
            sem_seg_root=gt_dir,
            evaluator_type="sem_seg",
            ignore_label=255,
        )


_root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_all_ade20k_full(_root)