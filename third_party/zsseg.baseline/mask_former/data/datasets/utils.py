import json
import logging
import os

from detectron2.data.datasets.coco import load_sem_seg

logger = logging.getLogger(__name__)


def load_binary_mask(gt_root, image_root, gt_ext="png", image_ext="jpg"):
    """
    Flatten the results of `load_sem_seg` to annotations for binary mask.

    `label_count_file` contains a dictionary like:
    ```
    {
        "xxx.png":[0,3,5],
        "xxxx.png":[3,4,7],
    }
    ```
    """
    label_count_file = gt_root + "_label_count.json"
    with open(label_count_file) as f:
        label_count_dict = json.load(f)

    data_dicts = load_sem_seg(gt_root, image_root, gt_ext, image_ext)
    flattened_data_dicts = []
    for data in data_dicts:

        category_per_image = label_count_dict[
            os.path.basename(data["sem_seg_file_name"])
        ]
        flattened_data = [
            dict(**{"category_id": cat}, **data) for cat in category_per_image
        ]
        flattened_data_dicts.extend(flattened_data)
    logger.info(
        "Loaded {} images with flattened semantic segmentation from {}".format(
            len(flattened_data_dicts), image_root
        )
    )
    return flattened_data_dicts
