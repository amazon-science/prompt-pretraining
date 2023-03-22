import json
from functools import partial
from mmcv.utils import track_parallel_progress
import os
from pycocotools import mask as mask_utils
import numpy as np
from PIL import Image
from mask_former import *
from detectron2.data import MetadataCatalog


def merge(data, out_dir, full_clsID_to_trID=None):
    file_name, seg_list = data
    if len(seg_list) == 0:
        return None
    mask = np.ones_like(seg_list[0]["mask"]) * 255
    for seg in seg_list:
        if full_clsID_to_trID is None:
            mask[seg["mask"] == 1] = seg["category_id"]
        else:
            mask[seg["mask"] == 1] = full_clsID_to_trID[seg["category_id"]]
    seg_filename = os.path.join(
        out_dir, os.path.basename(file_name).split(".")[0] + ".png"
    )
    Image.fromarray(mask).save(seg_filename, "PNG")
    return mask


def main(json_file, out_dir="pseudo", dataset="coco_2017_train_stuff_sem_seg"):
    with open(json_file, "r") as f:
        sem_segs = json.load(f)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    seg_dict = [
        {
            "file_name": seg["file_name"],
            "category_id": seg["category_id"],
            "mask": mask_utils.decode(seg["segmentation"]),
        }
        for seg in sem_segs
    ]
    file_names = list(set([v["file_name"] for v in seg_dict]))
    seg_maps = {k: [] for k in file_names}
    for seg in seg_dict:
        seg_maps[seg["file_name"]].append(seg)

    try:
        full_clsID_to_trID = MetadataCatalog.get(
            dataset
        ).stuff_dataset_id_to_contiguous_id
    except:
        full_clsID_to_trID = None
    _func = partial(merge, out_dir=out_dir, full_clsID_to_trID=full_clsID_to_trID)
    track_parallel_progress(_func, [(k, v) for k, v in seg_maps.items()], 16)


if __name__ == "__main__":
    import sys

    main(*sys.argv[1:])
