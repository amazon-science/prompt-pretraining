import os
import glob
import functools
from mmcv.utils import track_parallel_progress
import numpy as np
from PIL import Image
import json
import warnings
import fire
from itertools import chain


def count_cls(file_path, ignore_index=[255], depth=1):
    cls_label = np.unique(np.asarray(Image.open(file_path))).tolist()
    cls_label = [l for l in cls_label if l not in ignore_index]
    return [os.path.join(*file_path.split(os.sep)[-depth:]), cls_label]


def main(gt_dir, map_file_save_path, ignore_index=[255], ext=".png", recursive=False):
    if not os.path.isdir(gt_dir):
        warnings.warn(f"{gt_dir} is not a valid directory")
        return
    gt_file_list = glob.glob(os.path.join(gt_dir, "*" + ext), recursive=recursive)
    print(f"Find {len(gt_file_list)}")
    _func = functools.partial(count_cls, ignore_index=ignore_index)
    results = track_parallel_progress(_func, gt_file_list, nproc=16)
    results = {r[0]: r[1] for r in results}
    with open(map_file_save_path, "w") as f:
        json.dump(results, f)


def main_ctyscapes(
    gt_dir, map_file_save_path, ignore_index=[255], ext=".png", recursive=False
):
    if not os.path.isdir(gt_dir):
        warnings.warn(f"{gt_dir} is not a valid directory")
        return
    cities = os.listdir(gt_dir)
    gt_file_list = list(
        chain.from_iterable(
            [
                glob.glob(
                    os.path.join(gt_dir, city, "*" + ext),
                )
                for city in cities
            ]
        )
    )
    print(gt_file_list[0])
    print(f"Find {len(gt_file_list)}")
    _func = functools.partial(count_cls, ignore_index=ignore_index, depth=2)
    results = track_parallel_progress(_func, gt_file_list, nproc=16)
    results = {r[0]: r[1] for r in results}
    with open(map_file_save_path, "w") as f:
        json.dump(results, f)


if __name__ == "__main__":
    fire.Fire(main)
