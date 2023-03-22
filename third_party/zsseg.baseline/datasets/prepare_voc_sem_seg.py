import argparse
import os
import os.path as osp
import shutil
from functools import partial
from glob import glob

import mmcv
import numpy as np
from PIL import Image


full_clsID_to_trID = {
    0: 255,
    1: 0,
    2: 1,
    3: 2,
    4: 3,
    5: 4,
    6: 5,
    7: 6,
    8: 7,
    9: 8,
    10: 9,
    11: 10,
    12: 11,
    13: 12,
    14: 13,
    15: 14,
    16: 15,
    17: 16,
    18: 17,
    19: 18,
    20: 19,
    255: 255,
}

novel_clsID = [16, 17, 18, 19, 20]
base_clsID = [k for k in full_clsID_to_trID.keys() if k not in novel_clsID + [0, 255]]
novel_clsID_to_trID = {k: i for i, k in enumerate(novel_clsID)}
base_clsID_to_trID = {k: i for i, k in enumerate(base_clsID)}


def convert_to_trainID(
    maskpath, out_mask_dir, is_train, clsID_to_trID=full_clsID_to_trID, suffix=""
):
    mask = np.array(Image.open(maskpath))
    mask_copy = np.ones_like(mask, dtype=np.uint8) * 255
    for clsID, trID in clsID_to_trID.items():
        mask_copy[mask == clsID] = trID
    seg_filename = (
        osp.join(out_mask_dir, "train" + suffix, osp.basename(maskpath))
        if is_train
        else osp.join(out_mask_dir, "val" + suffix, osp.basename(maskpath))
    )
    if len(np.unique(mask_copy)) == 1 and np.unique(mask_copy)[0] == 255:
        return
    Image.fromarray(mask_copy).save(seg_filename, "PNG")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert VOC2021 annotations to mmsegmentation format"
    )  # noqa
    parser.add_argument("voc_path", help="voc path")
    parser.add_argument("-o", "--out_dir", help="output path")
    parser.add_argument("--nproc", default=16, type=int, help="number of process")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    voc_path = args.voc_path
    nproc = args.nproc
    print(full_clsID_to_trID)
    print(base_clsID_to_trID)
    print(novel_clsID_to_trID)
    out_dir = args.out_dir or voc_path
    # out_img_dir = osp.join(out_dir, 'images')
    out_mask_dir = osp.join(out_dir, "annotations_detectron2")
    out_image_dir = osp.join(out_dir, "images_detectron2")
    for dir_name in [
        "train",
        "val",
        "train_base",
        "train_novel",
        "val_base",
        "val_novel",
    ]:
        os.makedirs(osp.join(out_mask_dir, dir_name), exist_ok=True)
        if dir_name in ["train", "val"]:
            os.makedirs(osp.join(out_image_dir, dir_name), exist_ok=True)

    train_list = [
        osp.join(voc_path, "SegmentationClassAug", f + ".png")
        for f in np.loadtxt(osp.join(voc_path, "train.txt"), dtype=np.str).tolist()
    ]
    test_list = [
        osp.join(voc_path, "SegmentationClassAug", f + ".png")
        for f in np.loadtxt(osp.join(voc_path, "val.txt"), dtype=np.str).tolist()
    ]

    if args.nproc > 1:
        mmcv.track_parallel_progress(
            partial(convert_to_trainID, out_mask_dir=out_mask_dir, is_train=True),
            train_list,
            nproc=nproc,
        )
        mmcv.track_parallel_progress(
            partial(convert_to_trainID, out_mask_dir=out_mask_dir, is_train=False),
            test_list,
            nproc=nproc,
        )
        mmcv.track_parallel_progress(
            partial(
                convert_to_trainID,
                out_mask_dir=out_mask_dir,
                is_train=True,
                clsID_to_trID=base_clsID_to_trID,
                suffix="_base",
            ),
            train_list,
            nproc=nproc,
        )
        # mmcv.track_parallel_progress(
        #     partial(
        #         convert_to_trainID,
        #         out_mask_dir=out_mask_dir,
        #         is_train=True,
        #         clsID_to_trID=base_clsID_to_trID,
        #         suffix="_base_10class",
        #     ),
        #     train_list,
        #     nproc=nproc,
        # )
        mmcv.track_parallel_progress(
            partial(
                convert_to_trainID,
                out_mask_dir=out_mask_dir,
                is_train=True,
                clsID_to_trID=base_clsID_to_trID,
                suffix="_base",
            ),
            test_list,
            nproc=nproc,
        )
        mmcv.track_parallel_progress(
            partial(
                convert_to_trainID,
                out_mask_dir=out_mask_dir,
                is_train=True,
                clsID_to_trID=novel_clsID_to_trID,
                suffix="_novel",
            ),
            train_list,
            nproc=nproc,
        )
        mmcv.track_parallel_progress(
            partial(
                convert_to_trainID,
                out_mask_dir=out_mask_dir,
                is_train=False,
                clsID_to_trID=novel_clsID_to_trID,
                suffix="_novel",
            ),
            test_list,
            nproc=nproc,
        )
    else:
        mmcv.track_progress(
            partial(convert_to_trainID, out_mask_dir=out_mask_dir, is_train=True),
            train_list,
        )
        mmcv.track_progress(
            partial(convert_to_trainID, out_mask_dir=out_mask_dir, is_train=False),
            test_list,
        )
        mmcv.track_progress(
            partial(
                convert_to_trainID,
                out_mask_dir=out_mask_dir,
                is_train=True,
                clsID_to_trID=base_clsID_to_trID,
                suffix="_base",
            ),
            train_list,
        )
        mmcv.track_progress(
            partial(
                convert_to_trainID,
                out_mask_dir=out_mask_dir,
                is_train=False,
                clsID_to_trID=base_clsID_to_trID,
                suffix="_base",
            ),
            test_list,
        )
        mmcv.track_progress(
            partial(
                convert_to_trainID,
                out_mask_dir=out_mask_dir,
                is_train=True,
                clsID_to_trID=novel_clsID_to_trID,
                suffix="_novel",
            ),
            train_list,
        )
        mmcv.track_progress(
            partial(
                convert_to_trainID,
                out_mask_dir=out_mask_dir,
                is_train=False,
                clsID_to_trID=novel_clsID_to_trID,
                suffix="_novel",
            ),
            test_list,
        )
    print("Done!")


if __name__ == "__main__":
    main()
