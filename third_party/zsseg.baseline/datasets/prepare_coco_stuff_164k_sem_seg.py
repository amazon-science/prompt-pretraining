import argparse
import os
import os.path as osp
import shutil
from functools import partial
from glob import glob

import mmcv
import numpy as np
from PIL import Image

COCO_LEN = 123287

full_clsID_to_trID = {
    0: 0,
    1: 1,
    2: 2,
    3: 3,
    4: 4,
    5: 5,
    6: 6,
    7: 7,
    8: 8,
    9: 9,
    10: 10,
    12: 11,
    13: 12,
    14: 13,
    15: 14,
    16: 15,
    17: 16,
    18: 17,
    19: 18,
    20: 19,
    21: 20,
    22: 21,
    23: 22,
    24: 23,
    26: 24,
    27: 25,
    30: 26,
    31: 27,
    32: 28,
    33: 29,
    34: 30,
    35: 31,
    36: 32,
    37: 33,
    38: 34,
    39: 35,
    40: 36,
    41: 37,
    42: 38,
    43: 39,
    45: 40,
    46: 41,
    47: 42,
    48: 43,
    49: 44,
    50: 45,
    51: 46,
    52: 47,
    53: 48,
    54: 49,
    55: 50,
    56: 51,
    57: 52,
    58: 53,
    59: 54,
    60: 55,
    61: 56,
    62: 57,
    63: 58,
    64: 59,
    66: 60,
    69: 61,
    71: 62,
    72: 63,
    73: 64,
    74: 65,
    75: 66,
    76: 67,
    77: 68,
    78: 69,
    79: 70,
    80: 71,
    81: 72,
    83: 73,
    84: 74,
    85: 75,
    86: 76,
    87: 77,
    88: 78,
    89: 79,
    91: 80,
    92: 81,
    93: 82,
    94: 83,
    95: 84,
    96: 85,
    97: 86,
    98: 87,
    99: 88,
    100: 89,
    101: 90,
    102: 91,
    103: 92,
    104: 93,
    105: 94,
    106: 95,
    107: 96,
    108: 97,
    109: 98,
    110: 99,
    111: 100,
    112: 101,
    113: 102,
    114: 103,
    115: 104,
    116: 105,
    117: 106,
    118: 107,
    119: 108,
    120: 109,
    121: 110,
    122: 111,
    123: 112,
    124: 113,
    125: 114,
    126: 115,
    127: 116,
    128: 117,
    129: 118,
    130: 119,
    131: 120,
    132: 121,
    133: 122,
    134: 123,
    135: 124,
    136: 125,
    137: 126,
    138: 127,
    139: 128,
    140: 129,
    141: 130,
    142: 131,
    143: 132,
    144: 133,
    145: 134,
    146: 135,
    147: 136,
    148: 137,
    149: 138,
    150: 139,
    151: 140,
    152: 141,
    153: 142,
    154: 143,
    155: 144,
    156: 145,
    157: 146,
    158: 147,
    159: 148,
    160: 149,
    161: 150,
    162: 151,
    163: 152,
    164: 153,
    165: 154,
    166: 155,
    167: 156,
    168: 157,
    169: 158,
    170: 159,
    171: 160,
    172: 161,
    173: 162,
    174: 163,
    175: 164,
    176: 165,
    177: 166,
    178: 167,
    179: 168,
    180: 169,
    181: 170,
    255: 255,
}

novel_clsID = [20, 24, 32, 33, 40, 56, 86, 99, 105, 123, 144, 147, 148, 168, 171]
base_clsID = [k for k in full_clsID_to_trID.keys() if k not in novel_clsID + [255]]
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
        osp.join(out_mask_dir, "train2017" + suffix, osp.basename(maskpath))
        if is_train
        else osp.join(out_mask_dir, "val2017" + suffix, osp.basename(maskpath))
    )
    if len(np.unique(mask_copy)) == 1 and np.unique(mask_copy)[0] == 255:
        return
    Image.fromarray(mask_copy).save(seg_filename, "PNG")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert COCO Stuff 164k annotations to mmsegmentation format"
    )  # noqa
    parser.add_argument("coco_path", help="coco stuff path")
    parser.add_argument("-o", "--out_dir", help="output path")
    parser.add_argument("--nproc", default=16, type=int, help="number of process")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    coco_path = args.coco_path
    nproc = args.nproc
    print(full_clsID_to_trID)
    print(base_clsID_to_trID)
    print(novel_clsID_to_trID)
    out_dir = args.out_dir or coco_path
    out_mask_dir = osp.join(out_dir, "stuffthingmaps_detectron2")
    for dir_name in [
        "train2017",
        "val2017",
        "train2017_base",
        "train2017_novel",
        "val2017_base",
        "val2017_novel",
    ]:
        os.makedirs(osp.join(out_mask_dir, dir_name), exist_ok=True)
    train_list = glob(osp.join(coco_path, "stuffthingmaps", "train2017", "*.png"))
    test_list = glob(osp.join(coco_path, "stuffthingmaps", "val2017", "*.png"))
    assert (
        len(train_list) + len(test_list)
    ) == COCO_LEN, "Wrong length of list {} & {}".format(
        len(train_list), len(test_list)
    )

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

        mmcv.track_parallel_progress(
            partial(
                convert_to_trainID,
                out_mask_dir=out_mask_dir,
                is_train=False,
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
