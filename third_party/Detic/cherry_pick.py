# Copyright (c) Facebook, Inc. and its affiliates.
import logging
import os
import sys
from collections import OrderedDict
import torch
from torch.nn.parallel import DistributedDataParallel
import time
import datetime

from fvcore.common.timer import Timer
import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer, PeriodicCheckpointer
from detectron2.config import get_cfg
from detectron2.data import (
    MetadataCatalog,
    build_detection_test_loader,
)
from detectron2.engine import default_argument_parser, default_setup, launch

from detectron2.evaluation import (
    inference_on_dataset,
    print_csv_format,
    COCOEvaluator,
)
from detectron2.modeling import build_model
from detectron2.solver import build_lr_scheduler, build_optimizer
from detectron2.utils.events import (
    CommonMetricPrinter,
    EventStorage,
    JSONWriter,
    TensorboardXWriter,
)
from detectron2.data.dataset_mapper import DatasetMapper
from detectron2.data.build import build_detection_train_loader
from detectron2.utils.logger import setup_logger
from torch.cuda.amp import GradScaler

sys.path.insert(0, 'third_party/CenterNet2/')
from centernet.config import add_centernet_config

sys.path.insert(0, 'third_party/Deformable-DETR')
from detic.config import add_detic_config
from detic.data.custom_build_augmentation import build_custom_augmentation
from detic.data.custom_dataset_dataloader import  build_custom_train_loader
from detic.data.custom_dataset_mapper import CustomDatasetMapper, DetrDatasetMapper
from detic.custom_solver import build_custom_optimizer
from detic.evaluation.oideval import OIDEvaluator
from detic.evaluation.custom_coco_eval import CustomCOCOEvaluator
from detic.modeling.utils import reset_cls_test


logger = logging.getLogger("detectron2")


# Copyright (c) Facebook, Inc. and its affiliates.
import copy
import itertools
import json
import logging
import os
import pickle
from collections import OrderedDict
import torch

import detectron2.utils.comm as comm
from detectron2.config import CfgNode
from detectron2.data import MetadataCatalog
from detectron2.structures import Boxes, BoxMode, pairwise_iou
from detectron2.utils.file_io import PathManager
from detectron2.utils.logger import create_small_table

from detectron2.evaluation.coco_evaluation import instances_to_coco_json
from detectron2.evaluation.evaluator import DatasetEvaluator

import datetime
import logging
from collections import OrderedDict
from collections import defaultdict

import numpy as np

from lvis.lvis import LVIS
from lvis.results import LVISResults

import pycocotools.mask as mask_utils


class LVISEval:
    def __init__(self, lvis_gt, lvis_dt, iou_type="segm"):
        """Constructor for LVISEval.
        Args:
            lvis_gt (LVIS class instance, or str containing path of annotation file)
            lvis_dt (LVISResult class instance, or str containing path of result file,
            or list of dict)
            iou_type (str): segm or bbox evaluation
        """
        self.logger = logging.getLogger(__name__)

        if iou_type not in ["bbox", "segm"]:
            raise ValueError("iou_type: {} is not supported.".format(iou_type))

        if isinstance(lvis_gt, LVIS):
            self.lvis_gt = lvis_gt
        elif isinstance(lvis_gt, str):
            self.lvis_gt = LVIS(lvis_gt)
        else:
            raise TypeError("Unsupported type {} of lvis_gt.".format(lvis_gt))

        if isinstance(lvis_dt, LVISResults):
            self.lvis_dt = lvis_dt
        elif isinstance(lvis_dt, (str, list)):
            self.lvis_dt = LVISResults(self.lvis_gt, lvis_dt)
        else:
            raise TypeError("Unsupported type {} of lvis_dt.".format(lvis_dt))

        # per-image per-category evaluation results
        self.eval_imgs = defaultdict(list)
        self.eval = {}  # accumulated evaluation results
        self._gts = defaultdict(list)  # gt for evaluation
        self._dts = defaultdict(list)  # dt for evaluation
        self.params = Params(iou_type=iou_type)  # parameters
        self.results = OrderedDict()
        self.ious = {}  # ious between all gts and dts

        self.params.img_ids = sorted(self.lvis_gt.get_img_ids())
        self.params.cat_ids = sorted(self.lvis_gt.get_cat_ids())

    def _to_mask(self, anns, lvis):
        for ann in anns:
            rle = lvis.ann_to_rle(ann)
            ann["segmentation"] = rle

    def _prepare(self):
        """Prepare self._gts and self._dts for evaluation based on params."""

        cat_ids = self.params.cat_ids if self.params.cat_ids else None

        gts = self.lvis_gt.load_anns(
            self.lvis_gt.get_ann_ids(img_ids=self.params.img_ids, cat_ids=cat_ids)
        )
        dts = self.lvis_dt.load_anns(
            self.lvis_dt.get_ann_ids(img_ids=self.params.img_ids, cat_ids=cat_ids)
        )
        # convert ground truth to mask if iou_type == 'segm'
        if self.params.iou_type == "segm":
            self._to_mask(gts, self.lvis_gt)
            self._to_mask(dts, self.lvis_dt)

        # set ignore flag
        for gt in gts:
            if "ignore" not in gt:
                gt["ignore"] = 0

        for gt in gts:
            self._gts[gt["image_id"], gt["category_id"]].append(gt)

        # For federated dataset evaluation we will filter out all dt for an
        # image which belong to categories not present in gt and not present in
        # the negative list for an image. In other words detector is not penalized
        # for categories about which we don't have gt information about their
        # presence or absence in an image.
        img_data = self.lvis_gt.load_imgs(ids=self.params.img_ids)
        # per image map of categories not present in image
        img_nl = {d["id"]: d["neg_category_ids"] for d in img_data}
        # per image list of categories present in image
        img_pl = defaultdict(set)
        for ann in gts:
            img_pl[ann["image_id"]].add(ann["category_id"])
        # per image map of categoires which have missing gt. For these
        # categories we don't penalize the detector for flase positives.
        self.img_nel = {d["id"]: d["not_exhaustive_category_ids"] for d in img_data}

        for dt in dts:
            img_id, cat_id = dt["image_id"], dt["category_id"]
            if cat_id not in img_nl[img_id] and cat_id not in img_pl[img_id]:
                continue
            self._dts[img_id, cat_id].append(dt)

        self.freq_groups = self._prepare_freq_group()

    def _prepare_freq_group(self):
        freq_groups = [[] for _ in self.params.img_count_lbl]
        cat_data = self.lvis_gt.load_cats(self.params.cat_ids)
        for idx, _cat_data in enumerate(cat_data):
            frequency = _cat_data["frequency"]
            freq_groups[self.params.img_count_lbl.index(frequency)].append(idx)
        return freq_groups

    def evaluate(self):
        """
        Run per image evaluation on given images and store results
        (a list of dict) in self.eval_imgs.
        """
        self.logger.info("Running per image evaluation.")
        self.logger.info("Evaluate annotation type *{}*".format(self.params.iou_type))

        self.params.img_ids = list(np.unique(self.params.img_ids))

        if self.params.use_cats:
            cat_ids = self.params.cat_ids
        else:
            cat_ids = [-1]

        self._prepare()

        self.ious = {
            (img_id, cat_id): self.compute_iou(img_id, cat_id)
            for img_id in self.params.img_ids
            for cat_id in cat_ids
        }

        # loop through images, area range, max detection number
        self.eval_imgs = [
            self.evaluate_img(img_id, cat_id, area_rng)
            for cat_id in cat_ids
            for area_rng in self.params.area_rng
            for img_id in self.params.img_ids
        ]

    def _get_gt_dt(self, img_id, cat_id):
        """Create gt, dt which are list of anns/dets. If use_cats is true
        only anns/dets corresponding to tuple (img_id, cat_id) will be
        used. Else, all anns/dets in image are used and cat_id is not used.
        """
        if self.params.use_cats:
            gt = self._gts[img_id, cat_id]
            dt = self._dts[img_id, cat_id]
        else:
            gt = [
                _ann
                for _cat_id in self.params.cat_ids
                for _ann in self._gts[img_id, cat_id]
            ]
            dt = [
                _ann
                for _cat_id in self.params.cat_ids
                for _ann in self._dts[img_id, cat_id]
            ]
        return gt, dt

    def compute_iou(self, img_id, cat_id):
        gt, dt = self._get_gt_dt(img_id, cat_id)

        if len(gt) == 0 and len(dt) == 0:
            return []

        # Sort detections in decreasing order of score.
        idx = np.argsort([-d["score"] for d in dt], kind="mergesort")
        dt = [dt[i] for i in idx]

        iscrowd = [int(False)] * len(gt)

        if self.params.iou_type == "segm":
            ann_type = "segmentation"
        elif self.params.iou_type == "bbox":
            ann_type = "bbox"
        else:
            raise ValueError("Unknown iou_type for iou computation.")
        gt = [g[ann_type] for g in gt]
        dt = [d[ann_type] for d in dt]

        # compute iou between each dt and gt region
        # will return array of shape len(dt), len(gt)
        ious = mask_utils.iou(dt, gt, iscrowd)
        return ious

    def evaluate_img(self, img_id, cat_id, area_rng):
        """Perform evaluation for single category and image."""
        gt, dt = self._get_gt_dt(img_id, cat_id)

        if len(gt) == 0 and len(dt) == 0:
            return None

        # Add another filed _ignore to only consider anns based on area range.
        for g in gt:
            if g["ignore"] or (g["area"] < area_rng[0] or g["area"] > area_rng[1]):
                g["_ignore"] = 1
            else:
                g["_ignore"] = 0

        # Sort gt ignore last
        gt_idx = np.argsort([g["_ignore"] for g in gt], kind="mergesort")
        gt = [gt[i] for i in gt_idx]

        # Sort dt highest score first
        dt_idx = np.argsort([-d["score"] for d in dt], kind="mergesort")
        dt = [dt[i] for i in dt_idx]

        # load computed ious
        ious = (
            self.ious[img_id, cat_id][:, gt_idx]
            if len(self.ious[img_id, cat_id]) > 0
            else self.ious[img_id, cat_id]
        )

        num_thrs = len(self.params.iou_thrs)
        num_gt = len(gt)
        num_dt = len(dt)

        # Array to store the "id" of the matched dt/gt
        gt_m = np.zeros((num_thrs, num_gt))
        dt_m = np.zeros((num_thrs, num_dt))

        gt_ig = np.array([g["_ignore"] for g in gt])
        dt_ig = np.zeros((num_thrs, num_dt))

        for iou_thr_idx, iou_thr in enumerate(self.params.iou_thrs):
            if len(ious) == 0:
                break

            for dt_idx, _dt in enumerate(dt):
                iou = min([iou_thr, 1 - 1e-10])
                # information about best match so far (m=-1 -> unmatched)
                # store the gt_idx which matched for _dt
                m = -1
                for gt_idx, _ in enumerate(gt):
                    # if this gt already matched continue
                    if gt_m[iou_thr_idx, gt_idx] > 0:
                        continue
                    # if _dt matched to reg gt, and on ignore gt, stop
                    if m > -1 and gt_ig[m] == 0 and gt_ig[gt_idx] == 1:
                        break
                    # continue to next gt unless better match made
                    if ious[dt_idx, gt_idx] < iou:
                        continue
                    # if match successful and best so far, store appropriately
                    iou = ious[dt_idx, gt_idx]
                    m = gt_idx

                # No match found for _dt, go to next _dt
                if m == -1:
                    continue

                # if gt to ignore for some reason update dt_ig.
                # Should not be used in evaluation.
                dt_ig[iou_thr_idx, dt_idx] = gt_ig[m]
                # _dt match found, update gt_m, and dt_m with "id"
                dt_m[iou_thr_idx, dt_idx] = gt[m]["id"]
                gt_m[iou_thr_idx, m] = _dt["id"]

        # For LVIS we will ignore any unmatched detection if that category was
        # not exhaustively annotated in gt.
        dt_ig_mask = [
            d["area"] < area_rng[0]
            or d["area"] > area_rng[1]
            or d["category_id"] in self.img_nel[d["image_id"]]
            for d in dt
        ]
        dt_ig_mask = np.array(dt_ig_mask).reshape((1, num_dt))  # 1 X num_dt
        dt_ig_mask = np.repeat(dt_ig_mask, num_thrs, 0)  # num_thrs X num_dt
        # Based on dt_ig_mask ignore any unmatched detection by updating dt_ig
        dt_ig = np.logical_or(dt_ig, np.logical_and(dt_m == 0, dt_ig_mask))
        # store results for given image and category
        return {
            "image_id": img_id,
            "category_id": cat_id,
            "area_rng": area_rng,
            "dt_ids": [d["id"] for d in dt],
            "gt_ids": [g["id"] for g in gt],
            "dt_matches": dt_m,
            "gt_matches": gt_m,
            "dt_scores": [d["score"] for d in dt],
            "gt_ignore": gt_ig,
            "dt_ignore": dt_ig,
        }

    def accumulate(self):
        """Accumulate per image evaluation results and store the result in
        self.eval.
        """
        self.logger.info("Accumulating evaluation results.")

        if not self.eval_imgs:
            self.logger.warn("Please run evaluate first.")

        if self.params.use_cats:
            cat_ids = self.params.cat_ids
        else:
            cat_ids = [-1]

        num_thrs = len(self.params.iou_thrs)
        num_recalls = len(self.params.rec_thrs)
        num_cats = len(cat_ids)
        num_area_rngs = len(self.params.area_rng)
        num_imgs = len(self.params.img_ids)

        # -1 for absent categories
        precision = -np.ones(
            (num_thrs, num_recalls, num_cats, num_area_rngs)
        )
        recall = -np.ones((num_thrs, num_cats, num_area_rngs))

        # Initialize dt_pointers
        dt_pointers = {}
        for cat_idx in range(num_cats):
            dt_pointers[cat_idx] = {}
            for area_idx in range(num_area_rngs):
                dt_pointers[cat_idx][area_idx] = {}

        # Per category evaluation
        for cat_idx in range(num_cats):
            Nk = cat_idx * num_area_rngs * num_imgs
            for area_idx in range(num_area_rngs):
                Na = area_idx * num_imgs
                E = [
                    self.eval_imgs[Nk + Na + img_idx]
                    for img_idx in range(num_imgs)
                ]
                # Remove elements which are None
                E = [e for e in E if not e is None]
                if len(E) == 0:
                    continue

                # Append all scores: shape (N,)
                dt_scores = np.concatenate([e["dt_scores"] for e in E], axis=0)
                dt_ids = np.concatenate([e["dt_ids"] for e in E], axis=0)

                dt_idx = np.argsort(-dt_scores, kind="mergesort")
                dt_scores = dt_scores[dt_idx]
                dt_ids = dt_ids[dt_idx]

                dt_m = np.concatenate([e["dt_matches"] for e in E], axis=1)[:, dt_idx]
                dt_ig = np.concatenate([e["dt_ignore"] for e in E], axis=1)[:, dt_idx]

                gt_ig = np.concatenate([e["gt_ignore"] for e in E])
                # num gt anns to consider
                num_gt = np.count_nonzero(gt_ig == 0)

                if num_gt == 0:
                    continue

                tps = np.logical_and(dt_m, np.logical_not(dt_ig))
                fps = np.logical_and(np.logical_not(dt_m), np.logical_not(dt_ig))

                tp_sum = np.cumsum(tps, axis=1).astype(dtype=np.float)
                fp_sum = np.cumsum(fps, axis=1).astype(dtype=np.float)

                dt_pointers[cat_idx][area_idx] = {
                    "dt_ids": dt_ids,
                    "tps": tps,
                    "fps": fps,
                }

                for iou_thr_idx, (tp, fp) in enumerate(zip(tp_sum, fp_sum)):
                    tp = np.array(tp)
                    fp = np.array(fp)
                    num_tp = len(tp)
                    rc = tp / num_gt
                    if num_tp:
                        recall[iou_thr_idx, cat_idx, area_idx] = rc[
                            -1
                        ]
                    else:
                        recall[iou_thr_idx, cat_idx, area_idx] = 0

                    # np.spacing(1) ~= eps
                    pr = tp / (fp + tp + np.spacing(1))
                    pr = pr.tolist()

                    # Replace each precision value with the maximum precision
                    # value to the right of that recall level. This ensures
                    # that the  calculated AP value will be less suspectable
                    # to small variations in the ranking.
                    for i in range(num_tp - 1, 0, -1):
                        if pr[i] > pr[i - 1]:
                            pr[i - 1] = pr[i]

                    rec_thrs_insert_idx = np.searchsorted(
                        rc, self.params.rec_thrs, side="left"
                    )

                    pr_at_recall = [0.0] * num_recalls

                    try:
                        for _idx, pr_idx in enumerate(rec_thrs_insert_idx):
                            pr_at_recall[_idx] = pr[pr_idx]
                    except:
                        pass
                    precision[iou_thr_idx, :, cat_idx, area_idx] = np.array(pr_at_recall)

        self.eval = {
            "params": self.params,
            "counts": [num_thrs, num_recalls, num_cats, num_area_rngs],
            "date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "precision": precision,
            "recall": recall,
            "dt_pointers": dt_pointers,
        }

    def _summarize(
        self, summary_type, iou_thr=None, area_rng="all", freq_group_idx=None
    ):
        aidx = [
            idx
            for idx, _area_rng in enumerate(self.params.area_rng_lbl)
            if _area_rng == area_rng
        ]

        if summary_type == 'ap':
            s = self.eval["precision"]
            if iou_thr is not None:
                tidx = np.where(iou_thr == self.params.iou_thrs)[0]
                s = s[tidx]
            if freq_group_idx is not None:
                s = s[:, :, self.freq_groups[freq_group_idx], aidx]
            else:
                s = s[:, :, :, aidx]
        else:
            s = self.eval["recall"]
            if iou_thr is not None:
                tidx = np.where(iou_thr == self.params.iou_thrs)[0]
                s = s[tidx]
            s = s[:, :, aidx]

        if len(s[s > -1]) == 0:
            mean_s = -1
        else:
            mean_s = np.mean(s[s > -1])
        return mean_s

    def summarize(self):
        """Compute and display summary metrics for evaluation results."""
        if not self.eval:
            raise RuntimeError("Please run accumulate() first.")

        max_dets = self.params.max_dets

        self.results["AP"]   = self._summarize('ap')
        self.results["AP50"] = self._summarize('ap', iou_thr=0.50)
        self.results["AP75"] = self._summarize('ap', iou_thr=0.75)
        self.results["APs"]  = self._summarize('ap', area_rng="small")
        self.results["APm"]  = self._summarize('ap', area_rng="medium")
        self.results["APl"]  = self._summarize('ap', area_rng="large")
        self.results["APr"]  = self._summarize('ap', freq_group_idx=0)
        self.results["APc"]  = self._summarize('ap', freq_group_idx=1)
        self.results["APf"]  = self._summarize('ap', freq_group_idx=2)

        key = "AR@{}".format(max_dets)
        self.results[key] = self._summarize('ar')

        for area_rng in ["small", "medium", "large"]:
            key = "AR{}@{}".format(area_rng[0], max_dets)
            self.results[key] = self._summarize('ar', area_rng=area_rng)

    def run(self):
        """Wrapper function which calculates the results."""
        self.evaluate()
        self.accumulate()
        self.summarize()

    def print_results(self):
        template = " {:<18} {} @[ IoU={:<9} | area={:>6s} | maxDets={:>3d} catIds={:>3s}] = {:0.3f}"

        for key, value in self.results.items():
            max_dets = self.params.max_dets
            if "AP" in key:
                title = "Average Precision"
                _type = "(AP)"
            else:
                title = "Average Recall"
                _type = "(AR)"

            if len(key) > 2 and key[2].isdigit():
                iou_thr = (float(key[2:]) / 100)
                iou = "{:0.2f}".format(iou_thr)
            else:
                iou = "{:0.2f}:{:0.2f}".format(
                    self.params.iou_thrs[0], self.params.iou_thrs[-1]
                )

            if len(key) > 2 and key[2] in ["r", "c", "f"]:
                cat_group_name = key[2]
            else:
                cat_group_name = "all"

            if len(key) > 2 and key[2] in ["s", "m", "l"]:
                area_rng = key[2]
            else:
                area_rng = "all"

            print(template.format(title, _type, iou, area_rng, max_dets, cat_group_name, value))

    def get_results(self):
        if not self.results:
            self.logger.warn("results is empty. Call run().")
        return self.results


class Params:
    def __init__(self, iou_type):
        """Params for LVIS evaluation API."""
        self.img_ids = []
        self.cat_ids = []
        # np.arange causes trouble.  the data point on arange is slightly
        # larger than the true value
        self.iou_thrs = np.linspace(
            0.5, 0.95, int(np.round((0.95 - 0.5) / 0.05)) + 1, endpoint=True
        )
        self.rec_thrs = np.linspace(
            0.0, 1.00, int(np.round((1.00 - 0.0) / 0.01)) + 1, endpoint=True
        )
        self.max_dets = 300
        self.area_rng = [
            [0 ** 2, 1e5 ** 2],
            [0 ** 2, 32 ** 2],
            [32 ** 2, 96 ** 2],
            [96 ** 2, 1e5 ** 2],
        ]
        self.area_rng_lbl = ["all", "small", "medium", "large"]
        self.use_cats = 1
        # We bin categories in three bins based how many images of the training
        # set the category is present in.
        # r: Rare    :  < 10
        # c: Common  : >= 10 and < 100
        # f: Frequent: >= 100
        self.img_count_lbl = ["r", "c", "f"]
        self.iou_type = iou_type


class LVISEvaluator(DatasetEvaluator):
    """
    Evaluate object proposal and instance detection/segmentation outputs using
    LVIS's metrics and evaluation API.
    """

    def __init__(
        self,
        dataset_name,
        tasks=None,
        distributed=True,
        output_dir=None,
        *,
        max_dets_per_image=None,
    ):
        """
        Args:
            dataset_name (str): name of the dataset to be evaluated.
                It must have the following corresponding metadata:
                "json_file": the path to the LVIS format annotation
            tasks (tuple[str]): tasks that can be evaluated under the given
                configuration. A task is one of "bbox", "segm".
                By default, will infer this automatically from predictions.
            distributed (True): if True, will collect results from all ranks for evaluation.
                Otherwise, will evaluate the results in the current process.
            output_dir (str): optional, an output directory to dump results.
            max_dets_per_image (None or int): limit on maximum detections per image in evaluating AP
                This limit, by default of the LVIS dataset, is 300.
        """
        from lvis import LVIS

        self._logger = logging.getLogger(__name__)

        if tasks is not None and isinstance(tasks, CfgNode):
            self._logger.warn(
                "COCO Evaluator instantiated using config, this is deprecated behavior."
                " Please pass in explicit arguments instead."
            )
            self._tasks = None  # Infering it from predictions should be better
        else:
            self._tasks = tasks

        self._distributed = distributed
        self._output_dir = output_dir
        self._max_dets_per_image = max_dets_per_image

        self._cpu_device = torch.device("cpu")

        self._metadata = MetadataCatalog.get(dataset_name)
        json_file = PathManager.get_local_path(self._metadata.json_file)
        self._lvis_api = LVIS(json_file)
        # Test set json files do not contain annotations (evaluation must be
        # performed using the LVIS evaluation server).
        self._do_evaluation = len(self._lvis_api.get_ann_ids()) > 0
        file_path = "/home/ubuntu/efs/Detic/output/Detic/Detic_LI_CLIP_R5021k_640b64_4x_ft4x_max-size/inference_lvis_v1_val/instances_predictions.pth"
        with PathManager.open(file_path, "rb") as f:
            self._predictions = torch.load(f)

    def reset(self):
        file_path = "/home/ubuntu/efs/Detic/output/Detic/Detic_LI_CLIP_R5021k_640b64_4x_ft4x_max-size/inference_lvis_v1_val/instances_predictions.pth"
        with PathManager.open(file_path, "rb") as f:
            self._predictions = torch.load(f)

    def process(self, inputs, outputs):
        """
        Args:
            inputs: the inputs to a LVIS model (e.g., GeneralizedRCNN).
                It is a list of dict. Each dict corresponds to an image and
                contains keys like "height", "width", "file_name", "image_id".
            outputs: the outputs of a LVIS model. It is a list of dicts with key
                "instances" that contains :class:`Instances`.
        """
        pass

    def evaluate(self):
        if self._distributed:
            predictions = self._predictions

            if not comm.is_main_process():
                return
        else:
            predictions = self._predictions

        if len(predictions) == 0:
            self._logger.warning("[LVISEvaluator] Did not receive valid predictions.")
            return {}

        self._results = OrderedDict()
        if "proposals" in predictions[0]:
            self._eval_box_proposals(predictions)
        if "instances" in predictions[0]:
            self._eval_predictions(predictions)
        # Copy so the caller can do whatever with results
        return copy.deepcopy(self._results)

    def _tasks_from_predictions(self, predictions):
        for pred in predictions:
            if "segmentation" in pred:
                return ("bbox", "segm")
        return ("bbox",)

    def _eval_predictions(self, predictions):
        """
        Evaluate predictions. Fill self._results with the metrics of the tasks.

        Args:
            predictions (list[dict]): list of outputs from the model
        """
        self._logger.info("Preparing results in the LVIS format ...")
        lvis_results = list(itertools.chain(*[x["instances"] for x in predictions]))
        tasks = self._tasks or self._tasks_from_predictions(lvis_results)

        # LVIS evaluator can be used to evaluate results for COCO dataset categories.
        # In this case `_metadata` variable will have a field with COCO-specific category mapping.
        if hasattr(self._metadata, "thing_dataset_id_to_contiguous_id"):
            reverse_id_mapping = {
                v: k for k, v in self._metadata.thing_dataset_id_to_contiguous_id.items()
            }
            for result in lvis_results:
                result["category_id"] = reverse_id_mapping[result["category_id"]]
        else:
            # unmap the category ids for LVIS (from 0-indexed to 1-indexed)
            for result in lvis_results:
                result["category_id"] += 1

        if not self._do_evaluation:
            self._logger.info("Annotations are not available for evaluation.")
            return

        self._logger.info("Evaluating predictions ...")
        for task in sorted(tasks):
            res = _evaluate_predictions_on_lvis(
                self._lvis_api,
                lvis_results,
                task,
                max_dets_per_image=self._max_dets_per_image,
                class_names=self._metadata.get("thing_classes"),
            )
            self._results[task] = res

    def _eval_box_proposals(self, predictions):
        """
        Evaluate the box proposals in predictions.
        Fill self._results with the metrics for "box_proposals" task.
        """
        if self._output_dir:
            # Saving generated box proposals to file.
            # Predicted box_proposals are in XYXY_ABS mode.
            bbox_mode = BoxMode.XYXY_ABS.value
            ids, boxes, objectness_logits = [], [], []
            for prediction in predictions:
                ids.append(prediction["image_id"])
                boxes.append(prediction["proposals"].proposal_boxes.tensor.numpy())
                objectness_logits.append(prediction["proposals"].objectness_logits.numpy())

            proposal_data = {
                "boxes": boxes,
                "objectness_logits": objectness_logits,
                "ids": ids,
                "bbox_mode": bbox_mode,
            }
            with PathManager.open(os.path.join(self._output_dir, "box_proposals.pkl"), "wb") as f:
                pickle.dump(proposal_data, f)

        if not self._do_evaluation:
            self._logger.info("Annotations are not available for evaluation.")
            return

        self._logger.info("Evaluating bbox proposals ...")
        res = {}
        areas = {"all": "", "small": "s", "medium": "m", "large": "l"}
        for limit in [100, 1000]:
            for area, suffix in areas.items():
                stats = _evaluate_box_proposals(predictions, self._lvis_api, area=area, limit=limit)
                key = "AR{}@{:d}".format(suffix, limit)
                res[key] = float(stats["ar"].item() * 100)
        self._logger.info("Proposal metrics: \n" + create_small_table(res))
        self._results["box_proposals"] = res


# inspired from Detectron:
# https://github.com/facebookresearch/Detectron/blob/a6a835f5b8208c45d0dce217ce9bbda915f44df7/detectron/datasets/json_dataset_evaluator.py#L255 # noqa
def _evaluate_box_proposals(dataset_predictions, lvis_api, thresholds=None, area="all", limit=None):
    """
    Evaluate detection proposal recall metrics. This function is a much
    faster alternative to the official LVIS API recall evaluation code. However,
    it produces slightly different results.
    """
    # Record max overlap value for each gt box
    # Return vector of overlap values
    areas = {
        "all": 0,
        "small": 1,
        "medium": 2,
        "large": 3,
        "96-128": 4,
        "128-256": 5,
        "256-512": 6,
        "512-inf": 7,
    }
    area_ranges = [
        [0**2, 1e5**2],  # all
        [0**2, 32**2],  # small
        [32**2, 96**2],  # medium
        [96**2, 1e5**2],  # large
        [96**2, 128**2],  # 96-128
        [128**2, 256**2],  # 128-256
        [256**2, 512**2],  # 256-512
        [512**2, 1e5**2],
    ]  # 512-inf
    assert area in areas, "Unknown area range: {}".format(area)
    area_range = area_ranges[areas[area]]
    gt_overlaps = []
    num_pos = 0

    for prediction_dict in dataset_predictions:
        predictions = prediction_dict["proposals"]

        # sort predictions in descending order
        # TODO maybe remove this and make it explicit in the documentation
        inds = predictions.objectness_logits.sort(descending=True)[1]
        predictions = predictions[inds]

        ann_ids = lvis_api.get_ann_ids(img_ids=[prediction_dict["image_id"]])
        anno = lvis_api.load_anns(ann_ids)
        gt_boxes = [
            BoxMode.convert(obj["bbox"], BoxMode.XYWH_ABS, BoxMode.XYXY_ABS) for obj in anno
        ]
        gt_boxes = torch.as_tensor(gt_boxes).reshape(-1, 4)  # guard against no boxes
        gt_boxes = Boxes(gt_boxes)
        gt_areas = torch.as_tensor([obj["area"] for obj in anno])

        if len(gt_boxes) == 0 or len(predictions) == 0:
            continue

        valid_gt_inds = (gt_areas >= area_range[0]) & (gt_areas <= area_range[1])
        gt_boxes = gt_boxes[valid_gt_inds]

        num_pos += len(gt_boxes)

        if len(gt_boxes) == 0:
            continue

        if limit is not None and len(predictions) > limit:
            predictions = predictions[:limit]

        overlaps = pairwise_iou(predictions.proposal_boxes, gt_boxes)

        _gt_overlaps = torch.zeros(len(gt_boxes))
        for j in range(min(len(predictions), len(gt_boxes))):
            # find which proposal box maximally covers each gt box
            # and get the iou amount of coverage for each gt box
            max_overlaps, argmax_overlaps = overlaps.max(dim=0)

            # find which gt box is 'best' covered (i.e. 'best' = most iou)
            gt_ovr, gt_ind = max_overlaps.max(dim=0)
            assert gt_ovr >= 0
            # find the proposal box that covers the best covered gt box
            box_ind = argmax_overlaps[gt_ind]
            # record the iou coverage of this gt box
            _gt_overlaps[j] = overlaps[box_ind, gt_ind]
            assert _gt_overlaps[j] == gt_ovr
            # mark the proposal box and the gt box as used
            overlaps[box_ind, :] = -1
            overlaps[:, gt_ind] = -1

        # append recorded iou coverage level
        gt_overlaps.append(_gt_overlaps)
    gt_overlaps = (
        torch.cat(gt_overlaps, dim=0) if len(gt_overlaps) else torch.zeros(0, dtype=torch.float32)
    )
    gt_overlaps, _ = torch.sort(gt_overlaps)

    if thresholds is None:
        step = 0.05
        thresholds = torch.arange(0.5, 0.95 + 1e-5, step, dtype=torch.float32)
    recalls = torch.zeros_like(thresholds)
    # compute recall for each iou threshold
    for i, t in enumerate(thresholds):
        recalls[i] = (gt_overlaps >= t).float().sum() / float(num_pos)
    # ar = 2 * np.trapz(recalls, thresholds)
    ar = recalls.mean()
    return {
        "ar": ar,
        "recalls": recalls,
        "thresholds": thresholds,
        "gt_overlaps": gt_overlaps,
        "num_pos": num_pos,
    }


def _evaluate_predictions_on_lvis(
    lvis_gt, lvis_results, iou_type, max_dets_per_image=None, class_names=None
):
    """
    Args:
        iou_type (str):
        max_dets_per_image (None or int): limit on maximum detections per image in evaluating AP
            This limit, by default of the LVIS dataset, is 300.
        class_names (None or list[str]): if provided, will use it to predict
            per-category AP.

    Returns:
        a dict of {metric name: score}
    """
    metrics = {
        "bbox": ["AP", "AP50", "AP75", "APs", "APm", "APl", "APr", "APc", "APf"],
        "segm": ["AP", "AP50", "AP75", "APs", "APm", "APl", "APr", "APc", "APf"],
    }[iou_type]

    logger = logging.getLogger(__name__)

    if len(lvis_results) == 0:  # TODO: check if needed
        logger.warn("No predictions from the model!")
        return {metric: float("nan") for metric in metrics}

    if iou_type == "segm":
        lvis_results = copy.deepcopy(lvis_results)
        # When evaluating mask AP, if the results contain bbox, LVIS API will
        # use the box area as the area of the instance, instead of the mask area.
        # This leads to a different definition of small/medium/large.
        # We remove the bbox field to let mask AP use mask area.
        for c in lvis_results:
            c.pop("bbox", None)

    if max_dets_per_image is None:
        max_dets_per_image = 300  # Default for LVIS dataset

    from lvis import LVISResults

    logger.info(f"Evaluating with max detections per image = {max_dets_per_image}")
    lvis_results = LVISResults(lvis_gt, lvis_results, max_dets=max_dets_per_image)
    lvis_eval = LVISEval(lvis_gt, lvis_results, iou_type)
    lvis_eval.run()
    lvis_eval.print_results()

    # Pull the standard metrics from the LVIS results
    results = lvis_eval.get_results()
    results = {metric: float(results[metric] * 100) for metric in metrics}
    logger.info("Evaluation results for {}: \n".format(iou_type) + create_small_table(results))
    return results


def do_test(cfg, model):
    results = OrderedDict()
    for d, dataset_name in enumerate(cfg.DATASETS.TEST):
        if cfg.MODEL.RESET_CLS_TESTS:
            reset_cls_test(
                model,
                cfg.MODEL.TEST_CLASSIFIERS[d],
                cfg.MODEL.TEST_NUM_CLASSES[d])
        mapper = None if cfg.INPUT.TEST_INPUT_TYPE == 'default' \
            else DatasetMapper(
                cfg, False, augmentations=build_custom_augmentation(cfg, False))
        data_loader = build_detection_test_loader(cfg, dataset_name, mapper=mapper)
        output_folder = os.path.join(
            cfg.OUTPUT_DIR, "inference_{}".format(dataset_name))
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type

        if evaluator_type == "lvis" or cfg.GEN_PSEDO_LABELS:
            evaluator = LVISEvaluator(dataset_name, cfg, True, output_folder)
        elif evaluator_type == 'coco':
            if dataset_name == 'coco_generalized_zeroshot_val':
                # Additionally plot mAP for 'seen classes' and 'unseen classes'
                evaluator = CustomCOCOEvaluator(dataset_name, cfg, True, output_folder)
            else:
                evaluator = COCOEvaluator(dataset_name, cfg, True, output_folder)
        elif evaluator_type == 'oid':
            evaluator = OIDEvaluator(dataset_name, cfg, True, output_folder)
        else:
            assert 0, evaluator_type
            
        results[dataset_name] = evaluator.evaluate()
        if comm.is_main_process():
            logger.info("Evaluation results for {} in csv format:".format(
                dataset_name))
            print_csv_format(results[dataset_name])
    if len(results) == 1:
        results = list(results.values())[0]
    return results


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_centernet_config(cfg)
    add_detic_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    if '/auto' in cfg.OUTPUT_DIR:
        file_name = os.path.basename(args.config_file)[:-5]
        cfg.OUTPUT_DIR = cfg.OUTPUT_DIR.replace('/auto', '/{}'.format(file_name))
        logger.info('OUTPUT_DIR: {}'.format(cfg.OUTPUT_DIR))
    cfg.freeze()
    default_setup(cfg, args)
    setup_logger(output=cfg.OUTPUT_DIR, \
        distributed_rank=comm.get_rank(), name="detic")
    return cfg


def main(args):
    cfg = setup(args)

    model = build_model(cfg)
    logger.info("Model:\n{}".format(model))
    if args.eval_only:
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )

        return do_test(cfg, model)

    distributed = comm.get_world_size() > 1
    if distributed:
        model = DistributedDataParallel(
            model, device_ids=[comm.get_local_rank()], broadcast_buffers=False,
            find_unused_parameters=cfg.FIND_UNUSED_PARAM
        )

    return do_test(cfg, model)


if __name__ == "__main__":
    args = default_argument_parser()
    args = args.parse_args()
    if args.num_machines == 1:
        args.dist_url = 'tcp://127.0.0.1:{}'.format(
            torch.randint(11111, 60000, (1,))[0].item())
    else:
        if args.dist_url == 'host':
            args.dist_url = 'tcp://{}:12345'.format(
                os.environ['SLURM_JOB_NODELIST'])
        elif not args.dist_url.startswith('tcp'):
            tmp = os.popen(
                    'echo $(scontrol show job {} | grep BatchHost)'.format(
                        args.dist_url)
                ).read()
            tmp = tmp[tmp.find('=') + 1: -1]
            args.dist_url = 'tcp://{}:12345'.format(tmp)
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
