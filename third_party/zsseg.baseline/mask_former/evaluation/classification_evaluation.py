# Copyright (c) Facebook, Inc. and its affiliates.
import copy
import itertools
import logging
from collections import OrderedDict
import numpy as np
import torch
import detectron2.utils.comm as comm

from detectron2.data import MetadataCatalog
from detectron2.evaluation import DatasetEvaluator


class ClassificationEvaluator(DatasetEvaluator):
    def __init__(
        self,
        dataset_name,
        distributed=True,
    ):
        """
        Args:
            dataset_name (str): name of the dataset to be evaluated.
                It must have either the following corresponding metadata:

                    "json_file": the path to the COCO format annotation

                Or it must be in detectron2's standard dataset format
                so it can be converted to COCO format automatically.
            distributed (True): if True, will collect results from all ranks and run evaluation
                in the main process.
                Otherwise, will only evaluate the results in the current process.
            output_dir (str): optional, an output directory to dump all
                results predicted on the dataset. The dump contains two files:

                1. "instances_predictions.pth" a file that can be loaded with `torch.load` and
                   contains all the results in the format they are produced by the model.
                2. "coco_instances_results.json" a json file in COCO's result format.
        """
        self._logger = logging.getLogger(__name__)
        self._distributed = distributed

        self._cpu_device = torch.device("cpu")

        self._metadata = MetadataCatalog.get(dataset_name)
        try:
            self._evaluation_set = self._metadata.evaluation_set
        except AttributeError:
            self._evaluation_set = None
        self.dataset_name = dataset_name
        self._predictions = []
        self._gts = []

    def reset(self):
        self._predictions = []
        self._gts = []

    def process(self, inputs, outputs):
        """
        Args:
            inputs: the inputs to a classification model.
                It is a list of dict. Each dict corresponds to an image and
                contains keys like "height", "width", "file_name", "image_id".
            outputs: the outputs of a classification model. It is a list of dicts with key
                "instances" that contains :class:`Instances`.
        """
        for input, output in zip(inputs, outputs):
            prediction = {}
            if ("classification" in output) and (output["classification"] is not None):
                prediction["classification"] = output["classification"].cpu()
            if len(prediction) > 0:
                self._predictions.append(prediction["classification"])
                if "instances" in input:
                    self._gts.append(input["instances"].gt_classes.cpu())
                else:
                    self._gts.append(input["category_id"].cpu())

    def evaluate(self, img_ids=None):
        """
        Args:
            img_ids: a list of image IDs to evaluate on. Default to None for the whole dataset
        """
        if self._distributed:
            comm.synchronize()
            predictions = comm.gather(self._predictions, dst=0)
            predictions = list(itertools.chain(*predictions))
            gts = comm.gather(self._gts, dst=0)
            gts = list(itertools.chain(*gts))
            if not comm.is_main_process():
                return {}
        else:
            predictions = self._predictions
            gts = self._gts

        if len(predictions) == 0:
            self._logger.warning(
                "[ClassificationEvaluator] Did not receive valid predictions."
            )
            return {}
        self._results = OrderedDict()
        class_names = self._metadata.stuff_classes
        predictions = torch.cat(predictions)
        gts = torch.cat(gts)
        top1_acc, top5_acc = accuracy(predictions, gts, (1, 5))
        self._results["macc_top1"] = top1_acc.mean() * 100
        self._results["macc_top5"] = top5_acc.mean() * 100
        mmacc_top1 = []
        mmacc_top5 = []
        count_mat = np.zeros((len(class_names), len(class_names)))
        count_list = []
        for i in range(len(class_names)):
            flag = gts == i
            if flag.sum() == 0:
                count_list.append(i)
                continue
            res_id = predictions[gts == i].argmax(dim=-1)
            for j in range(len(class_names)):
                count_mat[i, j] += (res_id == j).float().sum()

        for i, name in enumerate(class_names):
            macc_top1_per_cls = safe_mean(top1_acc[gts == i]) * 100
            macc_top5_per_cls = safe_mean(top5_acc[gts == i]) * 100
            self._results[f"acc_top1_{name}"] = macc_top1_per_cls
            self._results[f"acc_top5_{name}"] = macc_top5_per_cls
            mmacc_top1.append(macc_top1_per_cls)
            mmacc_top5.append(macc_top5_per_cls)
        if self._evaluation_set is not None:

            for set_name, set_inds in self._evaluation_set.items():

                self._results[f"acc_top1_{set_name}"] = (
                    torch.cat([top1_acc[gts == i] for i in set_inds]).mean() * 100
                )
                self._results[f"acc_top1_w/o_{set_name}"] = (
                    torch.cat(
                        [
                            top1_acc[gts == i]
                            for i in range(len(class_names))
                            if i not in set_inds
                        ]
                    ).mean()
                    * 100
                )
        self._results["mmacc_top1"] = sum(mmacc_top1) / len(mmacc_top1)
        self._results["mmacc_top5"] = sum(mmacc_top5) / len(mmacc_top5)
        return copy.deepcopy(self._results)


def safe_mean(vec):
    if len(vec) == 0:
        return torch.Tensor([0])
    else:
        return vec.mean()


def accuracy(output: torch.Tensor, target: torch.Tensor, topk=(1,)):
    """
    Computes the accuracy over the k top predictions for the specified values of k
    In top-5 accuracy you give yourself credit for having the right answer
    if the right answer appears in your top five guesses.
    """
    maxk = max(topk)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = (pred == target.unsqueeze(dim=0)).expand_as(pred)

    res = []
    for k in topk:
        correct_k = correct[:k].float().sum(0)
        res.append(correct_k)
    return res
