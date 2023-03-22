import copy
import numpy as np
import random
from collections import defaultdict
import torch
import math
from torch.utils.data.sampler import Sampler, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler


class RandomDomainSampler(Sampler):
    """Randomly samples N domains each with K images
    to form a minibatch of size N*K.

    Args:
        data_source (list): list of Datums.
        batch_size (int): batch size.
        n_domain (int): number of domains to sample in a minibatch.
    """

    def __init__(self, data_source, batch_size, n_domain):
        self.data_source = data_source

        # Keep track of image indices for each domain
        self.domain_dict = defaultdict(list)
        for i, item in enumerate(data_source):
            self.domain_dict[item.domain].append(i)
        self.domains = list(self.domain_dict.keys())

        # Make sure each domain has equal number of images
        if n_domain is None or n_domain <= 0:
            n_domain = len(self.domains)
        assert batch_size % n_domain == 0
        self.n_img_per_domain = batch_size // n_domain

        self.batch_size = batch_size
        # n_domain denotes number of domains sampled in a minibatch
        self.n_domain = n_domain
        self.length = len(list(self.__iter__()))

    def __iter__(self):
        domain_dict = copy.deepcopy(self.domain_dict)
        final_idxs = []
        stop_sampling = False

        while not stop_sampling:
            selected_domains = random.sample(self.domains, self.n_domain)

            for domain in selected_domains:
                idxs = domain_dict[domain]
                selected_idxs = random.sample(idxs, self.n_img_per_domain)
                final_idxs.extend(selected_idxs)

                for idx in selected_idxs:
                    domain_dict[domain].remove(idx)

                remaining = len(domain_dict[domain])
                if remaining < self.n_img_per_domain:
                    stop_sampling = True

        return iter(final_idxs)

    def __len__(self):
        return self.length


class SeqDomainSampler(Sampler):
    """Sequential domain sampler, which randomly samples K
    images from each domain to form a minibatch.

    Args:
        data_source (list): list of Datums.
        batch_size (int): batch size.
    """

    def __init__(self, data_source, batch_size):
        self.data_source = data_source

        # Keep track of image indices for each domain
        self.domain_dict = defaultdict(list)
        for i, item in enumerate(data_source):
            self.domain_dict[item.domain].append(i)
        self.domains = list(self.domain_dict.keys())
        self.domains.sort()

        # Make sure each domain has equal number of images
        n_domain = len(self.domains)
        assert batch_size % n_domain == 0
        self.n_img_per_domain = batch_size // n_domain

        self.batch_size = batch_size
        # n_domain denotes number of domains sampled in a minibatch
        self.n_domain = n_domain
        self.length = len(list(self.__iter__()))

    def __iter__(self):
        domain_dict = copy.deepcopy(self.domain_dict)
        final_idxs = []
        stop_sampling = False

        while not stop_sampling:
            for domain in self.domains:
                idxs = domain_dict[domain]
                selected_idxs = random.sample(idxs, self.n_img_per_domain)
                final_idxs.extend(selected_idxs)

                for idx in selected_idxs:
                    domain_dict[domain].remove(idx)

                remaining = len(domain_dict[domain])
                if remaining < self.n_img_per_domain:
                    stop_sampling = True

        return iter(final_idxs)

    def __len__(self):
        return self.length


class RandomClassSampler(Sampler):
    """Randomly samples N classes each with K instances to
    form a minibatch of size N*K.

    Modified from https://github.com/KaiyangZhou/deep-person-reid.

    Args:
        data_source (list): list of Datums.
        batch_size (int): batch size.
        n_ins (int): number of instances per class to sample in a minibatch.
    """

    def __init__(self, data_source, batch_size, n_ins):
        if batch_size < n_ins:
            raise ValueError(
                "batch_size={} must be no less "
                "than n_ins={}".format(batch_size, n_ins)
            )

        self.data_source = data_source
        self.batch_size = batch_size
        self.n_ins = n_ins
        self.ncls_per_batch = self.batch_size // self.n_ins
        self.index_dic = defaultdict(list)
        for index, item in enumerate(data_source):
            self.index_dic[item.label].append(index)
        self.labels = list(self.index_dic.keys())
        assert len(self.labels) >= self.ncls_per_batch

        # estimate number of images in an epoch
        self.length = len(list(self.__iter__()))

    def __iter__(self):
        batch_idxs_dict = defaultdict(list)

        for label in self.labels:
            idxs = copy.deepcopy(self.index_dic[label])
            if len(idxs) < self.n_ins:
                idxs = np.random.choice(idxs, size=self.n_ins, replace=True)
            random.shuffle(idxs)
            batch_idxs = []
            for idx in idxs:
                batch_idxs.append(idx)
                if len(batch_idxs) == self.n_ins:
                    batch_idxs_dict[label].append(batch_idxs)
                    batch_idxs = []

        avai_labels = copy.deepcopy(self.labels)
        final_idxs = []

        while len(avai_labels) >= self.ncls_per_batch:
            selected_labels = random.sample(avai_labels, self.ncls_per_batch)
            for label in selected_labels:
                batch_idxs = batch_idxs_dict[label].pop(0)
                final_idxs.extend(batch_idxs)
                if len(batch_idxs_dict[label]) == 0:
                    avai_labels.remove(label)

        return iter(final_idxs)

    def __len__(self):
        return self.length


class SequentialDistributedSampler(torch.utils.data.sampler.Sampler):
    """
    Distributed Sampler that subsamples indicies sequentially,
    making it easier to collate all results at the end.
    Even though we only use this sampler for eval and predict (no training),
    which means that the model params won't have to be synced (i.e. will not hang
    for synchronization even if varied number of forward passes), we still add extra
    samples to the sampler to make it evenly divisible (like in `DistributedSampler`)
    to make it easy to `gather` or `reduce` resulting tensors at the end of the loop.
    """

    def __init__(self, dataset, batch_size, rank=None, num_replicas=None):
        if num_replicas is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = torch.distributed.get_world_size()
        if rank is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = torch.distributed.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.batch_size = batch_size
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.batch_size / self.num_replicas)) * self.batch_size
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        indices = list(range(len(self.dataset)))
        # add extra samples to make it evenly divisible
        indices += [indices[-1]] * (self.total_size - len(indices))
        # subsample
        indices = indices[self.rank * self.num_samples : (self.rank + 1) * self.num_samples]
        return iter(indices)

    def __len__(self):
        return self.num_samples


def build_sampler(
    sampler_type,
    cfg=None,
    data_source=None,
    batch_size=32,
    n_domain=0,
    n_ins=16
):
    if sampler_type == "RandomSampler":
        return RandomSampler(data_source)

    elif sampler_type == "SequentialSampler":
        return SequentialSampler(data_source)

    elif sampler_type == "RandomDomainSampler":
        return RandomDomainSampler(data_source, batch_size, n_domain)

    elif sampler_type == "SeqDomainSampler":
        return SeqDomainSampler(data_source, batch_size)

    elif sampler_type == "RandomClassSampler":
        return RandomClassSampler(data_source, batch_size, n_ins)

    elif sampler_type == "DistributedSampler":
        return DistributedSampler(data_source)
    elif sampler_type == "SequentialDistributedSampler":
        return SequentialDistributedSampler(data_source, batch_size)

    else:
        raise ValueError("Unknown sampler type: {}".format(sampler_type))
