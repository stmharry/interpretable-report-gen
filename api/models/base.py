import logging
import numpy as np
import operator
import os
import torch
import torch.multiprocessing as multiprocessing
import torch.nn as nn
import tqdm

from api import Phase

logger = logging.getLogger(__name__)
multiprocessing.set_sharing_strategy('file_system')


class DeviceMixin:
    def __init__(self):
        self.device = None

    def to(self, device):
        self.device = device
        return self


class Module(nn.Module):
    def forward(self, batch, phase=None, **kwargs):
        if phase == Phase.train:
            return self._train(batch, **kwargs)
        elif phase == Phase.val:
            return self._val(batch, **kwargs)
        elif phase == Phase.test:
            return self._test(batch, **kwargs)


""" Helper function for `multiprocessing'

Has to be here to be pickle-able. Simply *sad*. And guess what? GLOBAL!
"""

_model_dict = {}


def _pool_initializer(model_cls):
    logger.debug(f'[{os.getpid()}] DataParallelCPU.initializer')

    global _model_dict
    _model_dict[model_cls.__name__] = model_cls()


def _pool_func(args):
    num = args[0]
    model_cls = args[1]
    args = args[2:]

    logger.debug(f'[{os.getpid()}] DataParallelCPU.func(num={num}, model_cls={model_cls}, args={args})')

    global _model_dict
    return (num, _model_dict[model_cls.__name__](*args))


class DataParallelCPU(DeviceMixin):
    """ Mock interface for parallelism on CPU.

    Unfortunately this requires `global' keyword to work as current `multiprocessing'
    is not quite class-friendly.

    ### IMPORTANT NOTE ###
    If you have modules returning tensors on CPUs, check out
        `https://pytorch.org/docs/stable/multiprocessing.html#sharing-strategies',
    as your system can be easily crippled by queueing a lot of CPU tensors.
    """

    def __init__(self, model_cls, num_jobs=None, maxtasksperchild=256, verbose=False):
        super(DataParallelCPU, self).__init__()

        self.model_cls = model_cls
        self.verbose = verbose

        num_jobs = num_jobs or multiprocessing.cpu_count()
        self.pool = multiprocessing.Pool(
            num_jobs,
            initializer=_pool_initializer,
            initargs=(model_cls,),
            maxtasksperchild=maxtasksperchild,
        )

        logger.info(f'DataParallelCPU using {num_jobs} processes')

    def scatter(self, obj):
        if isinstance(obj, np.ndarray):
            return np.split(obj, len(obj))
        if isinstance(obj, torch.Tensor):
            return torch.split(obj, len(obj))
        if isinstance(obj, (tuple, list)):
            return list(map(type(obj), zip(*map(self.scatter, obj))))
        if isinstance(obj, dict):
            return list(map(type(obj), [zip(obj.keys(), values) for values in zip(*map(self.scatter, obj.values()))]))

    def gather(self, objs):
        obj = objs[0]
        if isinstance(obj, np.ndarray):
            return np.concatenate(objs)
        if isinstance(obj, torch.Tensor):
            return torch.cat(objs)
        if isinstance(obj, (tuple, list)):
            return type(obj)(map(self.gather, zip(*objs)))
        if isinstance(obj, dict):
            return type(obj)([(key, self.gather(map(operator.itemgetter(key), objs))) for key in obj.keys()])

    def __call__(self, *args):
        args = self.scatter(args)
        if self.verbose:
            args = tqdm.tqdm(total=len(args))

        objs = self.pool.imap_unordered(_pool_func, (
            (num, self.model_cls,) + _args
            for (num, _args) in enumerate(args)
        ))
        objs = dict(objs)
        objs = [objs[num] for num in range(len(objs))]
        objs = self.gather(objs).to(self.device)

        return objs


class ExponentialMovingAverage(nn.Module):
    def __init__(self, beta=0.95):
        super(ExponentialMovingAverage, self).__init__()

        self.beta = beta
        self.average = None

    def __call__(self, x):
        if self.average is None:
            self.average = x
        else:
            self.average = self.beta * self.average + (1 - self.beta) * x

        return self.average
