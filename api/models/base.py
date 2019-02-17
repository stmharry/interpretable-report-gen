import logging
import numpy as np
import operator
import os
import torch
import torch.multiprocessing as multiprocessing
import torch.nn as nn

from api import Phase

logger = logging.getLogger(__name__)


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


def _pool_func(model_cls, *args, **kwargs):
    logger.debug(f'[{os.getpid()}] DataParallelCPU.func(model_cls={model_cls}, args={args})')
    global _model_dict
    return _model_dict[model_cls.__name__](*args, **kwargs)


class DataParallelCPU(object):
    """ Mock interface for parallelism on CPU.

    Unfortunately this requires `global' keyword to work as current `multiprocessing'
    is not quite class-friendly.
    """

    def __init__(self, model_cls, num_jobs):
        self.model_cls = model_cls
        self.pool = multiprocessing.Pool(num_jobs, initializer=_pool_initializer, initargs=(model_cls,))

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
            return type(out)(map(self.gather, zip(*objs)))
        if isinstance(obj, dict):
            return type(out)([(key, self.gather(map(operator.itemgetter(key), objs))) for key in obj.keys()])

    def __call__(self, *args):
        args = self.scatter(args)
        objs = self.pool.starmap(_pool_func, [(self.model_cls,) + _args for _args in args])
        objs = self.gather(objs)
        return objs
