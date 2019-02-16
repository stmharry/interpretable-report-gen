import joblib
import numpy as np
import torch.nn as nn

from api import Phase


class Module(nn.Module):
    def forward(self, batch, phase=None, **kwargs):
        if phase == Phase.train:
            return self._train(batch, **kwargs)
        elif phase == Phase.val:
            return self._val(batch, **kwargs)
        elif phase == Phase.test:
            return self._test(batch, **kwargs)


class DataParallelCPU(object):
    def __init__(self, model, num_jobs):
        self.model = model
        self.parallel = joblib.Parallel(n_jobs=num_jobs)

    def scatter(self, obj):
        if isinstance(obj, np.ndarray):
            return np.split(obj, len(obj))
        if isinstance(obj, (tuple, list)):
            return list(map(type(obj), zip(*map(self.scatter, obj))))
        if isinstance(obj, dict):
            return list(map(type(obj), [zip(obj.keys(), values) for values in zip(*map(self.scatter, obj.values()))]))

    def __call__(self, *args):
        args = self.scatter(args)

        results = self.parallel(joblib.delayed(self.model.__call__)(*_args) for _args in args)
        return np.concatenate(results)
