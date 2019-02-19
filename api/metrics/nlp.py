import numpy as np
import torch

from pyciderevalcap3.ciderD.ciderD import CiderD as _CiderD
from pyciderevalcap3.ciderD.ciderD_scorer import CiderScorer as _CiderScorer
from pycocoevalcap3.bleu.bleu import Bleu as _Bleu
from pycocoevalcap3.bleu.bleu_scorer import BleuScorer
from pycocoevalcap3.rouge.rouge import Rouge as _Rouge

from api.models.base import DeviceMixin


class MetricMixin(DeviceMixin):
    def __call__(self, input_, target):
        (_, scores) = self.compute_score(
            {num: [_target] for (num, _target) in enumerate(target)},
            {num: [_input] for (num, _input) in enumerate(input_)},
        )

        score = torch.as_tensor(scores, dtype=torch.float, device=self.device)

        return score


class CiderScorer(_CiderScorer):
    def compute_score(self, df_mode, option=None, verbose=0):
        score = self.compute_cider(df_mode)
        return np.mean(np.array(score)), np.array(score)


class CiderD(_CiderD, MetricMixin):
    def __init__(self, df_cache, *args, **kwargs):
        super(CiderD, self).__init__(*args, **kwargs)

        self.df_cache = df_cache

    def compute_score(self, gts, res):
        cider_scorer = CiderScorer(n=self._n, sigma=self._sigma)
        cider_scorer.document_frequency = self.df_cache['document_frequency']
        cider_scorer.ref_len = self.df_cache['ref_len']

        for id in gts:
            cider_scorer += (res[id][0], gts[id])

        return cider_scorer.compute_score(self._df)


class Bleu(_Bleu, MetricMixin):
    def compute_score(self, gts, res):
        bleu_scorer = BleuScorer(n=self._n)

        for id in gts:
            bleu_scorer += (res[id][0], gts[id])

        return bleu_scorer.compute_score(option='closest')


class Rouge(_Rouge, MetricMixin):
    pass
