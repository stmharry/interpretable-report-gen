import numpy as np

from pyciderevalcap3.ciderD.ciderD import CiderD as _CiderD
from pyciderevalcap3.ciderD.ciderD_scorer import CiderScorer as _CiderScorer


class CiderScorer(_CiderScorer):
    def compute_score(self, df_mode, option=None, verbose=0):
        score = self.compute_cider(df_mode)
        return np.mean(np.array(score)), np.array(score)


class CiderD(_CiderD):
    def __init__(self, df_cache, *args, **kwargs):
        super(CiderD, self).__init__(*args, **kwargs)

        self.df_cache = df_cache

    def compute_score(self, gts, res):
        cider_scorer = CiderScorer(n=self._n, sigma=self._sigma)
        cider_scorer.document_frequency = self.df_cache['document_frequency']
        cider_scorer.ref_len = np.log(float(self.df_cache['ref_len']))

        for id in gts:
            cider_scorer += (res[id][0], gts[id])

        (score, scores) = cider_scorer.compute_score(self._df)

        return (score, scores)


from pycocoevalcap3.bleu.bleu import Bleu
from pycocoevalcap3.meteor.meteor import Meteor
from pycocoevalcap3.rouge.rouge import Rouge
from pycocoevalcap3.spice.spice import Spice
