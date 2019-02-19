import io
import numpy as np
import psutil
import os
import re
import torch

import chexpert

from api import Token
from api.models.base import DeviceMixin
from api.utils import to_numpy
from api.utils.rnn import pack_padded_sequence, pad_packed_sequence


class SentIndex2Report(object):
    def __init__(self, index_to_word):
        self.index_to_word = index_to_word

    def forward(self, sent, sent_length, text_length):
        word = pack_padded_sequence(sent, length=sent_length)
        length = pad_packed_sequence(sent_length, length=text_length).sum(1)

        word = self.index_to_word[to_numpy(word)]
        words = np.split(word, np.cumsum(to_numpy(length)))[:-1]

        return np.array([' '.join(word) for word in words], dtype=object)

    __call__ = forward


class CheXpert(DeviceMixin):
    def __init__(self):
        super(CheXpert, self).__init__()

        self.extractor = chexpert.Extractor()
        self.classifier = chexpert.Classifier()
        self.aggregator = chexpert.Aggregator()

        self.re_objs = {}
        self.process = psutil.Process(os.getpid())

        self('CheXpert initializing.')

    def get_re_obj(self, pattern):
        self.re_objs[pattern] = self.re_objs.get(pattern) or re.compile(pattern)

        return self.re_objs[pattern]

    def clean(self, s):
        s = self.get_re_obj(Token.eos).sub('', s)
        s = self.get_re_obj(r'\.\s*\.').sub('.', s)
        return s

    def forward(self, s):
        """ Label radiology reports.

        Args:
            s: numpy array of strings.

        Returns:
            labels (np.int64): annotation of diseases, the meaning of which is
                1) 3: potisitive mention
                2) 2: negative mention
                3) 1: uncertain mention
                4) 0: no mention

        """

        if not isinstance(s, np.ndarray):
            s = np.array([s], dtype=object)

        s = '"' + s + '"'
        s = '\n'.join(s)
        s = self.clean(s)

        with io.BytesIO(s.encode()) as f:
            loader = chexpert.Loader(reports_path=f)
            loader.load()

        self.extractor.extract(loader.collection)
        self.classifier.classify(loader.collection)
        labels = self.aggregator.aggregate(loader.collection)

        labels = torch.as_tensor(labels, device=self.device)
        labels = torch.where(torch.isnan(labels), torch.zeros_like(labels), labels + 2).long()

        return labels

    __call__ = forward
