import io
import numpy as np
import os
import torch

import chexpert

from api import Token
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


class CheXpert(object):
    def __init__(self):
        self.extractor = chexpert.Extractor()
        self.classifier = chexpert.Classifier()
        self.aggregator = chexpert.Aggregator()

    def forward(self, s):
        """ Label radiology reports.

        Args:
            s: array of strings.

        Returns:
            labels (torch.float32): annotation of diseases, the meaning of which is
                1) 3: potisitive mention
                2) 2: negative mention
                3) 1: uncertain mention
                4) 0: no mention

        """

        s = '"' + s + '"'
        s = '\n'.join(s)
        s = s.replace(Token.eos, '')

        f = io.BytesIO(s.encode())
        loader = chexpert.Loader(reports_path=f)
        loader.load()

        self.extractor.extract(loader.collection)
        self.classifier.classify(loader.collection)
        labels = self.aggregator.aggregate(loader.collection)

        labels = torch.tensor(labels).float()
        labels = torch.where(torch.isnan(labels), torch.zeros_like(labels), labels + 2).long()

        return labels

    __call__ = forward
