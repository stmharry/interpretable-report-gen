import io
import numpy as np
import psutil
import os
import re
import grequests
import torch
import torch.nn as nn

import chexpert_labeler

from api import Token
from api.models.base import DeviceMixin
from api.utils import to_numpy
from api.utils.rnn import pack_padded_sequence, pad_packed_sequence


class SentIndex2Report(object):
    def __init__(self, index_to_word):
        self.index_to_word = index_to_word

    def forward(self, sent, sent_length, text_length, is_eos=True):
        if not is_eos:
            sent_length = sent_length - 1

        word = pack_padded_sequence(sent, length=sent_length)
        length = pad_packed_sequence(sent_length, length=text_length).sum(1)

        word = self.index_to_word[to_numpy(word)]
        words = np.split(word, np.cumsum(to_numpy(length)))[:-1]

        return np.array(list(map(' '.join, words)), dtype=object)

    __call__ = forward


class CheXpert(DeviceMixin):
    def __init__(self):
        super(CheXpert, self).__init__()

        self.extractor = chexpert_labeler.Extractor()
        self.classifier = chexpert_labeler.Classifier()
        self.aggregator = chexpert_labeler.Aggregator()

        self.re_objs = {}
        self.process = psutil.Process(os.getpid())

        self('CheXpert initializing.')

    def get_re_obj(self, pattern):
        self.re_objs[pattern] = self.re_objs.get(pattern) or re.compile(pattern)

        return self.re_objs[pattern]

    def clean(self, s):
        s = self.get_re_obj(Token.eos).sub(' ', s)
        s = self.get_re_obj(r'\.\s*\.').sub('.', s)
        return s

    def forward(self, s):
        """ Label radiology reports.

        Args:
            s: numpy array of strings.

        Returns:
            labels (torch.int64): annotation of diseases, the meaning of which is
                1) 3: potisitive mention
                2) 2: negative mention
                3) 1: uncertain mention
                4) 0: no mention

        """

        if not isinstance(s, np.ndarray):
            s = np.array([s], dtype=object)

        size = len(s)

        # stmharry: extra space to ensure the string is not empty
        s = '"' + s + ' "'
        s = '\n'.join(s)
        s = self.clean(s)

        try:
            with io.BytesIO(s.encode()) as f:
                loader = chexpert_labeler.Loader(reports_path=f)
                loader.load()

            self.extractor.extract(loader.collection)
            self.classifier.classify(loader.collection)
            labels = self.aggregator.aggregate(loader.collection)

        except Exception:
            labels = np.asarray([[1] + [np.nan] * (len(chexpert_labeler.CATEGORIES) - 1)] * size)

        labels = torch.as_tensor(labels, device=self.device)
        labels = torch.where(torch.isnan(labels), torch.zeros_like(labels), labels + 2).long()

        return labels

    __call__ = forward


class CheXpertAggregator(nn.Module):
    def __init__(self):
        super(CheXpertAggregator, self).__init__()

        self._importance_lookup = nn.Parameter(torch.as_tensor([0, 2, 1, 3]), requires_grad=False)
        self._inv_importance_lookup = self._importance_lookup

    def forward(self, chexpert_label_sent, text_length):
        chexpert_label_sents = chexpert_label_sent.split(text_length.tolist(), 0)

        chexpert_labels = []
        for chexpert_label_sent in chexpert_label_sents:
            chexpert_label = self._importance_lookup[chexpert_label_sent].max(0)[0]
            chexpert_label[0] = 3 * chexpert_label[1:13].max().lt(2).long()
            chexpert_label = self._inv_importance_lookup[chexpert_label]

            chexpert_labels.append(chexpert_label)

        return torch.stack(chexpert_labels, 0)


class CheXpertRemote(DeviceMixin):
    def __init__(self, urls, local):
        super(CheXpertRemote).__init__()

        self.urls = urls
        self.local = local

    def forward(self, x):
        if not isinstance(x, np.ndarray):
            x = np.array([x], dtype=object)

        xs = np.array_split(x, len(self.urls))
        objs = [grequests.post(url, json=x.tolist()) for (url, x) in zip(self.urls, xs)]
        objs = grequests.map(objs)
        for (num, obj) in enumerate(objs):
            if obj is None:
                objs[num] = self.local(xs[num])
            else:
                objs[num] = torch.as_tensor(obj.json(), dtype=torch.long, device=self.device)

        return torch.cat(objs)

    __call__ = forward
