import torch
import numpy as np
import os
import pandas as pd
import re
import sys
import tqdm

from absl import flags

import chexpert_labeler

from api.models.base import DataParallelCPU
from api.models.nondiff import CheXpert
from api.metrics import Bleu, Rouge, CiderD as Cider, MentionSim
from api.utils import to_numpy

flags.DEFINE_string('csv_path', None, 'csv file to evaluate')
flags.DEFINE_list('remove_tokens', [], 'Tokens to strip, separated by comma')
FLAGS = flags.FLAGS


if __name__ == '__main__':
    argv = FLAGS(sys.argv)

    re_objs = [re.compile(token) for token in FLAGS.remove_tokens]

    bleu = Bleu(4)
    rouge = Rouge()
    cider = Cider(df_cache=torch.load(os.path.join(os.getenv('CACHE_DIR'), 'cider-cache.pkl')))

    _df = pd.read_csv(FLAGS.csv_path, sep='\t')
    df_sentence = pd.read_csv(os.path.join(os.getenv('CACHE_DIR'), f'sentence-label-field-findings.tsv'), sep='\t')
    df_chexpert = pd.read_csv(os.path.join(os.getenv('CACHE_DIR'), f'report-chexpert-field-findings.tsv'), sep='\t')

    df = pd.merge(
        df_sentence.loc[df_sentence.rad_id.isin(_df.rad_id)].groupby('rad_id').sentence.apply(' '.join).rename('text').reset_index(),
        df_chexpert.loc[df_chexpert.rad_id.isin(_df.rad_id)],
    )
    df = _df[['rad_id']].merge(df)

    print('Evaluating NLP metrics...')

    metric = pd.DataFrame(index=range(len(_df)))
    for index in tqdm.trange(len(_df)):
        _text = _df.text.iloc[index]
        for re_obj in re_objs:
            _text = re_obj.sub('', _text)

        text = df.text.iloc[index]

        for scorer in [bleu, rouge, cider]:
            report_score = scorer([_text], [text])

            if report_score.dim() == 2:
                for (num, _report_score) in enumerate(report_score):
                    metric.loc[index, f'{scorer.method()}-{num + 1}'] = _report_score.mean().item()
            else:
                metric.loc[index, f'{scorer.method()}'] = report_score.mean().item()

    print('Evaluating CheXpert label...')

    label = df[chexpert_labeler.CATEGORIES].values
    if all(map(_df.columns.__contains__, chexpert_labeler.CATEGORIES)):
        _label = _df[chexpert_labeler.CATEGORIES].values
    else:
        chexpert = DataParallelCPU(CheXpert, num_jobs=None, maxtasksperchild=256, verbose=True)
        _label = chexpert(_df.text.values)

    acc = np.array([
        1.0, np.nan, 1.0, 0.0,
        0.5, np.nan, 0.5, 0.5,
        1.0, np.nan, 1.0, 0.0,
        0.0, np.nan, 0.0, 1.0,
    ])[_label * 4 + label]

    for (category, _acc) in zip(chexpert_labeler.CATEGORIES, acc.T):
        metric[category] = _acc

    metric_mean = metric.mean(axis=0)
    metric_mean['CheXpert'] = metric_mean[chexpert_labeler.CATEGORIES].mean()

    print('=' * 32)
    print(metric_mean)
    print('=' * 32)
