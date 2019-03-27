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

flags.DEFINE_string('do', None, '')
flags.DEFINE_enum('dataset', None, ['mimic-cxr', 'open-i'], 'Dataset to use')
flags.DEFINE_string('raw', None, '')
flags.DEFINE_string('cache', None, '')
flags.DEFINE_list('remove_tokens', [], '')
FLAGS = flags.FLAGS


def compile():
    re_objs = [re.compile(token) for token in FLAGS.remove_tokens]

    bleu = Bleu(4)
    rouge = Rouge()
    cider = Cider(df_cache=torch.load(os.path.join(cache_dir, 'cider-cache.pkl')))

    _df = pd.read_csv(FLAGS.raw, sep='\t').fillna('')
    _df = _df.rename(columns={'pred_text': 'text'})

    df_sentence = pd.read_csv(df_sentence_path, sep='\t')
    df_report = pd.read_csv(df_report_path, sep='\t')

    rad_ids = set(_df.rad_id) & set(df_sentence.rad_id)
    df = pd.merge(
        df_sentence.loc[df_sentence.rad_id.isin(rad_ids)].groupby('rad_id').sentence.apply(' '.join).rename('text').reset_index(),
        df_report.loc[df_report.rad_id.isin(rad_ids)].drop(columns='text', errors='ignore'),
        on='rad_id',
    )

    _df = _df[_df.rad_id.isin(rad_ids)]
    df = _df[['rad_id']].merge(df, on='rad_id', how='left')

    df_metric = pd.DataFrame(
        {'rad_id': _df.rad_id},
        index=range(len(_df)),
    )

    for index in tqdm.trange(len(_df)):
        _text = _df.text.iloc[index]
        for re_obj in re_objs:
            _text = re_obj.sub('', _text)

        text = df.text.iloc[index]

        for scorer in [bleu, rouge, cider]:
            report_score = scorer([_text], [text])

            if report_score.dim() == 2:
                for (num, _report_score) in enumerate(report_score):
                    df_metric.loc[index, f'{scorer.method()}-{num + 1}'] = _report_score.mean().item()
            else:
                df_metric.loc[index, f'{scorer.method()}'] = report_score.mean().item()

    print('Evaluating CheXpert label...')

    label = df[chexpert_labeler.CATEGORIES].values
    if all(map(_df.columns.__contains__, chexpert_labeler.CATEGORIES)):
        chexpert = None
        _label = _df[chexpert_labeler.CATEGORIES].values
    else:
        chexpert = DataParallelCPU(CheXpert, num_jobs=None, maxtasksperchild=256, verbose=True)
        _label = to_numpy(chexpert(_df.text.values))

    index = _label * 4 + label
    for (num, category) in enumerate(chexpert_labeler.CATEGORIES):
        df_metric[category] = index[:, num]

    df_metric.to_csv(FLAGS.cache, sep='\t', index=False)

    if chexpert is not None:
        chexpert.close()


def calc():
    df_metric = pd.read_csv(FLAGS.cache, sep='\t')

    tp = np.array([
        np.nan, np.nan, np.nan, 0.0,
        np.nan, np.nan, np.nan, 0.5,
        np.nan, np.nan, np.nan, 0.0,
        np.nan, np.nan, np.nan, 1.0,
    ])
    fn = 1 - tp

    fp = np.array([
        0.0, np.nan, 0.0, np.nan,
        0.5, np.nan, 0.5, np.nan,
        0.0, np.nan, 0.0, np.nan,
        1.0, np.nan, 1.0, np.nan,
    ])
    tn = 1 - fp

    index = df_metric[chexpert_labeler.CATEGORIES]
    tp = np.nansum(tp[index], axis=0)
    fn = np.nansum(fn[index], axis=0)
    fp = np.nansum(fp[index], axis=0)
    tn = np.nansum(tn[index], axis=0)

    tps = np.sum(tp)
    fns = np.sum(fn)
    fps = np.sum(fp)
    tns = np.sum(tn)

    epsilon = 1e-5
    names = ['Accuracy', 'Precision', 'Recall']

    df_metric = df_metric[['CIDEr-D', 'Rouge', 'Bleu-1', 'Bleu-2', 'Bleu-3', 'Bleu-4']].mean(0)
    for name in names:
        if name == 'Accuracy':
            metric = (tp + tn) / (tp + fp + tn + fn)
            micro_metric = (tps + tns) / (tps + fps + tns + fns)
        elif name == 'Precision':
            metric = tp / (tp + fp + epsilon)
            micro_metric = tps / (tps + fps + epsilon)
        elif name == 'Recall':
            metric = tp / (tp + fn + epsilon)
            micro_metric = tps / (tps + fns + epsilon)

        if name in ['Accuracy', 'Precision']:
            for (num, category) in enumerate(chexpert_labeler.CATEGORIES):
                df_metric[f'{category} ({name})'] = metric[num]

        df_metric[f'CheXpert (macro {name})'] = metric.mean()
        df_metric[f'CheXpert (micro {name})'] = micro_metric

    beta = 0.5
    df_metric[f'CheXpert (macro F{beta})'] = (1 + beta ** 2) / (1 / df_metric[f'CheXpert (macro Precision)'] + beta ** 2 / df_metric[f'CheXpert (macro Recall)'])
    df_metric[f'CheXpert (micro F{beta})'] = (1 + beta ** 2) / (1 / df_metric[f'CheXpert (micro Precision)'] + beta ** 2 / df_metric[f'CheXpert (micro Recall)'])

    print('=' * 32)
    print(df_metric)
    print('=' * 32)


if __name__ == '__main__':
    argv = FLAGS(sys.argv)

    if FLAGS.dataset == 'mimic-cxr':
        cache_dir = os.getenv('CACHE_DIR')
        df_sentence_path = os.path.join(cache_dir, f'sentence-label-field-findings.tsv')
        df_report_path = os.path.join(cache_dir, f'report-chexpert-field-findings.tsv')

    elif FLAGS.dataset == 'open-i':
        cache_dir = os.path.join(os.getenv('CACHE_DIR'), 'open-i')
        df_sentence_path = os.path.join(cache_dir, f'sentence-field-findings.tsv')
        df_report_path = os.path.join(cache_dir, f'report-chexpert-field-findings.tsv')

    locals()[FLAGS.do]()
