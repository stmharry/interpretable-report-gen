import pandas as pd
import torch
import tqdm

from api.metrics import (
    Bleu,
    Rouge,
    CiderD as Cider,
)

df_cache = torch.load('/data/medg/misc/interpretable-report-gen/cache/cider-cache.pkl')
scorers = [Bleu(4), Rouge(), Cider(df_cache=df_cache)]

df_gen = pd.read_csv('/data/medg/misc/liuguanx/gen-report-15.tsv', sep='\t', dtype=str)
df_gt = pd.read_csv('/data/medg/misc/interpretable-report-gen/cache/sentence-label-field-findings.tsv', sep='\t', dtype=str)

df_gen = df_gen[~df_gen.text.isnull().values]
df_gen = df_gen.set_index('rad_id').text.str.slice(8, -6).to_frame()
df_gt = df_gt.groupby('rad_id').sentence.apply(' '.join).to_frame()

df = pd.merge(df_gt, df_gen, on='rad_id')

metrics = []
for item in tqdm.tqdm(df.itertuples(), total=len(df)):
    metric = {}
    for scorer in scorers:
        s = scorer([item.text], [item.sentence])

        if s.dim() == 2:
            for (num, _s) in enumerate(s):
                metric[f'{scorer.method()}-{num + 1}'] = _s.mean().item()
        else:
            metric[f'{scorer.method()}'] = s.mean().item()

    metrics.append(metric)

metrics = {
    key: torch.mean(torch.FloatTensor([metric[key] for metric in metrics]))
    for key in metrics[0].keys()
}

print(metrics)
