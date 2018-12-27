import datetime
import functools
import logging
import nltk
import numpy as np
import os
import pandas as pd
import sys
import torch
import torch.utils.data
import tqdm
import warnings

from absl import flags
from torch.nn import Module, functional as F

from mimic_cxr.data import MIMIC_CXR
from mimic_cxr.utils import (
    Log,
    SummaryWriter,
)

from api import Phase
from api.datasets import Dataset as _Dataset
from api.data_loader import CollateFn
from api.models import ImageEncoder, ReportDecoder, SentenceDecoder
from api.utils import unpad_sequence, teacher_sequence


### Global
flags.DEFINE_string('do', 'main', 'Function to execute (default: "main")')
flags.DEFINE_string('device', 'cuda', 'GPU device to use (default: "cuda")')
flags.DEFINE_bool('debug', False, 'Turn on debug mode')

### Image
flags.DEFINE_integer('image_size', 8, 'Image feature map size (default: 8)')

### Text
flags.DEFINE_string('field', 'findings', 'The field to use in text reports (default: "findings")')
flags.DEFINE_integer('max_report_length', 16, 'Maximum number of sentences in a report (default: 16)')
flags.DEFINE_integer('max_sentence_length', 64, 'Maximum number of words in a sentence (default: 64)')

### General
flags.DEFINE_integer('embedding_size', 256, 'Embedding size before feeding into the RNN (default: 256)')
flags.DEFINE_integer('hidden_size', 256, 'Hidden size in the RNN (default: 256)')
flags.DEFINE_integer('num_workers', 8, 'Number of data loading workers (default: 8)')
flags.DEFINE_integer('batch_size', 4, 'Batch size (default: 4)')
flags.DEFINE_integer('num_epochs', 16, 'Number of training epochs (default: 16)')
flags.DEFINE_float('lr', None, 'Learning rate')
flags.DEFINE_string('working_dir', os.getenv('WORKING_DIR'), 'Working directory (default: $WORKING_DIR)')
FLAGS = flags.FLAGS


class Model(Module):
    def __init__(self, **kwargs):
        super(Model, self).__init__()

        self.image_encoder = ImageEncoder(**kwargs)
        self.report_decoder = ReportDecoder(**kwargs)
        self.sentence_decoder = SentenceDecoder(**kwargs)

        self.optimizer = kwargs['optim_cls'](
            self.parameters(),
            lr=FLAGS.lr,
        )

    def forward(self, batch):
        (image, view_position, text_length, text, label, stop, sent_length) = [
            batch[key]
            for key in ['image', 'view_position', 'text_length', 'text', 'label', 'stop', 'sent_length']
        ]

        losses = {}
        metrics = {}

        image = self.image_encoder(image)

        ### Forwarding for Report Level
        (_label, _topic, _stop, _temp) = self.report_decoder(image, view_position, label, length=text_length)

        ### Teacher-forcing for Report Level
        (text, label, stop, sent_length) = [
            teacher_sequence(value)
            for value in [text, label, stop, sent_length]
        ]

        ### Convert to Sentence Level
        (image, view_position) = [
            torch.cat(unpad_sequence(value, text_length - 1, is_sequence=False), 0)
            for value in [image, view_position]
        ]
        (text, label, stop, sent_length, _label, _topic, _stop, _temp) = [
            torch.cat(unpad_sequence(value, text_length - 1, is_sequence=True), 0)
            for value in [text, label, stop, sent_length, _label, _topic, _stop, _temp]
        ]

        losses['end_ce'] = F.binary_cross_entropy(_stop, stop)

        ### Forwarding for Sentence Level
        (_attention, _log_probability) = self.sentence_decoder(image, view_position, text, label, _topic, _temp, length=sent_length)

        ### Teacher-forcing for Sentence Level
        text = teacher_sequence(text)

        ### Convert to Word Level
        (text, _log_probability) = [
            torch.cat(unpad_sequence(value, sent_length - 1, is_sequence=True), 0)
            for value in [text, _log_probability]
        ]

        losses['word_ce'] = F.nll_loss(_log_probability, text)

        import pdb; pdb.set_trace()

        return {
            'topic': _topic,
            'temp': _temp,
            'attention': _attention,
            'log_probability': _log_probability,
            'losses': losses,
            'metrics': metrics,
        }

    def decode(self):
        pass

    def pre_epoch(self, phase):
        self.phase = phase

    def pre_batch(self):
        self.num_step = num_step

        if self.phase == Phase.train:
            self.train()
            self.optimizer.zero_grad()

    def step(self, batch):
        batch = self.forward(batch)

        if self.phase == Phase.train:
            total_loss = sum(batch['losses'].values())
            total_loss.backward()
            self.optimizer.step()

        return batch


def main():
    logger.info('Loading meta')
    df_meta = pd.read_csv(mimic_cxr.meta_path())

    logger.info('Loading text features')
    df_rad = pd.read_csv(mimic_cxr.corpus_path(field=FLAGS.field), sep='\t', dtype=str)

    logger.info('Loading image features')
    df_dicom = pd.Series(os.listdir(mimic_cxr.image_path())).str.split('.', expand=True)[0].rename('dicom_id').to_frame()

    on_dfs = [
        ('rad_id', df_rad),
        ('dicom_id', df_meta),
        ('dicom_id', df_dicom),
    ]
    df = mimic_cxr.inner_merge(on_dfs)
    logger.info(f'Dataset size={len(df)}')

    train_df = df[df.subject_id.isin(mimic_cxr.train_subject_ids)]
    test_df = df[df.subject_id.isin(mimic_cxr.test_subject_ids)]

    df_word = pd.read_csv(os.path.join(os.getenv('CACHE_DIR'), 'word_map.csv'))

    Dataset = functools.partial(
        _Dataset,
        word_to_index=dict(zip(df_word.word, df_word.word_index)),
        max_report_length=FLAGS.max_report_length,
        max_sentence_length=FLAGS.max_sentence_length,
    )
    train_dataset = Dataset(
        df=train_df,
        is_train=True,
    )
    test_dataset = Dataset(
        df=test_df,
        is_train=False,
    )

    DataLoader = functools.partial(
        torch.utils.data.DataLoader,
        batch_size=FLAGS.batch_size,
        num_workers=FLAGS.num_workers,
        collate_fn=CollateFn(sequence_fields=['text', 'label', 'stop', 'sent_length']),
    )
    train_loader = DataLoader(
        dataset=train_dataset,
        shuffle=True,
    )
    test_loader = DataLoader(
        dataset = test_dataset,
        shuffle=False,
    )

    kwargs = FLAGS.flag_values_dict()
    kwargs.update({
        'image_embedding_size': ImageEncoder.image_embedding_size,
        'view_position_size': train_dataset.num_view_position,
        'vocab_size': len(train_dataset.word_to_index),
        'label_size': 16,  # TODO(stmharry)
        'optim_cls': torch.optim.Adam,
    })
    model = Model(**kwargs).to(device)
    logger.info(model)

    working_dir = os.path.join(FLAGS.working_dir, '{:s}-{:s}'.format(
        datetime.datetime.now().strftime('%Y-%m-%d-%H%M%S-%f'),
        model.get_name(),
    ))
    os.makedirs(working_dir)
    with open(os.path.join(working_dir, 'meta.json'), 'w') as f:
        json.dump(kwargs, f, indent=4)

    writer = SummaryWriter(working_dir)

    ### Training Loop

    for num_epoch in range(FLAGS.num_epochs):
        for phase in Phase.__all__:
            log = Log()

            data_loader = {
                Phase.train: train_loader,
                Phase.test: test_loader,
            }[phase]

            model.pre_epoch(phase=phase)

            prog = tqdm.tqdm(enumerate(data_loader), total=len(data_loader))
            for (num_batch, batch) in prog:
                num_step = num_epoch * len(train_loader) + num_batch
                model.pre_batch(num_step=num_step)

                for (key, value) in batch.items():
                    batch[key] = value.to(device=device)

                batch = model.step(batch)

                # TODO(stmharry)
                log.update({
                    **batch['losses'],
                })

                summary_keys = batch['losses'].keys()
                prog.set_description(', '.join(
                    [f'[{phase:5s}] Epoch {num_epoch:2d}'] +
                    [f'{key:s}: {log[key]:8.2e}' for key in sorted(summary_keys)]
                ))

                if phase == Phase.train:
                    writer.add_log(log, prefix=phase, global_step=num_step)

            if phase == Phase.test:
                writer.add_log(log, prefix=phase, global_step=num_step)

    writer.close()


def make_wordmap(field, min_freq):
    import tqdm

    from nltk.tokenize.punkt import PunktSentenceTokenizer
    from nltk.tokenize.casual import TweetTokenizer

    df = pd.read_csv(mimic_cxr.corpus_path(field=field), sep='\t', dtype=str)
    df = mimic_cxr.inner_merge([('rad_id', df)])
    df = df[df.subject_id.isin(mimic_cxr.train_subject_ids)]

    sent_tokenizer = PunktSentenceTokenizer()
    word_tokenizer = TweetTokenizer()

    counter = {}
    for item in tqdm.tqdm(df.itertuples(), total=len(df)):
        for sentence in sent_tokenizer.tokenize(item.text):
            for word in word_tokenizer.tokenize(sentence):
                counter[word] = counter.get(word, 0) + 1

    df = pd.DataFrame(list(counter.items()), columns=['word', 'word_count']).set_index('word').sort_values('word_count', ascending=False)
    df['word_index'] = range(1, len(counter) + 1)

    df.to_csv(os.path.join(os.getenv('CACHE_DIR'), 'word_map.csv'))


if __name__ == '__main__':
    argv = FLAGS(sys.argv)

    logging.basicConfig(level=logging.DEBUG if FLAGS.debug else logging.INFO)
    logger = logging.getLogger(__name__)

    warnings.filterwarnings('ignore')

    logger.info('Loading MIMIC-CXR')
    mimic_cxr = MIMIC_CXR()

    make_wordmap = functools.partial(make_wordmap, field=FLAGS.field, min_freq=1)

    if FLAGS.debug:
        FLAGS.num_workers = 0

    device = torch.device(FLAGS.device)

    locals()[FLAGS.do]()
