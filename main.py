""" TODO """
# Train all from scratch
# Pretrain w/ label then train
# Pretrained models for image embeddings
# Training time annealing
# Reinforcement Learning on CIDEr
# Testing time beam search

import datetime
import functools
import json
import logging
import os
import pandas as pd
import sys
import torch
import torch.utils.data
import tqdm
import warnings

from absl import flags
from torch.nn import (
    functional as F,
    DataParallel,
    Module,
    Embedding,
)

from mimic_cxr.data import MIMIC_CXR
from mimic_cxr.utils import Log

from api import Phase
from api.datasets import MimicCXRDataset
from api.data_loader import CollateFn
from api.models import ImageEncoder, ReportDecoder, SentenceDecoder
from api.utils import (
    SummaryWriter,
    expand_to_sequence,
    pack_padded_sequence,
    pad_packed_sequence,
    teacher_sequence,
    print_batch,
)


""" Global
"""
flags.DEFINE_string('do', 'main', 'Function to execute (default: "main")')
flags.DEFINE_string('device', 'cuda', 'GPU device to use (default: "cuda")')
flags.DEFINE_bool('debug', False, 'Turn on debug mode')

""" Image
"""
flags.DEFINE_integer('image_size', 8, 'Image feature map size (default: 8)')

""" Text
"""
flags.DEFINE_string('field', 'findings', 'The field to use in text reports (default: "findings")')
flags.DEFINE_integer('min_word_freq', 5, 'Minimum frequency of words in vocabulary (default: 5)')
flags.DEFINE_integer('max_report_length', 16, 'Maximum number of sentences in a report (default: 16)')
flags.DEFINE_integer('max_sentence_length', 64, 'Maximum number of words in a sentence (default: 64)')

""" General
"""
flags.DEFINE_integer('embedding_size', 256, 'Embedding size before feeding into the RNN (default: 256)')
flags.DEFINE_integer('hidden_size', 256, 'Hidden size in the RNN (default: 256)')
flags.DEFINE_float('dropout', 0.5, 'Dropout (default: 0.5)')

flags.DEFINE_integer('num_workers', 8, 'Number of data loading workers (default: 8)')
flags.DEFINE_integer('batch_size', 16, 'Batch size (default: 16)')
flags.DEFINE_integer('num_epochs', 16, 'Number of training epochs (default: 16)')
flags.DEFINE_float('lr', None, 'Learning rate')
flags.DEFINE_integer('log_steps', 10, 'Logging per # steps for numerical values (default: 10)')
flags.DEFINE_integer('log_text_steps', 100, 'Logging per # steps for text (default: 100)')
flags.DEFINE_string('working_dir', os.getenv('WORKING_DIR'), 'Working directory (default: $WORKING_DIR)')
FLAGS = flags.FLAGS


class Model(Module):
    def __init__(self, **kwargs):
        super(Model, self).__init__()

        self.image_encoder = ImageEncoder(**kwargs)
        self.report_decoder = ReportDecoder(**kwargs)
        self.sentence_decoder = SentenceDecoder(**kwargs)

    def forward(self, batch):
        """
        Args:
            image
            view_position
            text_length
            text
            label
            stop
            sent_length
        """

        batch.update(self.image_encoder(batch))  # image

        # Forwarding for Report Level
        batch.update(self.report_decoder(batch, length=batch['text_length']))  # _label, _topic, _stop, _temp

        # Teacher-forcing for Report Level
        for key in ['text', 'label', 'stop', 'sent_length']:
            batch[key] = teacher_sequence(batch[key])

        # Convert to Sentence Level
        for key in ['image', 'view_position']:
            batch[key] = expand_to_sequence(batch[key], torch.max(batch['text_length'] - 1))

        for key in ['image', 'view_position', 'text', 'label', 'stop', 'sent_length', '_label', '_topic', '_stop', '_temp']:
            batch[key] = pack_padded_sequence(batch[key], batch['text_length'] - 1)

        # Forwarding for Sentence Level
        # batch.update(self.sentence_decoder(batch, length=batch['sent_length']))  # _attention, _log_probability
        _batch = self.sentence_decoder.decode(batch)  # TODO: DEBUG

        # Teacher-forcing for Sentence Level
        for key in ['text']:
            batch[key] = teacher_sequence(batch[key])

        # Convert to Word Level
        for key in ['text', '_attention', '_log_probability', '_text']:
            batch[key] = pack_padded_sequence(batch[key], batch['sent_length'] - 1)

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

    Dataset = functools.partial(
        MimicCXRDataset,
        field=FLAGS.field,
        min_word_freq=FLAGS.min_word_freq,
        max_report_length=FLAGS.max_report_length,
        max_sentence_length=FLAGS.max_sentence_length,
        embedding_size=FLAGS.embedding_size,
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
        dataset=test_dataset,
        shuffle=False,
    )

    kwargs = FLAGS.flag_values_dict()
    kwargs.update({
        'word_to_index': train_dataset.word_to_index,
        'view_position_size': train_dataset.num_view_position,
        'label_size': 17,  # TODO(stmharry)
    })
    model = Model(**kwargs)
    model.sentence_decoder.word_embedding = Embedding.from_pretrained(torch.from_numpy(train_dataset.word_embedding))

    model = DataParallel(model).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=FLAGS.lr)
    logger.info(model)

    working_dir = os.path.join(FLAGS.working_dir, datetime.datetime.now().strftime('%Y-%m-%d-%H%M%S-%f'))
    os.makedirs(working_dir)
    with open(os.path.join(working_dir, 'meta.json'), 'w') as f:
        json.dump(kwargs, f, indent=4)

    writer = SummaryWriter(working_dir)

    """ Training Loop
    """

    for num_epoch in range(FLAGS.num_epochs):
        # for phase in Phase.__all__:  # TODO: DEBUG
        for phase in [Phase.train]:
            log = Log()

            if phase == Phase.train:
                data_loader = train_loader
                model.train()
            elif phase == Phase.test:
                data_loader = test_loader
                model.eval()

            prog = tqdm.tqdm(enumerate(data_loader), total=len(data_loader))
            for (num_batch, batch) in prog:
                num_step = num_epoch * len(train_loader) + num_batch

                if phase == Phase.train:
                    optimizer.zero_grad()

                for (key, value) in batch.items():
                    batch[key] = value.to(device)

                if phase == Phase.train:
                    batch = model.forward(batch)
                elif phase == Phase.test:
                    with torch.no_grad():
                        batch = model.decode(batch)

                losses = {
                    'label_ce': F.binary_cross_entropy(batch['_label'], batch['label']),
                    'stop_bce': F.binary_cross_entropy(batch['_stop'], batch['stop']),
                    'word_ce': F.nll_loss(batch['_log_probability'], batch['text']),
                }
                metrics = {
                    'word_perplexity': torch.exp(losses['word_ce']),
                }

                if phase == Phase.train:
                    total_loss = sum(losses.values())
                    total_loss.backward()
                    optimizer.step()

                _log = {**losses, **metrics}
                log.update(_log)

                prog.set_description(', '.join(
                    [f'[{phase:5s}] Epoch {num_epoch:2d}'] +
                    [f'{key:s}: {log[key]:8.2e}' for key in sorted(losses.keys())]
                ))

                if phase == Phase.train:
                    if num_step % FLAGS.log_steps == 0:
                        writer.add_log(_log, prefix=phase, global_step=num_step)

                    if num_step % FLAGS.log_text_steps == 0:
                        text_length_in_words = pad_packed_sequence(batch['sent_length'] - 1, batch['text_length'] - 1).sum(1)
                        texts = batch['text'].split(text_length_in_words.tolist())
                        _texts = batch['_text'].split(text_length_in_words.tolist())

                        writer.add_texts(texts, 'text', prefix=phase, index_to_word=train_dataset.index_to_word, global_step=num_step)
                        writer.add_texts(_texts, '_text', prefix=phase, index_to_word=train_dataset.index_to_word, global_step=num_step)

            if phase == Phase.test:
                writer.add_log(log, prefix=phase, global_step=num_step)

    writer.close()


if __name__ == '__main__':
    argv = FLAGS(sys.argv)

    logging.basicConfig(level=logging.DEBUG if FLAGS.debug else logging.INFO)
    logger = logging.getLogger(__name__)

    warnings.filterwarnings('ignore')

    logger.info('Loading MIMIC-CXR')
    mimic_cxr = MIMIC_CXR()

    if FLAGS.debug:
        FLAGS.num_workers = 0

    device = torch.device(FLAGS.device)

    locals()[FLAGS.do]()