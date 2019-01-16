""" TODO """
# Pretrain w/ label then train
# Pretrained models for image embeddings
# Reinforcement Learning on CIDEr

# scenarios
# - find best seqs to eval at test time

# baselines
# - nn in image space and return associated report: Harry
# - TieNet: generous/mean/mean + 1 std: GX
# - whole report gen: Harry
# - markov chain on clustered images

import datetime
import functools
import html
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
from htmltag import HTML, html, head, body, link, div, img
from json2html import json2html
from torch.nn import (
    functional as F,
    DataParallel,
    Embedding,
)

from mimic_cxr.data import MIMIC_CXR
from mimic_cxr.utils import Log

from api import Phase
from api.datasets import MimicCXRDataset
from api.data_loader import CollateFn
from api.models import (
    Module,
    ImageEncoder,
    ReportDecoder,
    SentenceDecoder,
)
from api.utils import (
    SummaryWriter,
    to_numpy,
    expand_to_sequence,
    pack_padded_sequence,
    pad_packed_sequence,
    print_batch,
)


""" Global
"""
flags.DEFINE_string('do', 'train', 'Function to execute (default: "train")')
flags.DEFINE_string('device', 'cuda', 'GPU device to use (default: "cuda")')
flags.DEFINE_bool('debug', False, 'Turn on debug mode (default: False)')
flags.DEFINE_bool('debug_subsample', False, 'Turn on subsampling for debugging (default: False)')

""" Image
"""
flags.DEFINE_integer('image_size', 8, 'Image feature map size (default: 8)')

""" Text
"""
flags.DEFINE_string('field', 'findings', 'The field to use in text reports (default: "findings")')
flags.DEFINE_integer('min_word_freq', 5, 'Minimum frequency of words in vocabulary (default: 5)')
flags.DEFINE_integer('max_report_length', 16, 'Maximum number of sentences in a report (default: 16)')
flags.DEFINE_integer('max_sentence_length', 64, 'Maximum number of words in a sentence (default: 64)')
flags.DEFINE_integer('beam_size', 4, 'Beam size at testing (default: 4)')
flags.DEFINE_float('alpha', 0.65, 'Power for length penaly (default: 0.65)')
flags.DEFINE_float('beta', 5.0, 'Base for length penaly (default: 5.0)')

""" General
"""
flags.DEFINE_integer('embedding_size', 256, 'Embedding size before feeding into the RNN (default: 256)')
flags.DEFINE_integer('hidden_size', 256, 'Hidden size in the RNN (default: 256)')
flags.DEFINE_float('dropout', 0.5, 'Dropout (default: 0.5)')

flags.DEFINE_integer('num_workers', 8, 'Number of data loading workers (default: 8)')
flags.DEFINE_integer('batch_size', 16, 'Batch size (default: 16)')
flags.DEFINE_integer('num_epochs', 64, 'Number of training epochs (default: 64)')
flags.DEFINE_integer('save_epochs', 1, 'Save model per # epochs (default: 1)')
flags.DEFINE_float('lr', 1e-3, 'Learning rate (default: 1e-3)')
flags.DEFINE_float('lr_decay', 0.5, 'Learning rate decay (default: 0.5)')
flags.DEFINE_integer('lr_decay_epochs', 8, 'Learning rate decay per # epochs (default: 8)')
flags.DEFINE_float('tf_decay', 0.05, 'Teacher forcing ratio decay (default: 0.05)')
flags.DEFINE_integer('tf_decay_epochs', 4, 'Teacher forcing ratio decay (default: 4)')
flags.DEFINE_integer('log_steps', 10, 'Logging per # steps for numerical values (default: 10)')
flags.DEFINE_integer('log_text_steps', 100, 'Logging per # steps for text (default: 100)')
flags.DEFINE_string('ckpt_path', None, 'Checkpoint path to load (default: None)')
flags.DEFINE_string('working_dir', os.getenv('WORKING_DIR'), 'Working directory (default: $WORKING_DIR)')
FLAGS = flags.FLAGS


class Model(Module):
    def __init__(self, **kwargs):
        super(Model, self).__init__()

        self.image_encoder = ImageEncoder(**kwargs)
        self.report_decoder = ReportDecoder(**kwargs)
        self.sentence_decoder = SentenceDecoder(**kwargs)

    def _train(self, batch):
        batch.update(self.image_encoder(batch))
        batch.update(self.report_decoder._train(batch, length=batch['text_length'], teacher_forcing_ratio=FLAGS.teacher_forcing_ratio))

        for key in ['image', 'view_position']:
            batch[key] = expand_to_sequence(batch[key], torch.max(batch['text_length']))

        for key in ['image', 'view_position', 'text', 'label', 'stop', 'sent_length', '_label', '_topic', '_stop', '_temp']:
            batch[key] = pack_padded_sequence(batch[key], batch['text_length'])

        batch.update(self.sentence_decoder._train(batch, length=batch['sent_length'], teacher_forcing_ratio=FLAGS.teacher_forcing_ratio))

        return batch

    def _test(self, batch):
        batch.update(self.image_encoder(batch))
        batch.update(self.report_decoder._test(batch))

        for key in ['image', 'view_position']:
            batch[key] = expand_to_sequence(batch[key], torch.max(batch['_text_length']))

        for key in ['image', 'view_position', '_label', '_topic', '_stop', '_temp']:
            batch[key] = pack_padded_sequence(batch[key], batch['_text_length'])

        batch.update(self.sentence_decoder._test(batch, beam_size=FLAGS.beam_size, alpha=FLAGS.alpha, beta=FLAGS.beta))

        return batch


def train():
    optimizer = torch.optim.Adam(model.parameters(), lr=FLAGS.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, FLAGS.lr_decay_epochs, FLAGS.lr_decay)

    with open(os.path.join(working_dir, 'meta.json'), 'w') as f:
        json.dump(kwargs, f, indent=4)
    torch.save(model, os.path.join(working_dir, f'model-init.pth'))
    writer = SummaryWriter(working_dir)
    logger.info(f'Writing to {working_dir}')

    """ Training Loop
    """

    for num_epoch in range(FLAGS.num_epochs):
        scheduler.step()
        teacher_forcing_ratio = max(0, 1 - (num_epoch // FLAGS.tf_decay_epochs) * FLAGS.tf_decay)

        for phase in phases:
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
                    batch = model(batch, phase=phase)
                    text = pack_padded_sequence(batch['text'], batch['sent_length'])
                    _text = pack_padded_sequence(batch['_text'], batch['sent_length'])

                    losses = {
                        'stop_bce': F.binary_cross_entropy(batch['_stop'], batch['stop']),
                        'word_ce': F.nll_loss(batch['_log_probability'], batch['text']),
                    }
                elif phase == Phase.test:
                    with torch.no_grad():
                        batch = model(batch, phase=phase)

                    losses = {}

                metrics = {}  # TODO(stmharry)

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
                        writer.add_scalar(f'{phase}/learning_rate', optimizer.param_groups[0]['lr'], global_step=num_step)
                        writer.add_scalar(f'{phase}/teacher_forcing_ratio', teacher_forcing_ratio, global_step=num_step)
                        writer.add_log(_log, prefix=phase, global_step=num_step)

                    if num_step % FLAGS.log_text_steps == 0:
                        text_length_in_words = pad_packed_sequence(batch['sent_length'], batch['text_length']).sum(1)

                        texts = batch['text'].split(text_length_in_words.tolist())
                        _texts = batch['_text'].split(text_length_in_words.tolist())

                        writer.add_texts(texts, 'text', prefix=phase, index_to_word=train_dataset.index_to_word, global_step=num_step)
                        writer.add_texts(_texts, '_text', prefix=phase, index_to_word=train_dataset.index_to_word, global_step=num_step)

            if phase == Phase.test:
                writer.add_log(log, prefix=phase, global_step=num_step)

        if num_epoch % FLAGS.save_epochs == 0:
            torch.save(model.state_dict(), os.path.join(working_dir, f'model-epoch-{num_epoch}.pth'))

    writer.close()


def test():
    data_loader = test_loader
    dataset = test_dataset
    model.eval()

    phase = Phase.test
    log = Log()

    prog = tqdm.tqdm(enumerate(data_loader), total=len(data_loader))
    for (num_batch, batch) in prog:
        for (key, value) in batch.items():
            batch[key] = value.to(device)

        with torch.no_grad():
            batch = model(batch, phase=phase)

        losses = {}
        metrics = {}

        _log = {**losses, **metrics}
        log.update(_log)

        prog.set_description(', '.join(
            [f'[{phase:5s}]'] +
            [f'{key:s}: {log[key]:8.2e}' for key in sorted(losses.keys())]
        ))

        item_index = batch['item_index']
        _text_length = batch['_text_length']  # (num_reports,)
        (
            _stop,             # (num_reports, max_num_sentences, 1)
            _temp,             # (num_reports, max_num_sentences, 1)
            _score,            # (num_reports, max_num_sentences, beam_size)
            _log_probability,  # (num_reports, max_num_sentences, beam_size)
            _text,             # (num_reports, max_num_sentences, beam_size, max_num_words)
            _sent_length,      # (num_reports, max_num_sentences, beam_size)
        ) = [
            pad_packed_sequence(batch[key], _text_length)
            for key in ['_stop', '_temp', '_score', '_log_probability', '_text', '_sent_length']
        ]

        reports = []
        for num_report in range(len(_text_length)):
            item = dataset.df.iloc[int(item_index[num_report])]
            image_path = mimic_cxr.image_path(dicom_id=item.dicom_id)

            sentences = []
            for num_sentence in range(_text_length[num_report]):
                beams = []
                for num_beam in range(FLAGS.beam_size):
                    num_words = _sent_length[num_report, num_sentence, num_beam]
                    beam_texts = [
                        train_dataset.index_to_word[_text[num_report, num_sentence, num_beam, num_word]]
                        for num_word in range(num_words)
                    ]

                    beams.append({
                        'score': '{:.2f}'.format(float(_score[num_report, num_sentence, num_beam])),
                        'log_prob': '{:.2f}'.format(float(_log_probability[num_report, num_sentence, num_beam])),
                        'text': str(html.escape(' '.join(beam_texts))),
                    })

                sentences.append({
                    'stop': '{:.4f}'.format(float(_stop[num_report, num_sentence])),
                    'temp': '{:.2f}'.format(float(_temp[num_report, num_sentence])),
                    'beams': beams,
                })

            reports.append({
                'image': str(img(src=f'http://monday.csail.mit.edu/xiuming{image_path}', width='256')),
                'text': sentences,
            })

        if num_batch == 4:
            break

    s = json2html.convert(json=reports, table_attributes='class="table table-striped"', escape=False)
    s = head(link(rel='stylesheet', href='https://maxcdn.bootstrapcdn.com/bootstrap/3.3.7/css/bootstrap.min.css')) + \
        body(div(div(div(HTML(s), _class='panel-body'), _class='panel panel-default'), _class='container'))

    with open(os.path.join(working_dir, 'index.html'), 'w') as f:
        f.write(s)

if __name__ == '__main__':
    argv = FLAGS(sys.argv)

    warnings.filterwarnings('ignore')

    logging.basicConfig(level=logging.DEBUG if FLAGS.debug else logging.INFO)
    logger = logging.getLogger(__name__)

    device = torch.device(FLAGS.device)
    if FLAGS.debug:
        FLAGS.num_workers = 0

    logger.info('Loading MIMIC-CXR')
    mimic_cxr = MIMIC_CXR()

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

    if FLAGS.debug_subsample:
        train_df = train_df.iloc[:int(0.001 * len(train_df))]

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
        'index_to_word': train_dataset.index_to_word,
        'word_to_index': train_dataset.word_to_index,
        'view_position_size': train_dataset.num_view_position,
        'label_size': 16,  # TODO(stmharry)
    })

    model = Model(**kwargs)
    model.sentence_decoder.word_embedding = Embedding.from_pretrained(torch.from_numpy(train_dataset.word_embedding), freeze=False)
    model = DataParallel(model).to(device)

    if FLAGS.ckpt_path:
        logger.info(f'Loading model from {FLAGS.ckpt_path}')
        model.load_state_dict(torch.load(FLAGS.ckpt_path))
    logger.info(f'Model info:\n{model}')

    if FLAGS.debug:
        working_dir = os.path.join(FLAGS.working_dir, 'debug')
    else:
        working_dir = FLAGS.working_dir

    working_dir = os.path.join(working_dir, datetime.datetime.now().strftime('%Y-%m-%d-%H%M%S-%f'))
    os.makedirs(working_dir)

    locals()[FLAGS.do]()
