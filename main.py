""" TODO """
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
import sklearn.metrics
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

from api import Mode, Phase
from api.datasets import MimicCXRDataset
from api.data_loader import CollateFn
from api.models import Model
from api.utils import (
    SummaryWriter,
    to_numpy,
    pack_padded_sequence,
    pad_packed_sequence,
    pad_sequence,
    print_batch,
    version_of,
    profile,
)
from api.metrics import (
    Bleu,
    Rouge,
    CiderD as Cider,
)

""" Global
"""
flags.DEFINE_string('do', 'train', 'Function to execute (default: "train")')
flags.DEFINE_enum('test_split', 'test', ['val', 'test'], 'Split to test for')
flags.DEFINE_enum('mode', None, Mode.__all__, 'Training mode (' + ' or '.join(map('"{}"'.format, Mode.__all__)) + ')')
flags.DEFINE_string('device', 'cuda', 'GPU device to use (default: "cuda")')
flags.DEFINE_bool('debug', False, 'Turn on debug mode (default: False)')
flags.DEFINE_bool('debug_subsample', False, 'Turn on subsampling for debugging (default: False)')
flags.DEFINE_bool('profile', False, 'Turn on profiling mode (default: False)')

""" Image
"""
flags.DEFINE_integer('image_size', 8, 'Image feature map size (default: 8)')

""" Text
"""
flags.DEFINE_string('field', 'findings', 'The field to use in text reports (default: "findings")')
flags.DEFINE_integer('min_word_freq', 5, 'Minimum frequency of words in vocabulary (default: 5)')
flags.DEFINE_integer('max_report_length', 16, 'Maximum number of sentences in a report (default: 16)')
flags.DEFINE_integer('max_sentence_length', 48, 'Maximum number of words in a sentence (default: 48)')
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
flags.DEFINE_float('tf_decay', 0.0, 'Teacher forcing ratio decay (default: 0.0)')
flags.DEFINE_integer('tf_decay_epochs', 8, 'Teacher forcing ratio decay (default: 8)')
flags.DEFINE_integer('log_steps', 10, 'Logging per # steps for numerical values (default: 10)')
flags.DEFINE_integer('log_text_steps', 100, 'Logging per # steps for text (default: 100)')
flags.DEFINE_string('ckpt_path', None, 'Checkpoint path to load (default: None)')
flags.DEFINE_string('working_dir', os.getenv('WORKING_DIR'), 'Working directory (default: $WORKING_DIR)')
FLAGS = flags.FLAGS


@profile
def train():
    optimizer = torch.optim.Adam(model.parameters(), lr=FLAGS.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, FLAGS.lr_decay_epochs, FLAGS.lr_decay)

    scorers = [Bleu(4), Rouge(), Cider(df_cache=dataset.df_cache)]
    cider = scorers[2]

    if FLAGS.do == 'train':
        os.makedirs(working_dir)
        with open(os.path.join(working_dir, 'meta.json'), 'w') as f:
            json.dump(FLAGS.flag_values_dict(), f, indent=4)

        torch.save(model, os.path.join(working_dir, f'model-init.pth'))
        writer = SummaryWriter(working_dir)
        logger.info(f'Writing to {working_dir}')

        num_epochs = FLAGS.num_epochs
        phases = [Phase.train, Phase.val]

    elif FLAGS.do == 'val':
        num_epochs = 1
        phases = [Phase.val]

    if FLAGS.profile:
        num_epochs = 1
        phases = [Phase.train]

    """ Training Loop
    """

    for num_epoch in range(num_epochs):
        scheduler.step()

        if FLAGS.mode == Mode.teacher_forcing:
            teacher_forcing_ratio = max(0, 1 - (num_epoch // FLAGS.tf_decay_epochs) * FLAGS.tf_decay)

        elif FLAGS.mode == Mode.self_critical:
            teacher_forcing_ratio = 0.0

        for phase in phases:
            log = Log()
            log_items = []

            if phase == Phase.train:
                data_loader = train_loader

            elif phase == Phase.val:
                data_loader = val_loader

            prog = tqdm.tqdm(enumerate(data_loader), total=len(data_loader))
            for (num_batch, batch) in prog:
                num_step = num_epoch * len(train_loader) + num_batch

                if phase == Phase.train:
                    optimizer.zero_grad()

                for (key, value) in batch.items():
                    batch[key] = value.to(device)

                losses = {}
                metrics = {}

                if phase == Phase.train:
                    if FLAGS.mode == Mode.debug_label:
                        output = model(batch, phase=Phase.train)

                    if FLAGS.mode in [Mode.auto_regress, Mode.teacher_forcing]:
                        output = model(batch, phase=Phase.train, teacher_forcing_ratio=teacher_forcing_ratio)

                    if FLAGS.mode == Mode.self_critical:
                        output = model(batch, phase=Phase.train, beam_size=1)

                    ###

                    if FLAGS.mode == Mode.debug_label:
                        losses['label_ce'] = F.binary_cross_entropy(output['_label'], output['label'].sum(1).clamp(0, 1))

                    if FLAGS.mode in [Mode.auto_regress, Mode.teacher_forcing]:
                        losses['label_ce'] = F.binary_cross_entropy(output['_label'], output['label'])
                        losses['stop_bce'] = F.binary_cross_entropy(output['_stop'], output['stop'])

                    if FLAGS.mode == Mode.teacher_forcing:
                        reports = train_dataset.convert_sentence(output['text'], output['sent_length'], output['text_length'])
                        _reports = train_dataset.convert_sentence(output['_text'], output['sent_length'], output['text_length'])

                        word = pack_padded_sequence(output['text'], length=output['sent_length'])
                        _log_probability = pack_padded_sequence(output['_log_probability'], length=output['sent_length'])

                        losses['word_ce'] = F.nll_loss(_log_probability, word)

                    if FLAGS.mode == Mode.self_critical:
                        output = model(batch, phase=Phase.train, beam_size=1)

                        reports = train_dataset.convert_sentence(output['text'], output['sent_length'], output['text_length'])
                        _reports = train_dataset.convert_sentence(output['_text'][:, 0], output['_sent_length'][:, 0], output['_text_length'])
                        _sum_log_probability = pad_packed_sequence(output['_sum_log_probability'][:, 0], length=output['_text_length']).sum(1)

                        with torch.no_grad():
                            output_greedy = model(batch, phase=Phase.test, beam_size=1)
                        _reports_greedy = train_dataset.convert_sentence(output_greedy['_text'][:, 0], output_greedy['_sent_length'][:, 0], output_greedy['_text_length'])

                        metric = cider(_reports, reports)
                        metric_greedy = cider(_reports_greedy, reports)
                        reward = metric - metric_greedy

                        losses['report_sc'] = - (_sum_log_probability * reward).mean()
                        metrics['cider'] = metric.mean()
                        metrics['reward'] = reward.mean()

                elif phase == Phase.val:
                    with torch.no_grad():
                        if FLAGS.mode in [Mode.debug_label, Mode.auto_regress]:
                            output = model(batch, phase=Phase.val)

                        if FLAGS.mode in [Mode.teacher_forcing, Mode.self_critical]:
                            output = model(batch, phase=Phase.val, beam_size=FLAGS.beam_size, alpha=FLAGS.alpha, beta=FLAGS.beta)

                    ###

                    if FLAGS.mode == Mode.debug_label:
                        label_all = output['label'].sum(1).clamp(0, 1)
                        _label_all = output['_label']

                        losses['label_ce_all'] = F.binary_cross_entropy(_label_all, label_all)

                        log_items.append({
                            'label_all': label_all,
                            '_label_all': _label_all,
                        })

                    if FLAGS.mode in [Mode.auto_regress, Mode.teacher_forcing]:
                        label = pad_packed_sequence(output['label'], length=output['text_length'])
                        _label = pad_packed_sequence(output['_label'], length=output['_text_length'])

                        label_first = label[:, 0]
                        _label_first = _label[:, 0]
                        label_all = label.sum(1).clamp(0, 1)
                        _label_all = _label.sum(1).clamp(0, 1)

                        metrics['label_ce_first'] = F.binary_cross_entropy(_label_first, label_first)
                        metrics['label_ce_all'] = F.binary_cross_entropy(_label_all, label_all)

                        log_items.append({
                            'label_all': label_all,
                            '_label_all': _label_all,
                        })

                    if FLAGS.mode in [Mode.teacher_forcing, Mode.self_critical]:
                        reports = train_dataset.convert_sentence(output['text'], output['sent_length'], output['text_length'])
                        _reports = train_dataset.convert_sentence(output['_text'][:, 0], output['_sent_length'][:, 0], output['_text_length'])

                        for scorer in scorers:
                            metric = scorer(_reports, reports)

                            if metric.dim() == 2:
                                for (num, _metric) in enumerate(metric):
                                    metrics[f'{scorer.method()}-{num + 1}'] = _metric.mean().item()
                            else:
                                metrics[f'{scorer.method()}'] = metric.mean().item()

                if phase == Phase.train:
                    total_loss = sum(losses.values())
                    total_loss.backward()
                    optimizer.step()

                _log = {**losses, **metrics}
                log.update(_log)

                prog.set_description(', '.join(
                    [f'[{phase:5s}] Epoch {num_epoch:2d}'] +
                    [f'{key:s}: {log[key]:8.2e}' for key in losses.keys()]
                ))

                if phase == Phase.train:
                    if num_step % FLAGS.log_steps == 0:
                        writer.add_scalar(f'{phase}/learning_rate', optimizer.param_groups[0]['lr'], global_step=num_step)
                        writer.add_scalar(f'{phase}/teacher_forcing_ratio', teacher_forcing_ratio, global_step=num_step)
                        writer.add_log(_log, prefix=phase, global_step=num_step)

                    if (num_step % FLAGS.log_text_steps == 0) and (FLAGS.mode in [Mode.teacher_forcing, Mode.self_critical]):
                        writer.add_texts(reports, 'text', prefix=phase, global_step=num_step)
                        writer.add_texts(_reports, '_text', prefix=phase, global_step=num_step)

                if FLAGS.profile and (num_batch == 32):
                    break

            if phase == Phase.train:
                if num_epoch % FLAGS.save_epochs == 0:
                    torch.save(model.state_dict(), os.path.join(working_dir, f'model-epoch-{num_epoch}.pth'))

            elif phase == Phase.val:
                if FLAGS.mode in [Mode.debug_label, Mode.auto_regress, Mode.teacher_forcing]:
                    (label_all, _label_all) = [
                        torch.cat([log_item[key] for log_item in log_items], 0)
                        for key in ['label_all', '_label_all']
                    ]

                    log.update({
                        'AUCROC__macro_': sklearn.metrics.roc_auc_score(to_numpy(label_all), to_numpy(_label_all), average='macro'),
                        'AUCROC__micro_': sklearn.metrics.roc_auc_score(to_numpy(label_all), to_numpy(_label_all), average='micro'),
                        'AP__macro_': sklearn.metrics.average_precision_score(to_numpy(label_all), to_numpy(_label_all), average='macro'),
                        'AP__micro_': sklearn.metrics.average_precision_score(to_numpy(label_all), to_numpy(_label_all), average='micro'),
                    })

                if FLAGS.do == 'train':
                    writer.add_log(log, prefix=phase, global_step=num_step)

    if FLAGS.do == 'train':
        writer.close()

    elif FLAGS.do == 'val':
        logger.info('\n'.join(
            [f'{key:s}: {log[key]:8.2e}' for key in log.keys()]
        ))


val = train


def test():
    def to_str(x):
        return '{:.2f}'.format(x.item())

    phase = FLAGS.test_split
    (data_loader, dataset) = {
        Phase.val: (val_loader, val_dataset),
        Phase.test: (test_loader, test_dataset),
    }[FLAGS.test_split]

    log = Log()

    prog = tqdm.tqdm(enumerate(data_loader), total=len(data_loader))
    for (num_batch, batch) in prog:
        for (key, value) in batch.items():
            batch[key] = value.to(device)

        losses = {}
        metrics = {}

        with torch.no_grad():
            batch = model(batch, phase=phase, beam_size=FLAGS.beam_size, alpha=FLAGS.alpha, beta=FLAGS.beta)

        _log = {**losses, **metrics}
        log.update(_log)

        prog.set_description(', '.join(
            [f'[{phase:5s}]'] +
            [f'{key:s}: {log[key]:8.2e}' for key in sorted(losses.keys())]
        ))

        if FLAGS.test_split == 'val':
            report_texts = train_dataset.convert_sentence(batch['text'], batch['sent_length'], batch['text_length'])

        item_index = batch['item_index']
        _text_length = batch['_text_length']  # (num_reports,)
        (
            _label,                # (num_reports, max_num_sentences, label_size)
            _stop,                 # (num_reports, max_num_sentences, 1)
            _temp,                 # (num_reports, max_num_sentences, 1)
            _score,                # (num_reports, max_num_sentences, beam_size)
            _sum_log_probability,  # (num_reports, max_num_sentences, beam_size)
            _text,                 # (num_reports, max_num_sentences, beam_size, max_num_words)
            _sent_length,          # (num_reports, max_num_sentences, beam_size)
        ) = [
            pad_packed_sequence(batch[key], length=_text_length)
            for key in ['_label', '_stop', '_temp', '_score', '_sum_log_probability', '_text', '_sent_length']
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
                    _sentence = _text[num_report, num_sentence, num_beam]

                    beam_text = ' '.join(train_dataset.index_to_word[to_numpy(_sentence[:num_words])])

                    beams.append({
                        'score': to_str(_score[num_report, num_sentence, num_beam]),
                        'log_prob': to_str(_sum_log_probability[num_report, num_sentence, num_beam]),
                        'text': str(html.escape(beam_text)),
                    })

                sentences.append({
                    'stop': to_str(_stop[num_report, num_sentence]),
                    'temp': to_str(_temp[num_report, num_sentence]),
                    'labels': [{
                        'label': label_col,
                        'prob': to_str(_l),
                    } for (label_col, _l) in zip(train_dataset.label_columns, _label[num_report, num_sentence])],
                    'beams': beams,
                })

            report = {
                'image': str(img(src=f'http://monday.csail.mit.edu/xiuming{image_path}', width='256')),
                'generated text': sentences,
            }
            if FLAGS.test_split == 'val':
                report['ground truth text'] = report_texts[num_report]

            reports.append(report)

        if num_batch == 4:
            break

    s = json2html.convert(json=reports, table_attributes='class="table table-striped"', escape=False)
    s = head(link(rel='stylesheet', href='https://maxcdn.bootstrapcdn.com/bootstrap/3.3.7/css/bootstrap.min.css')) + \
        body(div(div(div(HTML(s), _class='panel-body'), _class='panel panel-default'), _class='container-fluid'))

    os.makedirs(working_dir)
    with open(os.path.join(working_dir, 'index.html'), 'w') as f:
        f.write(s)


if __name__ == '__main__':
    argv = FLAGS(sys.argv)

    kwargs = FLAGS.flag_values_dict()
    kwargs.update({
        '__use_continuous_label': version_of(FLAGS.ckpt_path) > 1548152415,
        '__no_recurrent_label': version_of(FLAGS.ckpt_path) > 1548347568,
        '__image_encoder_relu': version_of(FLAGS.ckpt_path) > 1548428102,
        '__sample_text': version_of(FLAGS.ckpt_path) > 1548708003,
        '__use_densenet': version_of(FLAGS.ckpt_path) > 1548881554,
        '__no_temp': version_of(FLAGS.ckpt_path) > 1548881554,
    })

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

    train_size = int(0.875 * len(mimic_cxr.train_subject_ids))
    train_df = df[df.subject_id.isin(mimic_cxr.train_subject_ids[:train_size])]
    val_df = df[df.subject_id.isin(mimic_cxr.train_subject_ids[train_size:])]
    test_df = df[df.subject_id.isin(mimic_cxr.test_subject_ids)]

    if FLAGS.debug_subsample:
        train_df = train_df.sample(n=128)
        val_df = val_df.sample(n=128)
        test_df = test_df.sample(n=128)

    Dataset = functools.partial(
        MimicCXRDataset,
        field=FLAGS.field,
        min_word_freq=FLAGS.min_word_freq,
        max_report_length=FLAGS.max_report_length,
        max_sentence_length=FLAGS.max_sentence_length,
        embedding_size=FLAGS.embedding_size,
        kwargs=kwargs,
    )
    dataset = Dataset(df=df, phase=None)

    train_dataset = Dataset(df=train_df, phase=Phase.train)
    val_dataset   = Dataset(df=val_df,   phase=Phase.val)
    test_dataset  = Dataset(df=test_df,  phase=Phase.test)

    DataLoader = functools.partial(
        torch.utils.data.DataLoader,
        batch_size=FLAGS.batch_size,
        num_workers=FLAGS.num_workers,
        collate_fn=CollateFn(sequence_fields=['text', 'label', 'stop', 'sent_length']),
        pin_memory=True,
    )
    train_loader = DataLoader(dataset=train_dataset, shuffle=True)
    val_loader   = DataLoader(dataset=val_dataset,   shuffle=True)
    test_loader  = DataLoader(dataset=test_dataset,  shuffle=False)

    kwargs.update({
        'word_embedding': train_dataset.word_embedding,
        'index_to_word': train_dataset.index_to_word,
        'word_to_index': train_dataset.word_to_index,
        'view_position_size': train_dataset.view_position_size,
        'label_size': train_dataset.label_size,
    })

    model = DataParallel(Model(**kwargs)).to(device)
    logger.info(f'Model info:\n{model}')

    if FLAGS.ckpt_path:
        logger.info(f'Loading model from {FLAGS.ckpt_path}')
        model.load_state_dict(torch.load(FLAGS.ckpt_path), strict=False)

    if FLAGS.debug:
        working_dir = os.path.join(FLAGS.working_dir, 'debug')
    else:
        working_dir = FLAGS.working_dir

    working_dir = os.path.join(working_dir, datetime.datetime.now().strftime('%Y-%m-%d-%H%M%S-%f'))

    locals()[FLAGS.do]()
