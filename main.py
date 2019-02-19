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
import matplotlib
import matplotlib.pyplot as plot
import os
import pandas as pd
import sklearn.metrics
import sys
import torch
import torch.utils.data
import tqdm
import warnings

from absl import flags
from htmltag import HTML, head, body, link, script, div, span, img, p
from json2html import json2html
from torch.nn import (
    functional as F,
    DataParallel,
)

from mimic_cxr.data import MIMIC_CXR
from mimic_cxr.utils import Log

from api import Mode, Phase
from api.datasets import MimicCXRDataset
from api.data_loader import CollateFn
from api.models import Model, DataParallelCPU, SentIndex2Report, CheXpert, ExponentialMovingAverage
from api.metrics import Bleu, Rouge, CiderD as Cider, MentionSim
from api.utils import to_numpy, profile
from api.utils.io import version_of, load_state_dict
from api.utils.log import print_batch, SummaryWriter
from api.utils.rnn import pack_padded_sequence, pad_packed_sequence

""" Global
"""
flags.DEFINE_string('do', 'train', 'Function to execute (default: "train")')
flags.DEFINE_enum('test_split', 'test', ['val', 'test'], 'Split to test for')
flags.DEFINE_enum('mode', None, list(Mode.__members__.keys()), 'Training mode (' + ' or '.join(map('"{}"'.format, Mode.__members__)) + ')')
flags.DEFINE_string('device', 'cuda', 'GPU device to use (default: "cuda")')
flags.DEFINE_bool('debug', False, 'Turn on debug mode (default: False)')
flags.DEFINE_bool('debug_subsample', False, 'Turn on subsampling for debugging (default: False)')
flags.DEFINE_string('debug_use_dataset', None, 'Turn on alternate dataset postfix (default: None)')
flags.DEFINE_bool('debug_one_sentence', False, 'Turn on one sentence mode (default: False)')
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

    version = version_of(FLAGS.ckpt_path)
    kwargs = FLAGS.flag_values_dict()
    kwargs.update({
        '__use_continuous_label': version_of(FLAGS.ckpt_path) > 1548152415,
        '__no_recurrent_label': version_of(FLAGS.ckpt_path) > 1548347568,
        '__image_encoder_relu': version_of(FLAGS.ckpt_path) > 1548428102,
        '__sample_text': version_of(FLAGS.ckpt_path) > 1548708003,
        '__use_densenet': version_of(FLAGS.ckpt_path, ascend=True) > 1548881554,
        '__no_temp': version_of(FLAGS.ckpt_path) > 1548881554,
    })
    logger.info(json.dumps(kwargs, indent=4))

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
        'word_embedding': MimicCXRDataset.word_embedding,
        'index_to_word': MimicCXRDataset.index_to_word,
        'word_to_index': MimicCXRDataset.word_to_index,
        'view_position_size': MimicCXRDataset.view_position_size,
        'label_size': MimicCXRDataset.label_size,
    })

    model = Model(**kwargs)
    if kwargs['__use_densenet']:
        path = os.path.join(os.path.dirname(__file__), 'checkpoints', 'model.pkl')
        logger.info(f'Loading model from {path}')
        load_state_dict(model.image_encoder, torch.load(path)['state_dict'])

    model = DataParallel(model).to(FLAGS.device)
    logger.info(f'Model info:\n{model}')

    if FLAGS.ckpt_path:
        logger.info(f'Loading model from {FLAGS.ckpt_path}')
        load_state_dict(model, torch.load(FLAGS.ckpt_path))

    if FLAGS.debug:
        working_dir = os.path.join(FLAGS.working_dir, 'debug')
    else:
        working_dir = FLAGS.working_dir

    working_dir = os.path.join(working_dir, datetime.datetime.now().strftime('%Y-%m-%d-%H%M%S-%f'))

    optimizer = torch.optim.Adam(model.parameters(), lr=FLAGS.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, FLAGS.lr_decay_epochs, FLAGS.lr_decay)

    converter = SentIndex2Report(index_to_word=MimicCXRDataset.index_to_word)
    chexpert = DataParallelCPU(CheXpert).to(FLAGS.device)
    bleu = Bleu(4).to(FLAGS.device)
    rouge = Rouge().to(FLAGS.device)
    cider = Cider(df_cache=MimicCXRDataset.df_cache).to(FLAGS.device)
    mention_sim = MentionSim(alpha=0.5).to(FLAGS.device)
    ema = ExponentialMovingAverage(beta=0.95).to(FLAGS.device)

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

        if mode & Mode.use_teacher_forcing:
            teacher_forcing_ratio = max(0, 1 - (num_epoch // FLAGS.tf_decay_epochs) * FLAGS.tf_decay)

        for phase in phases:
            log = Log()
            log_items = []

            if phase == Phase.train:
                data_loader = train_loader
                torch.set_grad_enabled(True)

            elif phase == Phase.val:
                data_loader = val_loader
                torch.set_grad_enabled(False)

            prog = tqdm.tqdm(enumerate(data_loader), total=len(data_loader))
            for (num_batch, batch) in prog:
                num_step = num_epoch * len(train_loader) + num_batch

                if phase == Phase.train:
                    optimizer.zero_grad()

                for (key, value) in batch.items():
                    batch[key] = value.to(FLAGS.device)

                losses = {}
                metrics = {}

                if phase == Phase.train:
                    kwargs = {}

                    if mode & Mode.use_teacher_forcing:
                        kwargs.update({'teacher_forcing_ratio': teacher_forcing_ratio})

                    if mode & Mode.use_self_critical:
                        kwargs.update({'beam_size': 1})

                    output = model(batch, phase=Phase.train, **kwargs)

                    if mode & Mode.use_label_all_ce:
                        losses['label_ce'] = F.binary_cross_entropy(output['_label'], output['label'].sum(1).clamp(0, 1))

                    if mode & Mode.use_label_ce:
                        losses['label_ce'] = F.binary_cross_entropy(output['_label'], output['label'])

                    if mode & Mode.use_stop_bce:
                        losses['stop_bce'] = F.binary_cross_entropy(output['_stop'], output['stop'])

                    if mode & Mode.use_teacher_forcing:
                        report = converter(output['text'], output['sent_length'], output['text_length'])
                        _report = converter(output['_text'], output['sent_length'], output['text_length'])

                        word = pack_padded_sequence(output['text'], length=output['sent_length'])
                        _log_probability = pack_padded_sequence(output['_log_probability'], length=output['sent_length'])

                        losses['word_ce'] = F.nll_loss(_log_probability, word)

                    if mode & Mode.use_self_critical:
                        output = model(batch, phase=Phase.train, beam_size=1)

                        report = converter(output['text'], output['sent_length'], output['text_length'])
                        _report = converter(output['_text'][:, 0], output['_sent_length'][:, 0], output['_text_length'])
                        _sum_log_probability = pad_packed_sequence(output['_sum_log_probability'][:, 0], length=output['_text_length']).sum(1)

                        with torch.no_grad():
                            output_greedy = model(batch, phase=Phase.test, beam_size=1)
                        _report_greedy = converter(output_greedy['_text'][:, 0], output_greedy['_sent_length'][:, 0], output_greedy['_text_length'])

                        report_cider = cider(_report, report)
                        report_cider_greedy = cider(_report_greedy, report)

                        metrics['cider'] = report_cider.mean()

                        _chexpert_label = chexpert(_report)
                        sim = mention_sim(_chexpert_label, output['chexpert_label'])
                        sim_baseline = ema(sim).mean()

                        metrics['sim'] = sim.mean()

                        reward_report = report_cider - report_cider_greedy
                        reward_sim = sim - sim_baseline
                        reward = reward_report + reward_sim

                        losses['REINFORCE'] = - (_sum_log_probability * reward).mean()

                        metrics['reward_report'] = reward_report.mean()
                        metrics['reward_sim'] = reward_sim.mean()
                        metrics['reward'] = reward.mean()

                elif phase == Phase.val:
                    kwargs = {}

                    if mode & Mode.gen_text:
                        kwargs.update({
                            'beam_size': FLAGS.beam_size,
                            'alpha': FLAGS.alpha,
                            'beta': FLAGS.beta,
                        })

                    output = model(batch, phase=Phase.val, **kwargs)

                    if mode & Mode.use_label_all_ce:
                        label_all = output['label'].sum(1).clamp(0, 1)
                        _label_all = output['_label']

                        losses['label_ce_all'] = F.binary_cross_entropy(_label_all, label_all)

                        log_items.append({
                            'label_all': label_all,
                            '_label_all': _label_all,
                        })

                    if mode & Mode.use_label_ce:
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

                    if mode & Mode.gen_text:
                        report = converter(output['text'], output['sent_length'], output['text_length'])
                        _report = converter(output['_text'][:, 0], output['_sent_length'][:, 0], output['_text_length'])

                        for scorer in [bleu, rouge, cider]:
                            report_score = scorer(_report, report)

                            if report_score.dim() == 2:
                                for (num, _report_score) in enumerate(report_score):
                                    metrics[f'{scorer.method()}-{num + 1}'] = _report_score.mean()
                            else:
                                metrics[f'{scorer.method()}'] = report_score.mean()

                        _chexpert_label = chexpert(_report)
                        sim = mention_sim(_chexpert_label, output['chexpert_label'])

                        metrics['sim'] = sim.mean()

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
                        if mode & Mode.use_teacher_forcing:
                            writer.add_scalar(f'{phase}/teacher_forcing_ratio', teacher_forcing_ratio, global_step=num_step)
                        writer.add_scalar(f'{phase}/learning_rate', optimizer.param_groups[0]['lr'], global_step=num_step)
                        writer.add_log(_log, prefix=phase, global_step=num_step)

                    if (num_step % FLAGS.log_text_steps == 0) and (mode & Mode.gen_text):
                        writer.add_texts(report, 'text', prefix=phase, global_step=num_step)
                        writer.add_texts(_report, '_text', prefix=phase, global_step=num_step)

                if FLAGS.profile and (num_batch == 32):
                    break

            if phase == Phase.train:
                if num_epoch % FLAGS.save_epochs == 0:
                    torch.save(model.state_dict(), os.path.join(working_dir, f'model-epoch-{num_epoch}.pth'))

            elif phase == Phase.val:
                if (mode & Mode.use_label_all_ce) or (mode & Mode.use_label_ce):
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
    converter = SentIndex2Report(index_to_word=MimicCXRDataset.index_to_word)

    image_dir = os.path.join(working_dir, 'imgs')
    os.makedirs(image_dir)

    prog = tqdm.tqdm(enumerate(data_loader), total=len(data_loader))
    for (num_batch, batch) in prog:
        for (key, value) in batch.items():
            batch[key] = value.to(FLAGS.device)

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
            report_texts = convert(batch['text'], batch['sent_length'], batch['text_length'])

        item_index = batch['item_index']
        _text_length = batch['_text_length']  # (num_reports,)
        (
            _label,                # (num_reports, max_num_sentences, label_size)
            _stop,                 # (num_reports, max_num_sentences, 1)
            _temp,                 # (num_reports, max_num_sentences, 1)
            _attention,            # (num_repoers, max_num_sentences, beam_size, max_num_words, 65)
            _score,                # (num_reports, max_num_sentences, beam_size)
            _sum_log_probability,  # (num_reports, max_num_sentences, beam_size)
            _text,                 # (num_reports, max_num_sentences, beam_size, max_num_words)
            _sent_length,          # (num_reports, max_num_sentences, beam_size)
        ) = [
            pad_packed_sequence(batch[key], length=_text_length)
            for key in ['_label', '_stop', '_temp', '_attention', '_score', '_sum_log_probability', '_text', '_sent_length']
        ]

        (fig, ax) = plot.subplots(1, figsize=(6, 6), gridspec_kw={'left': 0, 'right': 1, 'bottom': 0, 'top': 1})
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
                    _words = MimicCXRDataset.index_to_word[to_numpy(_sentence[:num_words])]

                    beam_texts = []
                    for num_word in range(num_words):
                        _attention_path = os.path.join(image_dir, f'{num_report}-{num_sentence}-{num_beam}-{num_word}.png')

                        _a = _attention[num_report, num_sentence, num_beam, num_word, :FLAGS.image_size * FLAGS.image_size]
                        _a_sum = _a.sum()
                        _a = _a / _a.max()
                        _a = _a.reshape(FLAGS.image_size, FLAGS.image_size)
                        _a = F.interpolate(_a[None, None, :], scale_factor=(8, 8), mode='bilinear', align_corners=True)[0, 0]

                        ax.contourf(to_numpy(_a), cmap='Reds')
                        ax.set_axis_off()

                        fig.savefig(_attention_path, bbox_inches=0)
                        fig.clf()

                        beam_texts.append(span(html.escape(_words[num_word]), **{
                            'data-toggle': 'tooltip',
                            'title': (
                                p('Total attention: ' + to_str(_a_sum)) +
                                img(src=f'http://monday.csail.mit.edu/xiuming{_attention_path}', width='128', height='128')
                            ).replace('"', '\''),
                        }))

                    beams.append({
                        'score': to_str(_score[num_report, num_sentence, num_beam]),
                        'log_prob': to_str(_sum_log_probability[num_report, num_sentence, num_beam]),
                        'text': str('\n'.join(beam_texts)),
                    })

                sentence = {}

                if mode & Mode.use_label_ce:
                    sentence.update({
                        'stop': to_str(_stop[num_report, num_sentence]),
                        'temp': to_str(_temp[num_report, num_sentence]),
                        'labels': [{
                            'label': label_col,
                            'prob': to_str(_l),
                        } for (label_col, _l) in zip(MimicCXRDataset.label_columns, _label[num_report, num_sentence])],
                    })

                if mode & Mode.gen_text:
                    sentence.update({
                        'beams': beams,
                    })

                sentences.append(sentence)

            report = {
                'image': str(img(src=f'http://monday.csail.mit.edu/xiuming{image_path}', width='256')),
                'generated text': sentences,
            }
            if FLAGS.test_split == 'val':
                report['ground truth text'] = report_texts[num_report]

            reports.append(report)

        if num_batch == 1:
            break

    s = json2html.convert(json=reports, table_attributes='class="table table-striped"', escape=False)
    s = (
        head([
            link(rel='stylesheet', href='https://maxcdn.bootstrapcdn.com/bootstrap/3.3.7/css/bootstrap.min.css'),
            script(src='https://cdnjs.cloudflare.com/ajax/libs/jquery/3.3.1/jquery.min.js'),
            script(src='https://maxcdn.bootstrapcdn.com/bootstrap/3.3.7/js/bootstrap.min.js'),
            script('$(function () { $(\'[data-toggle="tooltip"]\').tooltip({placement: "bottom", html: true}); })', type="text/javascript"),
        ]) +
        body(div(div(div(HTML(s), _class='panel-body'), _class='panel panel-default'), _class='container-fluid'))
    )

    with open(os.path.join(working_dir, 'index.html'), 'w') as f:
        f.write(s)


if __name__ == '__main__':
    argv = FLAGS(sys.argv)

    logging.basicConfig(level=logging.DEBUG if FLAGS.debug else logging.INFO)
    logger = logging.getLogger(__name__)

    matplotlib.use('Agg')
    warnings.filterwarnings('ignore')

    mode = Mode[FLAGS.mode]
    if FLAGS.debug:
        FLAGS.num_workers = 0

    locals()[FLAGS.do]()
