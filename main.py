import functools
import logging
import nltk
import numpy as np
import os
import pandas as pd
import sys
import torch
import torch.utils.data
import warnings

from absl import flags

from mimic_cxr.data import MIMIC_CXR

from api.datasets import Dataset
from api.data_loader import CollateFn
from api.models import ImageEncoder, ReportDecoder, SetenceDecoder
from api.utils import expand_to_sequence, join_sequence

### Global
flags.DEFINE_string('do', 'main', 'Function to execute (default: "main")')
flags.DEFINE_list('devices', ['cuda'], 'GPU devices to use (default: "cuda")')
flags.DEFINE_bool('debug', False, 'Turn on debug mode')

### Image
flags.DEFINE_integer('image_size', 8, 'Image feature map size')

### Text
flags.DEFINE_string('field', 'findings', 'The field to use in text reports')
flags.DEFINE_integer('max_report_length', 16, 'Maximum number of sentences in a report')
flags.DEFINE_integer('max_sentence_length', 64, 'Maximum number of words in a sentence')

### General
flags.DEFINE_integer('embedding_size', 256, 'Embedding size before feeding into the RNN')
flags.DEFINE_integer('hidden_size', 256, 'Hidden size in the RNN')
flags.DEFINE_integer('num_workers', 8, 'Number of data loading workers')
flags.DEFINE_integer('batch_size', 4, 'Batch size')
FLAGS = flags.FLAGS

### __TEMP__
vocab_size = 1337
label_size = 16


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

    dataset = Dataset(
        df=df,
        mimic_cxr=mimic_cxr,
        max_report_length=FLAGS.max_report_length,
        max_sentence_length=FLAGS.max_sentence_length,
    )
    DataLoader = functools.partial(
        torch.utils.data.DataLoader,
        batch_size=FLAGS.batch_size,
        num_workers=FLAGS.num_workers,
        collate_fn=CollateFn(pad_fields=['text', 'label', 'sent_length']),
    )
    data_loader = DataLoader(
        dataset=dataset,
        shuffle=True,
    )

    device = torch.device(FLAGS.devices[0])
    image_encoder = ImageEncoder(
        image_size=FLAGS.image_size,
        embedding_size=FLAGS.embedding_size,
    ).to(device)
    report_decoder = ReportDecoder(
        label_size=label_size,
        embedding_size=FLAGS.embedding_size,
        hidden_size=FLAGS.hidden_size,
    ).to(device)
    sentence_decoder = SetenceDecoder(
        vocab_size=vocab_size,
        label_size=label_size,
        image_size=FLAGS.image_size,
        image_embedding_size=image_encoder.image_embedding_size,
        embedding_size=FLAGS.embedding_size,
        hidden_size=FLAGS.hidden_size,
    ).to(device)

    batch = iter(data_loader).next()

    (
        image,
        view_position,
        text_length,
        text,
        label,
        sent_length,
    ) = [batch[key].to(device) for key in [
        'image',
        'view_position',
        'text_length',
        'text',
        'label',
        'sent_length',
    ]]

    # TODO(stmharry): pass in view_position

    image = image_encoder(image)
    (_label, _topic, _temp, _stop) = report_decoder(image, label, length=text_length)

    # TODO(stmharry): report level loss here

    image = expand_to_sequence(image, max(text_length))
    image = join_sequence(image, text_length)

    text = join_sequence(text, text_length)
    label = join_sequence(label, text_length)
    sent_length = join_sequence(sent_length, text_length)

    _topic = join_sequence(_topic, text_length)
    _temp = join_sequence(_temp, text_length)
    _stop = join_sequence(_stop, text_length)

    (_attention, _probability) = sentence_decoder(image, text, label, _topic, _temp, length=sent_length)

    # TODO(stmharry): sentence level loss here

    import pdb; pdb.set_trace()


if __name__ == '__main__':
    argv = FLAGS(sys.argv)

    logging.basicConfig(level=logging.DEBUG if FLAGS.debug else logging.INFO)
    logger = logging.getLogger(__name__)

    warnings.filterwarnings('ignore')

    logger.info('Loading MIMIC-CXR')
    mimic_cxr = MIMIC_CXR()

    locals()[FLAGS.do]()
