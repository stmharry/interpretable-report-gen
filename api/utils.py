import datetime
import logging
import tensorboardX
import torch
import os

_logger = logging.getLogger(__name__)


def to_numpy(tensor):
    return tensor.detach().cpu().numpy()


def sent_to_report(tensor, sent_length, text_length):
    tensors = tensor.split(text_length.tolist(), 0)
    sent_lengths = sent_length.split(text_length.tolist(), 0)

    texts = [
        pack_padded_sequence(tensor, length=sent_length)
        for (tensor, sent_length) in zip(tensors, sent_lengths)
    ]
    return texts


""" RNN
"""

def expand_to_sequence(tensor, length):
    return torch.stack([tensor] * int(length), 1)


def pack_padded_sequence(tensor, length):
    sequences = tensor.unbind(0)
    sequences = [sequence[:length] for (sequence, length) in zip(sequences, length)]

    return torch.cat(sequences, 0)


def pad_packed_sequence(tensor, length, padding_value=0):
    sequences = tensor.split(length.tolist(), 0)

    return torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True, padding_value=padding_value)


def pad_sequence(sequences, batch_first=False, padding_value=0, total_length=None):
    total_length = total_length or max([sequence.size(0) for sequence in sequences])
    trailing_dims = sequences[0].size()[1:]

    if batch_first:
        out_dims = (len(sequences), total_length) + trailing_dims
    else:
        out_dims = (total_length, len(sequences)) + trailing_dims

    tensor = torch.full(out_dims, padding_value, dtype=sequences[0].dtype).cuda()
    for (num, sequence) in enumerate(sequences):
        length = sequence.size(0)

        if batch_first:
            tensor[num, :length, ...] = sequence
        else:
            tensor[:length, num, ...] = sequence

    return tensor


""" Log
"""


class SummaryWriter(tensorboardX.SummaryWriter):
    def add_log(self, log, prefix, global_step=None):
        for (key, value) in log.items():
            self.add_scalar(f'{prefix}/{key}', value, global_step=global_step)

    def add_texts(self, texts, name, prefix, index_to_word, global_step=None):
        log_text = '\n'.join(
            ['| num | text |'] +
            ['|:---:|:-----|'] +
            [
                '|{}|{}|'.format(num_text, ' '.join([index_to_word[int(word)] for word in text]))
                for (num_text, text) in enumerate(texts)
            ]
        )
        self.add_text(f'{prefix}/{name}', log_text, global_step=global_step)


def print_batch(batch, logger=None):
    logger = logger or _logger
    for (key, value) in batch.items():
        logger.info(f'{key}: shape={value.shape}')


""" Version
"""

def version_of(ckpt_path):
    if ckpt_path is None:
        version = datetime.datetime.now().timestamp()
    else:
        ckpt_dir = os.path.basename(os.path.dirname(ckpt_path))
        version = datetime.datetime.strptime(ckpt_dir, '%Y-%m-%d-%H%M%S-%f').timestamp()

    return version
