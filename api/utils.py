import datetime
import json
import logging
import tensorboardX
import torch
import os

_logger = logging.getLogger(__name__)


def to_numpy(tensor):
    return tensor.detach().cpu().numpy()


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
    max_length = max([sequence.size(0) for sequence in sequences])
    trailing_dims = sequences[0].size()[1:]

    if batch_first:
        out_dims = (len(sequences), total_length or max_length) + trailing_dims
    else:
        out_dims = (max_length, total_length or len(sequences)) + trailing_dims

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

    def add_texts(self, texts, name, prefix, global_step=None):
        log_text = '\n'.join(
            ['| num | text |'] +
            ['|:---:|:-----|'] +
            ['|{}|{}|'.format(num_text, text) for (num_text, text) in enumerate(texts)]
        )
        self.add_text(f'{prefix}/{name}', log_text, global_step=global_step)


def print_batch(batch, logger=None):
    logger = logger or _logger
    for (key, value) in batch.items():
        logger.info(f'{key}: shape={value.shape}')


""" Version
"""

def version_of(ckpt_path, ascend=False):
    if ckpt_path is None:
        version = datetime.datetime.now().timestamp()
    else:
        ckpt_dir = os.path.dirname(ckpt_path)
        with open(os.path.join(ckpt_dir, 'meta.json'), 'r') as f:
            ckpt_path = json.load(f).get('ckpt_path')

        if ckpt_path is None or (not ascend):
            version = datetime.datetime.strptime(os.path.basename(ckpt_dir), '%Y-%m-%d-%H%M%S-%f').timestamp()
        else:
            version = version_of(ckpt_path)

    return version


""" Misc
"""

def identity(func):
    return func

profile = __builtins__.get('profile', identity)
