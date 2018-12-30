import tensorboardX
import torch

""" RNN
"""

def expand_to_sequence(tensor, length):
    return torch.stack([tensor] * int(length), 1)


def pack_padded_sequence(tensor, lengths):
    sequences = tensor.unbind(0)
    sequences = [sequence[:length] for (sequence, length) in zip(sequences, lengths)]

    return torch.cat(sequences, 0)


def pad_packed_sequence(tensor, lengths):
    sequences = tensor.split(lengths.tolist(), 0)

    return torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True)


def teacher_sequence(tensor):
    return tensor[:, 1:]


def length_sorted_rnn(use_fields=None):
    use_fields = use_fields or []

    def _length_sorted_rnn(func):
        def _func(self, batch, length):
            index = length.argsort(descending=True)
            _index = index.argsort()

            batch = func(self, {key: batch[key][index] for key in use_fields}, length[index])
            return {key: batch[key][_index] for key in batch.keys()}

        return _func

    return _length_sorted_rnn


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


def print_batch(batch):
    for (key, value) in batch.items():
        print((key, value.shape))
