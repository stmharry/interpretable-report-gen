import tensorboardX
import torch


def expand_to_sequence(tensor, length):
    return torch.stack([tensor] * int(length), 1)


def unpad_sequence(tensor, lengths):
    sequences = tensor.unbind(0)
    sequences = [sequence[:length] for (sequence, length) in zip(sequences, lengths)]

    return sequences


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


def print_batch(batch):
    for (key, value) in batch.items():
        print((key, value.shape))
