import tensorboardX
import torch


def unpad_sequence(tensor, lengths, is_sequence=True):
    if not is_sequence:
        tensor = torch.stack([tensor] * int(lengths.max()), 1)

    sequences = tensor.unbind(0)
    sequences = [sequence[:length] for (sequence, length) in zip(sequences, lengths)]

    return sequences


def teacher_sequence(tensor):
    return tensor[:, 1:]


def length_sorted(func):
    def _maybe_slice(value, index):
        if isinstance(value, torch.Tensor):
            return value[index]
        else:
            return value

    def _func(*args, **kwargs):
        length = kwargs.get('length')

        assert length is not None

        index = length.argsort(descending=True)
        _index = index.argsort()

        _args = [_maybe_slice(value, index) for value in args]
        _kwargs = {key: _maybe_slice(value, index) for (key, value) in kwargs.items()}

        outputs = func(*args, **kwargs)

        return [output[_index] for output in outputs]

    return _func
