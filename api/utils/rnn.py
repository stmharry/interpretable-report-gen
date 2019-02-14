import torch


def expand_to_sequence(tensor, length):
    return torch.stack([tensor] * int(length), 1)


def pack_padded_sequence(tensor, length):
    sequences = tensor.unbind(0)
    sequences = [sequence[:_length] for (sequence, _length) in zip(sequences, length)]

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
