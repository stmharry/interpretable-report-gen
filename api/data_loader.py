import collections
import torch
import torch.utils.data


class CollateFn(object):
    def __init__(self, sequence_fields=None):
        self.sequence_fields = sequence_fields or []

    def __call__(self, batch, is_sequence=False):
        if isinstance(batch[0], torch.Tensor):
            if is_sequence:
                return torch.nn.utils.rnn.pad_sequence(batch, batch_first=True)
            else:
                return torch.stack(batch, 0)

        elif isinstance(batch[0], collections.Mapping):
            return {
                key: self([d[key] for d in batch], is_sequence=(key in self.sequence_fields))
                for key in batch[0]
            }

        return torch.utils.data.dataloader.default_collate(batch)
