import collections
import torch
import torch.utils.data


class CollateFn(object):
    def __init__(self, pad_fields=None):
        self.pad_fields = pad_fields or []

    def __call__(self, batch, pad=False):
        if isinstance(batch[0], torch.Tensor):
            if pad:
                return torch.nn.utils.rnn.pad_sequence(batch, batch_first=True)
            else:
                return torch.stack(batch, 0)

        elif isinstance(batch[0], collections.Mapping):
            return {key: self([d[key] for d in batch], pad=(key in self.pad_fields)) for key in batch[0]}

        return torch.utils.data.dataloader.default_collate(batch)
