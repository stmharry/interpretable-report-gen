import torch


class MetricMixin:
    def __call__(self, input_, target):
        (_, scores) = self.compute_score(
            {num: [_target] for (num, _target) in enumerate(target)},
            {num: [_input] for (num, _input) in enumerate(input_)},
        )

        metric = torch.as_tensor(scores, dtype=torch.float).cuda()
        if metric.dim() == 1:
            metric = metric.unsqueeze(0)

        return metric.t()
