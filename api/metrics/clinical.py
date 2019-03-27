import torch
import torch.nn as nn


class MentionSim(nn.Module):
    """ Computes the similarity of two CheXpert annotations

    Args:
        input_(torch.Tensor): Annotation of the report of interest (batch_size x num_labels)
        target(torch.Tensor): Annotation of the reference report (batch_size x num_labels)

    Returns:
        sim(torch.Tensor): The similarity, following this definition: (unfilled denote symmetric)

                                   target
                  | 0 |                     1 |     2 | 3
               ===+===+=======================+=======+===
                0 | 1 |                 1 - a |     1 | 0
               ---+---+-----------------------+-------+---
        input_  1 |   | (1 - a) ** 2 + a ** 2 | 1 - a | a
               ---+---+-----------------------+-------+---
                2 |   |                       |     1 | 0
               ---+---+-----------------------+-------+---
                3 |   |                       |       | 1
    """

    mention_size = 4

    def __init__(self, alpha=0.5):
        super(MentionSim, self).__init__()

        self._sim_lookup = nn.Parameter(torch.as_tensor(
            [
                1, 1 - alpha, 1, 0,
                1 - alpha, (1 - alpha) ** 2 + alpha ** 2, 1 - alpha, alpha,
                1, 1 - alpha, 1, 0,
                0, alpha, 0, 1
            ],
            dtype=torch.float32,
        ), requires_grad=False)

    def forward(self, input_, target):
        assert input_.shape == target.shape

        lookup = input_ * MentionSim.mention_size + target
        sim = self._sim_lookup[lookup]

        return sim
