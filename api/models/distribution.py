import torch


class Categorical(object):
    def __init__(self, logits):
        self.logits = logits

    def sample(self):
        u = torch.rand_like(self.logits)
        z = self.logits - torch.log(-u.log())
        return z.argmax(1)

    def argmax(self):
        return self.logits.argmax(1)

    def log_prob(self, x):
        return self.logits.gather(1, x.unsqueeze(1)).squeeze(1)
