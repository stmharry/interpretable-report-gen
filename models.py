import re
import torch
import torch.nn.functional as F
import torchvision

from torch.nn import (
    Module,
    Embedding,
    Conv2d,
    Linear,
    ReLU,
    Dropout,
)


class ImageEncoder(Module):
    image_embedding_size = 1024

    def __init__(self, image_size=7):
        super(ImageEncoder, self).__init__()

        self.net = torchvision.models.DenseNet(num_init_features=64, growth_rate=32, block_config=(6, 12, 24, 16))

    def load_state_dict(self, state_dict):
        pattern = re.compile(r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')

        _state_dict = {}
        for key in state_dict.keys():
            match = pattern.match(key)
            if match:
                _key = match.group(1) + match.group(2)
            else:
                _key = key

            _state_dict[_key[19:]] = state_dict[key]

        self.net.load_state_dict(_state_dict, strict=False)

    def forward(self, v):
        v = self.net.features(v)
        v = F.adaptive_avg_pool2d(v, self.image_size)
        v = v.view(-1, self.image_embedding_size, self.image_size * self.image_size).transpose_(1, 2)

        return v


class ReportDecoder(Module):
    pass


class LSTMCell(Module):
    def __init__(self, input_size, hidden_size):
        super(LSTMCell, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        self.fc = Linear(input_size, 5 * hidden_size)

    def forward(self, x, _h, _m):
        xh = torch.cat([x, _h], dim=1)
        (i, f, o, g, m) = self.fc(xh).split([self.hidden_size] * 5, dim=1)

        i = F.sigmoid(i)
        f = F.sigmoid(f)
        o = F.sigmoid(o)
        m = i * F.tanh(m) + f * _m

        h = o * F.tanh(m)
        s = g * F.tanh(m)

        return (h, m, s)


class AdaptiveAttention(Module):
    def __init__(self, image_size, hidden_size):
        super(AdaptiveAttention, self).__init__()

        self.image_size = image_size
        self.hidden_size = hidden_size

        self.fc_v = Linear(hidden_size, hidden_size)
        self.fc_h = Linear(hidden_size, hidden_size)
        self.fc_s = Linear(hidden_size, hidden_size)
        self.fc_z = Linear(hidden_size, 1)


    def forward(self, _v, _h, _s):
        '''
        v: [batch_size, image_size * image_size, hidden_size]
        h: [batch_size, hidden_size]
        s: [batch_size, hidden_size]
        '''

        _h = _h.unsqueeze_(_h, 1)
        _s = _s.unsqueeze_(_s, 1)

        v = self.fc_v(_v)
        h = self.fc_h(_h)
        s = self.fc_s(_s)

        z = F.tanh(torch.cat([v, h], 1) + s)
        z = self.fc_z(z)
        a = F.softmax(z, 1)

        c = torch.sum(a * torch.cat([_v, _h], 1), 1)

        return (a, c)


class SetenceDecoder(Module):
    def __init__(self,
                 vocab_size,
                 image_embedding_size,
                 embedding_size,
                 hidden_size):

        super(SetenceDecoder, self).__init__()

        self.word_embedding = Embedding(vocab_size, embedding_size)
        self.fc_v = Linear(1024, embedding_size)
        self.fc_h = Linear(embedding_size, hidden_size)
        self.fc_m = Linear(embedding_size, hidden_size)
        self.lstm_cell = LSTMCell(2 * embedding_size, hidden_size)
        self.adaptive_attention = AdaptiveAttention(image_size, hidden_size)
        self.fc_p = Linear(hidden_size, vocab_size)

        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size

    def forward(self, v, w, lengths):
        '''
        v: [batch_size, image_size * image_size, hidden_size]
        w: [batch_size, max_length]
        '''

        (lengths, indices) = lengths.sort(0, descending=True)
        v = v[indices]
        w = w[indices]

        w = self.word_embedding(w)
        v = self.fc_v(v)
        vg = v.mean(1)

        _h = self.fc_h(vg)
        _m = self.fc_m(vg)
        for t in range(max(lengths)):
            batch_size_t = sum(lengths > t)

            x = torch.cat([
                w[:batch_size_t, t],
                vg.expand(batch_size_t, -1),
            ], 1)
            (h, m, s) = self.lstm_cell(x, _h, _m)
            (a, c) = self.adaptive_attention(v, h, s)
            p = F.softmax(self.fc_p(c + h), 1)

            import pdb; pdb.set_trace()


if __name__ == '__main__':

    device = torch.device('cuda')
    image_encoder = ImageEncoder().to(device)
    sentence_decoder = SetenceDecoder(
        vocab_size=1337,
        image_embedding_size=image_encoder.image_embedding_size,
        embedding_size=256,
        hidden_size=256,
    ).to(device)

    v = torch.zeros((64, 3, 224, 224)).to(device)
    w = torch.zeros((64, 32), dtype=torch.long).to(device)
    lengths = torch.full((64,), 32, dtype=torch.long).to(device)

    v = image_encoder(v)
    sentence_decoder(v, w, lengths)
