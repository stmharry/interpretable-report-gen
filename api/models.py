import logging
import torch
import torch.nn.functional as F

from torch.nn import (
    Module,
    Embedding,
    LSTMCell,
    AdaptiveAvgPool2d,
    Conv2d,
    Linear,
    Dropout,
)

from torchvision.models.resnet import (
    ResNet,
    Bottleneck,
)

logger = logging.getLogger(__name__)


class AdaptiveAttention(Module):
    def __init__(self, image_size, hidden_size):
        super(AdaptiveAttention, self).__init__()

        self.image_size = image_size
        self.hidden_size = hidden_size

        self.fc = Linear(hidden_size, 2 * hidden_size)
        self.fc_v = Linear(hidden_size, hidden_size)
        self.fc_h = Linear(hidden_size, hidden_size)
        self.fc_s = Linear(hidden_size, hidden_size)
        self.fc_z = Linear(hidden_size, 1)
        self.dropout = Dropout(0.5)

    def forward(self, v, h):
        """

        Args:
            v (batch_size, image_size * image_size, hidden_size): Image feature map.
            h (batch_size, hidden_size): Current hidden state.

        Returns:
            a (batch_size, image_size * image_size + 1): Attention weights.
            c (batch_size, hidden_size): Context vector.

        """

        h = h.unsqueeze(1)
        (h, s) = torch.tanh(self.fc(self.dropout(h))).split([self.hidden_size] * 2, 2)

        _v = self.fc_v(self.dropout(v))
        _h = self.fc_h(self.dropout(h))
        _s = self.fc_s(self.dropout(s))

        z = torch.tanh(torch.cat([_v, _s], 1) + _h)
        z = self.fc_z(self.dropout(z)).squeeze(2)
        a = F.softmax(z, 1)
        c = torch.sum(a.unsqueeze(2) * torch.cat([v, s], 1), 1)

        return (a, c)


class ImageEncoder(ResNet):
    image_embedding_size = 2048

    def __init__(self, image_size, embedding_size):
        super(ImageEncoder, self).__init__(Bottleneck, [3, 4, 6, 3])

        self.image_size = image_size
        self.embedding_size = embedding_size

        self.conv1 = Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.avgpool = AdaptiveAvgPool2d((image_size, image_size))
        self.fc = Conv2d(self.image_embedding_size, embedding_size, (1, 1))
        self.dropout = Dropout(0.5)

    def forward(self, v):
        """

        Args:
            v (batch_size, 1, 256, 256): Grayscale Image.

        Returns:
            v (batch_size, image_embedding_size, image_size, image_size): Image feature map.

        """

        v = self.conv1(v)
        v = self.bn1(v)
        v = self.relu(v)
        v = self.maxpool(v)

        v = self.layer1(v)
        v = self.layer2(v)
        v = self.layer3(v)
        v = self.layer4(v)

        v = self.avgpool(v)
        v = self.dropout(v)
        v = self.fc(v)
        v = v.view(-1, self.embedding_size, self.image_size * self.image_size).transpose(1, 2)

        return v


class ReportDecoder(Module):
    def __init__(self,
                 label_size,
                 embedding_size,
                 hidden_size):

        super(ReportDecoder, self).__init__()

        self.label_size = label_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size

        self.fc_h = Linear(embedding_size, hidden_size)
        self.fc_m = Linear(embedding_size, hidden_size)
        # (label, topic, temp, stop)
        self.fc_sizes = [self.label_size, self.hidden_size, 1, 1]
        self.fc = Linear(hidden_size, sum(self.fc_sizes))
        self.lstm_cell = LSTMCell(embedding_size + label_size, hidden_size)
        self.dropout = Dropout(0.5)

    def forward(self, v, l, lengths):
        """

        Args:
            v (batch_size, image_size * image_size, hidden_size): Image feature map.
            l (batch_size, max_length, label_size): Interpretable labels.
            lengths (batch_size,): Sequence lengths.

        Returns:
            label (batch_size, max_length, label_size): Generated labels.
            topic (batch_size, max_length, embedding_size): Generated topic embeddings.
            temp (batch_size, max_length, 1): Temperatures.
            stop (batch_size, max_length, 1): Stop decoding.

        """

        indices = lengths.argsort(descending=True)
        _indices = indices.argsort()

        v = v[indices]
        l = l[indices]
        lengths = lengths[indices]

        vg = v.mean(1)

        h = torch.tanh(self.fc_h(self.dropout(vg)))
        m = torch.tanh(self.fc_m(self.dropout(vg)))

        outputs = []
        for t in range(max(lengths)):
            logger.debug(f'ReportDecoder.forward(): time_step={t}')
            batch_size_t = sum(lengths > t)

            x = torch.cat([vg[:batch_size_t], l[:batch_size_t, t]], 1)
            h = h[:batch_size_t]
            m = m[:batch_size_t]

            (h, m) = self.lstm_cell(x, (h, m))
            (label, topic, temp, stop) = F.relu(self.fc(self.dropout(h))).split(self.fc_sizes, 1)
            # TODO(stmharry): process label, topic
            temp = torch.exp(temp)
            stop = torch.sigmoid(stop)

            outputs.append((label, topic, temp, stop))

        (label, topic, temp, stop) = [torch.nn.utils.rnn.pad_sequence(output) for output in zip(*outputs)]

        return (
            label[_indices],
            topic[_indices],
            temp[_indices],
            stop[_indices],
        )


class SetenceDecoder(Module):
    def __init__(self,
                 vocab_size,
                 image_size,
                 image_embedding_size,
                 embedding_size,
                 hidden_size):

        super(SetenceDecoder, self).__init__()

        self.word_embedding = Embedding(vocab_size, embedding_size)
        self.fc_h = Linear(embedding_size, hidden_size)
        self.fc_m = Linear(embedding_size, hidden_size)
        self.lstm_cell = LSTMCell(2 * embedding_size, hidden_size)
        self.adaptive_attention = AdaptiveAttention(image_size, hidden_size)
        self.fc_p = Linear(hidden_size, vocab_size)
        self.dropout = Dropout(0.5)

        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size

    def forward(self, v, w, lengths):
        """

        Args:
            v (batch_size, image_size * image_size, hidden_size): Image feature map.
            w (batch_size, max_length): Sentences.
            lengths (batch_size,): Sequence lengths.

        Returns:
            attention (batch_size, max_length, image_size * image_size + 1): Attention weights.
            prob (batch_size, max_length, vocab_size): Generated probability on words.

        """

        max_length = max(lengths)

        indices = lengths.argsort(descending=True)
        _indices = indices.argsort()

        v = v[indices]
        w = w[indices]
        lengths = lengths[indices]

        w = self.word_embedding(self.dropout(w))
        vg = v.mean(1)

        h = torch.tanh(self.fc_h(self.dropout(vg)))
        m = torch.tanh(self.fc_m(self.dropout(vg)))

        outputs = []
        for t in range(max_length):
            batch_size_t = sum(lengths > t)

            x = torch.cat([vg.expand(batch_size_t, -1), w[:batch_size_t, t]], 1)
            (h, m) = self.lstm_cell(x, (h, m))
            (a, c) = self.adaptive_attention(v, h)
            p = F.softmax(self.fc_p(self.dropout(c + h)), 1)

            outputs.append((a, p))

        (attention, prob) = [torch.nn.utils.rnn.pad_sequence(output) for output in zip(*outputs)]

        return (
            attention[_indices],
            prob[_indices],
        )
