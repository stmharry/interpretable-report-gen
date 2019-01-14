import logging
import torch
import torch.nn.functional as F

from torch.nn import (
    Module,
    LSTMCell,
    AdaptiveAvgPool2d,
    Conv2d,
    Embedding,
    Linear,
    Dropout,
)

from torchvision.models.resnet import (
    ResNet,
    Bottleneck,
)

from api import Token
from api.utils import length_sorted_rnn, pack_padded_sequence, pad_packed_sequence

logger = logging.getLogger(__name__)

if 'profile' not in dir(__builtins__):
    def profile(func):
        return func


class ImageEncoder(ResNet):
    image_embedding_size = 2048

    def __init__(self, **kwargs):
        super(ImageEncoder, self).__init__(Bottleneck, [3, 4, 6, 3])

        self.image_size     = kwargs['image_size']
        self.embedding_size = kwargs['embedding_size']
        self.dropout        = kwargs['dropout']

        self.conv1 = Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.avgpool = AdaptiveAvgPool2d((self.image_size, self.image_size))
        self.fc = Conv2d(self.image_embedding_size, self.embedding_size, (1, 1))
        self.dropout = Dropout(self.dropout)

    @profile
    def forward(self, batch):
        """

        Args:
            image (batch_size, 1, 256, 256): Grayscale Image.

        Returns:
            image (batch_size, image_embedding_size, image_size, image_size): Image feature map.

        """

        image = batch['image']

        image = self.conv1(image)
        image = self.bn1(image)
        image = self.relu(image)
        image = self.maxpool(image)

        image = self.layer1(image)
        image = self.layer2(image)
        image = self.layer3(image)
        image = self.layer4(image)

        image = self.avgpool(image)
        image = self.dropout(image)
        image = self.fc(image)
        image = image.view(-1, self.embedding_size, self.image_size * self.image_size).transpose(1, 2)

        return {'image': image}


class ReportDecoder(Module):
    def __init__(self, **kwargs):
        super(ReportDecoder, self).__init__()

        self.view_position_size = kwargs['view_position_size']
        self.label_size         = kwargs['label_size']
        self.embedding_size     = kwargs['embedding_size']
        self.hidden_size        = kwargs['hidden_size']
        self.dropout            = kwargs['dropout']

        self.fc_h = Linear(self.embedding_size, self.hidden_size)
        self.fc_m = Linear(self.embedding_size, self.hidden_size)
        self.fc_sizes = [
            self.label_size,   # label
            self.hidden_size,  # topic
            1,                 # stop
            1,                 # temp
        ]
        self.fc = Linear(self.hidden_size, sum(self.fc_sizes))
        self.lstm_sizes = [
            self.embedding_size,      # image
            self.view_position_size,  # view_position
            1,                        # begin
            self.label_size,          # label
        ]
        self.lstm_cell = LSTMCell(sum(self.lstm_sizes), self.hidden_size)
        self.dropout = Dropout(self.dropout)

    @length_sorted_rnn(use_fields=['image', 'view_position', 'label'])
    @profile
    def forward(self, batch, length):
        """

        Args:
            image (batch_size, image_size * image_size, hidden_size): Image feature map.
            view_position (batch_size, view_position_size): Patient positions.
            label (batch_size, max_length + 1, label_size): Interpretable labels.
            length (batch_size,): Sequence lengths.

        Returns:
            _label (batch_size, max_length, label_size): Generated labels.
            _topic (batch_size, max_length, embedding_size): Generated topic embeddings.
            _stop (batch_size, max_length, 1): Stop signal.
            _temp (batch_size, max_length, 1): Temperatures.

        """

        image         = batch['image']
        view_position = batch['view_position']
        label         = batch['label']

        image_mean = image.mean(1)

        h = torch.tanh(self.fc_h(self.dropout(image_mean)))
        m = torch.tanh(self.fc_m(self.dropout(image_mean)))

        outputs = []
        for t in range(torch.max(length - 1)):
            logger.debug(f'ReportDecoder.forward(): time_step={t}')
            batch_size_t = torch.sum(length - 1 > t)

            begin = torch.ones((batch_size_t, 1), dtype=torch.float).cuda() * (t == 0)
            x = torch.cat([
                self.dropout(image_mean[:batch_size_t]),
                view_position[:batch_size_t],
                begin,
                label[:batch_size_t, t],
            ], 1)
            h = h[:batch_size_t]
            m = m[:batch_size_t]

            (h, m) = self.lstm_cell(x, (h, m))
            (_label, _topic, _stop, _temp) = self.fc(self.dropout(h)).split(self.fc_sizes, 1)
            # TODO(stmharry): on label apply softmax group
            _label = torch.sigmoid(_label)
            _label = torch.where(_label >= 0.5, torch.tensor(1.0).cuda(), torch.tensor(0.0).cuda())
            _topic = F.relu(_topic)
            _stop = torch.sigmoid(_stop)
            _temp = torch.exp(_temp)

            outputs.append({
                # '_label': _label,
                '_topic': _topic,
                '_stop': _stop,
                '_temp': _temp,
            })

        outputs = {key: torch.nn.utils.rnn.pad_sequence([output[key] for output in outputs]) for key in outputs[0].keys()}
        return outputs


class SentenceDecoder(Module):
    def __init__(self, **kwargs):
        super(SentenceDecoder, self).__init__()

        self.image_size          = kwargs['image_size']
        self.view_position_size  = kwargs['view_position_size']
        self.word_to_index       = kwargs['word_to_index']
        self.label_size          = kwargs['label_size']
        self.embedding_size      = kwargs['embedding_size']
        self.hidden_size         = kwargs['hidden_size']
        self.dropout             = kwargs['dropout']
        self.max_sentence_length = kwargs['max_sentence_length']

        self.vocab_size = len(self.word_to_index)

        self.word_embedding = Embedding(self.vocab_size, self.embedding_size)
        self.fc_v = Linear(self.embedding_size, self.hidden_size)
        self.fc_h = Linear(self.embedding_size, self.hidden_size)
        self.fc_m = Linear(self.embedding_size, self.hidden_size)
        self.lstm_sizes = [
            self.embedding_size,      # image
            self.view_position_size,  # view_position
            self.label_size,          # label
            self.hidden_size,         # topic
            self.embedding_size,      # text
        ]
        self.lstm_cell = LSTMCell(sum(self.lstm_sizes), self.hidden_size)
        self.fc_sizes = [self.embedding_size, self.embedding_size]  # hidden, sentinel
        self.fc = Linear(self.hidden_size, sum(self.fc_sizes))
        self.fc_hh = Linear(self.embedding_size, self.hidden_size)
        self.fc_s = Linear(self.embedding_size, self.hidden_size)
        self.fc_z = Linear(self.hidden_size, 1)
        self.fc_p = Linear(self.embedding_size, self.vocab_size)
        self.dropout = Dropout(self.dropout)

    @length_sorted_rnn(use_fields=['image', 'view_position', 'text', 'label', '_topic', '_temp'])
    @profile
    def forward(self, batch, length):

        """

        Args:
            image (batch_size, image_size * image_size, hidden_size): Image feature map.
            view_position (batch_size, view_position_size): Patient positions.
            text (batch_size, max_length): Sentences.
            label (batch_size, label_size): Interpretable labels.
            topic (batch_size, embedding_size): Topic embeddings.
            temp (batch_size, 1): Temperature parameters.
            length (batch_size,): Sequence lengths.

        Returns:
            _attention (batch_size, max_length, image_size * image_size + 1): Attention weights.
            _log_probability (batch_size, max_length, vocab_size): Generated probability on words.

        """

        image         = batch['image']
        view_position = batch['view_position']
        text          = batch['text']
        label         = batch['label']
        topic         = batch['_topic']
        temp          = batch['_temp']

        text_embedding = self.word_embedding(text)
        image_mean = image.mean(1)

        v = self.fc_v(self.dropout(image))
        h = torch.tanh(self.fc_h(self.dropout(image_mean)))
        m = torch.tanh(self.fc_m(self.dropout(image_mean)))

        outputs = []
        for t in range(torch.max(length - 1)):
            logger.debug(f'SentenceDecoder.forward(): time_step={t}')
            batch_size_t = torch.sum(length - 1 > t)

            x = torch.cat([
                self.dropout(image_mean[:batch_size_t]),
                view_position[:batch_size_t],
                label[:batch_size_t],
                self.dropout(topic[:batch_size_t]),
                self.dropout(text_embedding[:batch_size_t, t]),
            ], 1)
            h = h[:batch_size_t]
            m = m[:batch_size_t]

            (h, m) = self.lstm_cell(x, (h, m))
            (hh, s) = F.relu(self.fc(self.dropout(h))).unsqueeze(1).split(self.fc_sizes, 2)
            _hh = self.fc_hh(self.dropout(hh))
            _s = self.fc_s(self.dropout(s))

            z = torch.tanh(torch.cat([v[:batch_size_t], _s], 1) + _hh)
            z = self.fc_z(self.dropout(z))
            a = F.softmax(z, 1)
            c = torch.sum(a * torch.cat([image[:batch_size_t], s], 1), 1)

            _attention = a.squeeze(2)
            _log_probability = F.log_softmax(self.fc_p(self.dropout(c + hh.squeeze(1))) / temp[:batch_size_t], 1)
            _text = _log_probability.argmax(1)

            outputs.append({
                '_attention': _attention,
                '_log_probability': _log_probability,
                '_text': _text,
            })

        outputs = {key: torch.nn.utils.rnn.pad_sequence([output[key] for output in outputs]) for key in outputs[0].keys()}
        return outputs

    @profile
    def decode(self, batch, beam_size=4):

        """

        Args:
            image (batch_size, image_size * image_size, hidden_size): Image feature map.
            view_position (batch_size, view_position_size): Patient positions.
            label (batch_size, label_size): Interpretable labels.
            topic (batch_size, embedding_size): Topic embeddings.
            temp (batch_size,): Temperature parameters.

        Returns:

        """


        image         = batch['image']
        view_position = batch['view_position']
        label         = batch['label']
        topic         = batch['_topic']
        temp          = batch['_temp']

        image_mean = image.mean(1)

        batch_size = len(batch['label'])
        batch_index = torch.arange(batch_size, dtype=torch.long).view(-1, 1).expand(-1, beam_size).reshape(-1).cuda()

        v = self.fc_v(self.dropout(image))
        h = torch.tanh(self.fc_h(self.dropout(image_mean)))[batch_index]
        m = torch.tanh(self.fc_m(self.dropout(image_mean)))[batch_index]

        _text = torch.zeros((batch_size * beam_size, 0), dtype=torch.long).cuda()
        _this_text = torch.full((batch_size * beam_size, 1), self.word_to_index[Token.bos], dtype=torch.long).cuda()
        _sum_log_probability = torch.zeros((batch_size * beam_size, 1), dtype=torch.float).cuda()

        outputs = []
        for t in range(self.max_sentence_length + 1):
            logger.debug(f'SentenceDecoder.decode(): time_step={t}, num_sentences={len(batch_index)}')

            batch_length = torch.sum(batch_index.view(-1, 1) == torch.arange(batch_size).view(1, -1).cuda(), 0)
            batch_begin = batch_length.sum() - batch_length.flip(0).cumsum(0).flip(0)

            _text_embedding = self.word_embedding(_this_text.view(-1))
            x = torch.cat([
                self.dropout(image_mean[batch_index]),
                view_position[batch_index],
                label[batch_index],
                self.dropout(topic[batch_index]),
                self.dropout(_text_embedding),
            ], 1)

            (h, m) = self.lstm_cell(x, (h, m))
            (hh, s) = F.relu(self.fc(self.dropout(h))).unsqueeze(1).split(self.fc_sizes, 2)
            _hh = self.fc_hh(self.dropout(hh))
            _s = self.fc_s(self.dropout(s))

            z = torch.tanh(torch.cat([v[batch_index], _s], 1) + _hh)
            z = self.fc_z(self.dropout(z))
            a = F.softmax(z, 1)
            c = torch.sum(a * torch.cat([image[batch_index], s], 1), 1)

            _attention = a.squeeze(2)
            _log_probability = F.log_softmax(self.fc_p(self.dropout(c + hh.squeeze(1))) / temp[batch_index], 1)

            _log_probability = _sum_log_probability + _log_probability
            _log_probability = pad_packed_sequence(_log_probability, batch_length, padding_value=float('-inf'))
            (_sum_log_probability, _top_index) = _log_probability.view(batch_size, -1, 1).topk(beam_size, 1)

            _sum_log_probability = pack_padded_sequence(_sum_log_probability, batch_length)
            _index = pack_padded_sequence(_top_index / self.vocab_size + batch_begin.view(-1, 1, 1), batch_length).squeeze(1)
            _this_text = pack_padded_sequence(_top_index % self.vocab_size, batch_length)

            _this_text[-1] = self.word_to_index[Token.eos]  # DEBUG

            if t == self.max_sentence_length:
                _this_text = torch.full((len(batch_index), 1), self.word_to_index[Token.eos], dtype=torch.long).cuda()

            _text = torch.cat([_text[_index], _this_text], 1)
            is_end = (_this_text == self.word_to_index[Token.eos]).squeeze(1)

            if is_end.any():
                output = {
                    '_index': batch_index[is_end].unsqueeze(1),
                    '_attention': _attention[is_end],
                    '_log_probability': _sum_log_probability[is_end],
                    '_text': _text[is_end],
                }
                outputs.extend([dict(zip(output.keys(), values)) for values in zip(*output.values())])

                batch_index = batch_index[~is_end]
                h = h[~is_end]
                m = m[~is_end]
                _sum_log_probability = _sum_log_probability[~is_end]
                _this_text = _this_text[~is_end]
                _text = _text[~is_end]

            if is_end.all():
                break

        outputs = {key: torch.nn.utils.rnn.pad_sequence([output[key] for output in outputs], batch_first=True) for key in outputs[0].keys()}

        _index = outputs['_index'].squeeze(1).argsort(0)
        outputs = {key: value[_index].view(batch_size, beam_size, -1) for (key, value) in outputs.items()}

        _index = outputs['_log_probability'].argsort(1, descending=True)
        batch_begin = torch.arange(0, batch_size * beam_size, beam_size).view(-1, 1, 1).cuda()
        _index = (_index + batch_begin).view(-1)
        outputs = {key: value.view(batch_size * beam_size, -1)[_index].view(batch_size, beam_size, -1) for (key, value) in outputs.items()}

        return outputs
