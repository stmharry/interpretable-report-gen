import functools
import logging
import torch
import torch.nn.functional as F

from torch.nn import (
    Module as _Module,
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

from api import Phase, Token
from api.utils import (
    expand_to_sequence,
    pack_padded_sequence,
    pad_packed_sequence,
    pad_sequence,
    print_batch,
)

logger = logging.getLogger(__name__)

if 'profile' not in dir(__builtins__):
    def profile(func):
        return func


def length_sorted_rnn(use_fields=None):
    use_fields = use_fields or []

    def _length_sorted_rnn(func):
        def _func(self, batch, length, **kwargs):
            index = length.argsort(descending=True)
            _index = index.argsort()

            batch = func(self, {key: batch[key][index] for key in use_fields}, length[index], **kwargs)
            return {key: batch[key][_index] for key in batch.keys()}

        return _func

    return _length_sorted_rnn


class Module(_Module):
    def forward(self, batch, phase=None, **kwargs):
        if phase == Phase.train:
            return self._train(batch, **kwargs)
        elif phase == Phase.val:
            return self._val(batch, **kwargs)
        elif phase == Phase.test:
            return self._test(batch, **kwargs)


class Model(Module):
    report_keys = ['image', 'view_position']
    sentence_keys = ['image', 'view_position', 'text', 'label', 'stop', 'sent_length', '_label', '_topic', '_stop', '_temp']
    word_keys = ['text', '_attention', '_log_probability', '_score', '_text']

    def __init__(self, **kwargs):
        super(Model, self).__init__()

        self.mode = kwargs['mode']

        self.image_encoder = ImageEncoder(**kwargs)
        self.report_decoder = ReportDecoder(**kwargs)
        self.sentence_decoder = SentenceDecoder(**kwargs)

    def _map(self, batch, func, keys):
        _batch = {}
        for key in keys:
            if key in batch:
                _batch[key] = func(batch[key])

        return _batch

    def _train(self, batch, teacher_forcing_ratio):
        batch.update(self.image_encoder(batch))
        batch.update(self.report_decoder._train(batch, length=batch['text_length'], teacher_forcing_ratio=teacher_forcing_ratio))

        for key in ['image', 'view_position']:
            batch[key] = expand_to_sequence(batch[key], length=torch.max(batch['text_length']))

        for key in ['image', 'view_position', 'text', 'label', 'stop', 'sent_length', '_label', '_topic', '_stop', '_temp']:
            batch[key] = pack_padded_sequence(batch[key], length=batch['text_length'])

        if self.mode == 'full':
            batch.update(self.sentence_decoder._train(batch, length=batch['sent_length'], teacher_forcing_ratio=teacher_forcing_ratio))

        return batch

    def _val(self, batch, beam_size, alpha, beta, is_val=True):
        batch.update(self.image_encoder(batch))
        batch.update(self.report_decoder._test(batch))

        for key in ['image', 'view_position']:
            batch[key] = expand_to_sequence(batch[key], length=torch.max(batch['_text_length']))

        for key in ['image', 'view_position', '_label', '_topic', '_stop', '_temp']:
            batch[key] = pack_padded_sequence(batch[key], length=batch['_text_length'])

        if is_val:
            for key in ['text', 'label', 'stop', 'sent_length']:
                batch[key] = pack_padded_sequence(batch[key], length=batch['text_length'])

        if self.mode == 'full':
            batch.update(self.sentence_decoder._test(batch, beam_size=beam_size, alpha=alpha, beta=beta))

        return batch

    _test = functools.partialmethod(_val, is_val=False)


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
        self.max_report_length  = kwargs['max_report_length']

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
            self.embedding_size,          # image
            2 * self.view_position_size,  # view_position
            1,                            # begin
            self.label_size,              # label
        ]
        self.lstm_cell = LSTMCell(sum(self.lstm_sizes), self.hidden_size)
        self.dropout = Dropout(self.dropout)

    def _step(self, batch):
        x = torch.cat([
            self.dropout(batch['image_mean']),
            batch['view_position'],
            batch['begin'],
            batch['label'],
        ], 1)

        (h, m) = self.lstm_cell(x, (batch['h'], batch['m']))
        (_label, _topic, _stop, _temp) = self.fc(self.dropout(h)).split(self.fc_sizes, 1)

        return {
            'h': h,
            'm': m,
            '_label': torch.sigmoid(_label),
            '_topic': F.relu(_topic),
            '_stop': torch.sigmoid(_stop),
            '_temp': torch.exp(_temp),
        }

    @length_sorted_rnn(use_fields=['image', 'view_position', 'label'])
    @profile
    def _train(self, batch, length, teacher_forcing_ratio=1.0):
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

        batch_size = len(image)

        image_mean = image.mean(1)
        h = torch.tanh(self.fc_h(self.dropout(image_mean)))
        m = torch.tanh(self.fc_m(self.dropout(image_mean)))

        begin = torch.ones((batch_size, 1), dtype=torch.float).cuda()
        this_label = torch.zeros((batch_size, self.label_size), dtype=torch.float).cuda()

        outputs = []
        for t in range(torch.max(length)):
            batch_size_t = torch.sum(length > t)

            logger.debug(f'ReportDecoder.forward(): time_step={t}, num_reports={batch_size_t}')

            _batch = self._step({
                'image_mean': image_mean[:batch_size_t],
                'view_position': view_position[:batch_size_t],
                'begin': begin[:batch_size_t],
                'label': this_label[:batch_size_t],
                'h': h[:batch_size_t],
                'm': m[:batch_size_t],
            })

            (h, m, _label) = [
                _batch[key]
                for key in ['h', 'm', '_label']
            ]

            begin = torch.zeros((batch_size_t, 1), dtype=torch.float).cuda()
            this_label = torch.where(
                torch.rand((batch_size_t, 1), dtype=torch.float).cuda() < teacher_forcing_ratio,
                label[:batch_size_t, t],
                (_label > 0.5).float(),
            )

            outputs.append({key: _batch[key] for key in ['_label', '_topic', '_stop', '_temp']})

        outputs = {key: torch.nn.utils.rnn.pad_sequence([output[key] for output in outputs]) for key in outputs[0].keys()}

        return outputs

    @profile
    def _test(self, batch):
        """

        Args:
            image (batch_size, image_size * image_size, hidden_size): Image feature map.
            view_position (batch_size, view_position_size): Patient positions.

        Returns:
            _label (batch_size, max_length, label_size): Generated labels.
            _topic (batch_size, max_length, embedding_size): Generated topic embeddings.
            _stop (batch_size, max_length, 1): Stop signal.
            _temp (batch_size, max_length, 1): Temperatures.

        """

        image         = batch['image']
        view_position = batch['view_position']

        batch_size = len(image)
        batch_index = torch.arange(batch_size, dtype=torch.long).cuda()

        image_mean = image.mean(1)
        h = torch.tanh(self.fc_h(self.dropout(image_mean)))[batch_index]
        m = torch.tanh(self.fc_m(self.dropout(image_mean)))[batch_index]

        _this_label = torch.zeros((batch_size, self.label_size), dtype=torch.float).cuda()

        _label = torch.zeros((batch_size, 0, self.label_size), dtype=torch.float).cuda()
        _topic = torch.zeros((batch_size, 0, self.hidden_size), dtype=torch.float).cuda()
        _stop = torch.zeros((batch_size, 0, 1), dtype=torch.float).cuda()
        _temp = torch.zeros((batch_size, 0, 1), dtype=torch.float).cuda()

        outputs = []
        for t in range(self.max_report_length):
            batch_size_t = len(batch_index)
            batch_length = torch.sum(batch_index.view(-1, 1) == torch.arange(batch_size).view(1, -1).cuda(), 0)
            batch_begin = batch_length.sum() - batch_length.flip(0).cumsum(0).flip(0)

            logger.debug(f'ReportDecoder.decode(): time_step={t}, num_reports={batch_size_t}')

            _batch = self._step({
                'image_mean': image_mean[batch_index],
                'view_position': view_position[batch_index],
                'begin': torch.full((batch_size_t, 1), t == 0, dtype=torch.float).cuda(),
                'label': _this_label,
                'h': h,
                'm': m,
            })

            (h, m, _this_label, _this_topic, _this_stop, _this_temp) = [
                _batch[key]
                for key in ['h', 'm', '_label', '_topic', '_stop', '_temp']
            ]

            _this_label = (_this_label > 0.5).float()
            if t == self.max_report_length - 1:
                _this_stop = torch.full((batch_size_t, 1), 1.0, dtype=torch.float).cuda()

            _label = torch.cat([_label, _this_label.unsqueeze(1)], 1)
            _topic = torch.cat([_topic, _this_topic.unsqueeze(1)], 1)
            _stop = torch.cat([_stop, _this_stop.unsqueeze(1)], 1)
            _temp = torch.cat([_temp, _this_temp.unsqueeze(1)], 1)
            _length = torch.full((batch_size_t, 1), t + 1, dtype=torch.long).cuda()

            is_end = (_this_stop > 0.5).squeeze(1)
            if is_end.any():
                output = {
                    '_index': batch_index[is_end].unsqueeze(1),
                    '_label': _label[is_end],
                    '_topic': _topic[is_end],
                    '_stop': _stop[is_end],
                    '_temp': _temp[is_end],
                    '_text_length': _length[is_end],
                }
                outputs.extend([dict(zip(output.keys(), values)) for values in zip(*output.values())])

                (batch_index, h, m, _this_label, _label, _topic, _stop, _temp) = [
                    value[~is_end]
                    for value in [batch_index, h, m, _this_label, _label, _topic, _stop, _temp]
                ]

            if is_end.all():
                break

        outputs = {key: torch.nn.utils.rnn.pad_sequence([output[key] for output in outputs], batch_first=True) for key in outputs[0].keys()}

        # Index the beams
        _index = outputs['_index'].squeeze(1).argsort(0)
        outputs = {key: value[_index] for (key, value) in outputs.items()}

        outputs.pop('_index')
        outputs['_text_length'] = outputs['_text_length'].squeeze(1)

        return outputs


class SentenceDecoder(Module):
    def __init__(self, **kwargs):
        super(SentenceDecoder, self).__init__()

        self.image_size          = kwargs['image_size']
        self.view_position_size  = kwargs['view_position_size']
        self.index_to_word       = kwargs['index_to_word']
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
            self.embedding_size,          # image
            2 * self.view_position_size,  # view_position
            self.label_size,              # label
            self.hidden_size,             # topic
            self.embedding_size,          # text
        ]
        self.lstm_cell = LSTMCell(sum(self.lstm_sizes), self.hidden_size)
        self.fc_sizes = [
            self.embedding_size,  # hidden
            self.embedding_size,  # sentinel
        ]
        self.fc = Linear(self.hidden_size, sum(self.fc_sizes))
        self.fc_hh = Linear(self.embedding_size, self.hidden_size)
        self.fc_s = Linear(self.embedding_size, self.hidden_size)
        self.fc_z = Linear(self.hidden_size, 1)
        self.fc_p = Linear(self.embedding_size, self.vocab_size)
        self.dropout = Dropout(self.dropout)

    def _step(self, batch):
        text_embedding = self.word_embedding(batch['text'])
        x = torch.cat([
            self.dropout(batch['image_mean']),
            batch['view_position'],
            batch['label'],
            self.dropout(batch['topic']),
            self.dropout(text_embedding),
        ], 1)

        (h, m) = self.lstm_cell(x, (batch['h'], batch['m']))
        (hh, s) = F.relu(self.fc(self.dropout(h))).unsqueeze(1).split(self.fc_sizes, 2)
        _hh = self.fc_hh(self.dropout(hh))
        _s = self.fc_s(self.dropout(s))

        z = torch.tanh(torch.cat([batch['v'], _s], 1) + _hh)
        z = self.fc_z(self.dropout(z))
        a = F.softmax(z, 1)
        c = torch.sum(a * torch.cat([batch['image'], s], 1), 1)

        _attention = a.squeeze(2)
        _log_probability = F.log_softmax(self.fc_p(self.dropout(c + hh.squeeze(1))) / batch['temp'], 1)
        _text = _log_probability.argmax(1)

        return {
            'h': h,
            'm': m,
            '_attention': _attention,
            '_log_probability': _log_probability,
            '_text': _text,
        }


    @length_sorted_rnn(use_fields=['image', 'view_position', 'text', 'label', '_topic', '_temp'])
    @profile
    def _train(self, batch, length, teacher_forcing_ratio=1.0):
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

        batch_size = len(image)

        image_mean = image.mean(1)
        v = self.fc_v(self.dropout(image))
        h = torch.tanh(self.fc_h(self.dropout(image_mean)))
        m = torch.tanh(self.fc_m(self.dropout(image_mean)))

        this_text = torch.full((batch_size,), self.word_to_index[Token.bos], dtype=torch.long).cuda()

        outputs = []
        for t in range(torch.max(length)):
            batch_size_t = torch.sum(length > t)

            logger.debug(f'SentenceDecoder.forward(): time_step={t}, num_sentences={batch_size_t}')

            _batch = self._step({
                'image_mean': image_mean[:batch_size_t],
                'view_position': view_position[:batch_size_t],
                'label': label[:batch_size_t],
                'topic': topic[:batch_size_t],
                'temp': temp[:batch_size_t],
                'text': this_text[:batch_size_t],
                'image': image[:batch_size_t],
                'v': v[:batch_size_t],
                'h': h[:batch_size_t],
                'm': m[:batch_size_t],
            })

            (h, m, _text) = [
                _batch[key]
                for key in ['h', 'm', '_text']
            ]

            this_text = torch.where(
                torch.rand((batch_size_t,), dtype=torch.float).cuda() < teacher_forcing_ratio,
                text[:batch_size_t, t],
                _text,
            )

            outputs.append({key: _batch[key] for key in ['_attention', '_log_probability', '_text']})

        outputs.extend([{
            '_attention': torch.zeros((0, self.image_size * self.image_size + 1), dtype=torch.float).cuda(),
            '_log_probability': torch.zeros((0, self.vocab_size), dtype=torch.float).cuda(),
            '_text': torch.zeros((0,), dtype=torch.float).cuda(),
        }] * (self.max_sentence_length - int(torch.max(length))))

        outputs = {key: torch.nn.utils.rnn.pad_sequence([output[key] for output in outputs]) for key in outputs[0].keys()}

        return outputs

    @profile
    def _test(self, batch, beam_size=4, alpha=0.65, beta=5.0):
        """

        Args:
            image (batch_size, image_size * image_size, hidden_size): Image feature map.
            view_position (batch_size, view_position_size): Patient positions.
            label (batch_size, label_size): Interpretable labels.
            topic (batch_size, embedding_size): Topic embeddings.
            temp (batch_size,): Temperature parameters.

        Returns:
            _attention (batch_size, max_length, image_size * image_size + 1): Attention weights.
            _log_probability (batch_size, max_length, vocab_size): Generated probability on words.
            _text (batch_size, max_length): Generated sentences.

        """

        image         = batch['image']
        view_position = batch['view_position']
        label         = batch['_label']
        topic         = batch['_topic']
        temp          = batch['_temp']

        batch_size = len(image)
        batch_index = torch.arange(batch_size, dtype=torch.long).view(-1, 1).expand(-1, beam_size).reshape(-1).cuda()

        image_mean = image.mean(1)
        v = self.fc_v(self.dropout(image))
        h = torch.tanh(self.fc_h(self.dropout(image_mean)))[batch_index]
        m = torch.tanh(self.fc_m(self.dropout(image_mean)))[batch_index]

        _this_text = torch.full((batch_size * beam_size,), self.word_to_index[Token.bos], dtype=torch.long).cuda()

        _text = torch.zeros((batch_size * beam_size, 0), dtype=torch.long).cuda()
        _attention = torch.zeros((batch_size * beam_size, 0, self.image_size * self.image_size + 1), dtype=torch.float).cuda()
        _log_probability = torch.zeros((batch_size * beam_size,), dtype=torch.float).cuda()

        outputs = []
        for t in range(self.max_sentence_length):
            batch_size_t = len(batch_index)
            batch_length = torch.sum(batch_index.view(-1, 1) == torch.arange(batch_size).view(1, -1).cuda(), 0)
            batch_begin = batch_length.sum() - batch_length.flip(0).cumsum(0).flip(0)

            logger.debug(f'SentenceDecoder.decode(): time_step={t}, num_sentences={batch_size_t}')

            _batch = self._step({
                'image_mean': image_mean[batch_index],
                'view_position': view_position[batch_index],
                'label': label[batch_index],
                'topic': topic[batch_index],
                'temp': temp[batch_index],
                'text': _this_text,
                'image': image[batch_index],
                'v': v[batch_index],
                'h': h,
                'm': m,
            })

            (h, m, _this_attention, _this_log_probability) = [
                _batch[key]
                for key in ['h', 'm', '_attention', '_log_probability']
            ]

            _log_probability = _log_probability.unsqueeze(1) + _this_log_probability
            _log_probability = pad_packed_sequence(_log_probability, batch_length, padding_value=float('-inf'))
            if t == 0:  # At t = 0, there is only one beam
                _log_probability = _log_probability.narrow(1, 0, 1)
            (_log_probability, _top_index) = _log_probability.view(batch_size, -1).topk(beam_size, 1)

            _log_probability = pack_padded_sequence(_log_probability, batch_length)
            _index = pack_padded_sequence(_top_index / self.vocab_size + batch_begin.view(-1, 1), batch_length)
            _this_text = pack_padded_sequence(_top_index % self.vocab_size, batch_length)

            if t == self.max_sentence_length - 1:
                _this_text = torch.full((batch_size_t,), self.word_to_index[Token.eos], dtype=torch.long).cuda()

            _text = torch.cat([_text[_index], _this_text.unsqueeze(1)], 1)
            _attention = torch.cat([_attention[_index], _this_attention.unsqueeze(1)], 1)
            _length = torch.full((batch_size_t, 1), t + 1, dtype=torch.long).cuda()

            is_end = (_this_text == self.word_to_index[Token.eos])
            if is_end.any():
                output = {
                    '_index': batch_index[is_end].unsqueeze(1),
                    '_attention': _attention[is_end],
                    '_score': _log_probability[is_end].unsqueeze(1) / ((beta + t) / (beta + 1)) ** alpha,  # https://arxiv.org/pdf/1609.08144.pdf
                    '_log_probability': _log_probability[is_end].unsqueeze(1),
                    '_text': _text[is_end],
                    '_sent_length': _length[is_end],
                }
                outputs.extend([dict(zip(output.keys(), values)) for values in zip(*output.values())])

                (batch_index, h, m, _log_probability, _this_text, _text) = [
                    value[~is_end]
                    for value in [batch_index, h, m, _log_probability, _this_text, _text]
                ]

            if is_end.all():
                break

        outputs = {
            key: pad_sequence(
                [output[key] for output in outputs],
                batch_first=True,
                total_length=self.max_sentence_length if key in ['_attention', '_text'] else None,
            )
            for key in outputs[0].keys()
        }

        # Index the beams
        _index = outputs['_index'].squeeze(1).argsort(0)
        outputs = {key: value[_index] for (key, value) in outputs.items()}

        # Sort beams internally by scores
        _row = torch.arange(0, batch_size * beam_size, beam_size).view(-1, 1).cuda()
        _col = outputs['_score'].view(batch_size, beam_size).argsort(1, descending=True)
        outputs = {key: value[_row + _col] for (key, value) in outputs.items()}

        outputs.pop('_index')
        outputs['_score'] = outputs['_score'].squeeze(2)
        outputs['_log_probability'] = outputs['_log_probability'].squeeze(2)
        outputs['_sent_length'] = outputs['_sent_length'].squeeze(2)

        return outputs
