import functools
import logging
import os
import re
import torch
import torch.nn.functional as F

from torch.nn import (
    Module as _Module,
    LSTMCell,
    AdaptiveAvgPool2d,
    Conv2d,
    Embedding,
    Linear,
    ReLU,
    Dropout,
)
from torchvision.models.resnet import (
    ResNet,
    Bottleneck,
)
from torchvision.models.densenet import DenseNet

from api import Mode, Phase, Token
from api.utils import (
    expand_to_sequence,
    pack_padded_sequence,
    pad_packed_sequence,
    pad_sequence,
    print_batch,
    profile,
)

logger = logging.getLogger(__name__)


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


class Module(_Module):
    def forward(self, batch, phase=None, **kwargs):
        if phase == Phase.train:
            return self._train(batch, **kwargs)
        elif phase == Phase.val:
            return self._val(batch, **kwargs)
        elif phase == Phase.test:
            return self._test(batch, **kwargs)


class Model(Module):
    def __init__(self, **kwargs):
        super(Model, self).__init__()

        self.mode           = kwargs['mode']
        self.embedding_size = kwargs['embedding_size']
        self.label_size     = kwargs['label_size']
        self.dropout        = kwargs['dropout']

        self.__use_densenet = kwargs['__use_densenet']

        if self.__use_densenet:
            self.image_encoder = DenseNet121(**kwargs)
            path = os.path.join(os.path.dirname(__file__), os.pardir, 'checkpoints', 'model.pkl')
            self.image_encoder.load_state_dict(torch.load(path)['state_dict'])
        else:
            self.image_encoder = ResNet50(**kwargs)

        if self.mode == Mode.debug_label:
            self.fc_label = Linear(self.embedding_size, self.label_size)
            self.drop = Dropout(self.dropout)

        elif self.mode in Mode.label_modes:
            self.report_decoder = ReportDecoder(**kwargs)

        if self.mode in Mode.text_modes:
            self.sentence_decoder = SentenceDecoder(**kwargs)

    @profile
    def forward(self, batch, phase, **kwargs):
        output = batch.copy()

        if phase == Phase.train:
            self.train()

        elif phase in [Phase.val, Phase.test]:
            self.eval()

        output.update(self.image_encoder(output))

        if self.mode == Mode.debug_label:
            output['_label'] = torch.sigmoid(self.fc_label(self.drop(output['image'].mean(1))))

        if self.mode in Mode.label_modes:
            if (phase == Phase.train) and (self.mode in [Mode.auto_regress, Mode.teacher_forcing]):
                output.update(self.report_decoder._train(output, length=output['text_length'], **kwargs))
                _text_length = output['text_length']

            elif (phase == Phase.train) and (self.mode == Mode.self_critical) or (phase in [Phase.val, Phase.test]):
                output.update(self.report_decoder._test(output, **kwargs))
                _text_length = output['_text_length']

            for key in ['image', 'view_position']:
                output[key] = expand_to_sequence(output[key], length=torch.max(_text_length))

            for key in ['image', 'view_position', '_label', '_topic', '_stop', '_temp']:
                output[key] = pack_padded_sequence(output[key], length=_text_length)

            if phase in [Phase.train, Phase.val]:
                for key in ['text', 'label', 'stop', 'sent_length']:
                    output[key] = pack_padded_sequence(output[key], length=output['text_length'])

            if (phase == Phase.train) and (self.mode == Mode.teacher_forcing):
                output.update(self.sentence_decoder._train(output, length=output['sent_length'], **kwargs))

            elif (phase == Phase.train) and (self.mode == Mode.self_critical):
                output.update(self.sentence_decoder._test(output, probabilistic=True, **kwargs))

            elif phase in [Phase.val, Phase.test]:
                output.update(self.sentence_decoder._test(output, probabilistic=False, **kwargs))

        return output


class ResNet50(ResNet):
    image_embedding_size = 2048

    def __init__(self, **kwargs):
        super(ResNet50, self).__init__(Bottleneck, [3, 4, 6, 3])

        self.image_size     = kwargs['image_size']
        self.embedding_size = kwargs['embedding_size']
        self.dropout        = kwargs['dropout']

        self.__image_encoder_relu = kwargs['__image_encoder_relu']

        self.conv1 = Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.avgpool = AdaptiveAvgPool2d((self.image_size, self.image_size))
        self.fc = Conv2d(self.image_embedding_size, self.embedding_size, (1, 1))
        self.drop = Dropout(self.dropout)

    @profile
    def forward(self, batch):
        """

        Args:
            image (batch_size, 1, 256, 256): Grayscale Image.

        Returns:
            image (batch_size, image_size * image_size, image_embedding_size): Image feature map.

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
        image = self.drop(image)
        image = self.fc(image)

        if self.__image_encoder_relu:
            image = self.relu(image)

        image = image.view(-1, self.embedding_size, self.image_size * self.image_size).transpose(1, 2)

        return {'image': image}


class DenseNet121(DenseNet):
    image_embedding_size = 1024

    def __init__(self, **kwargs):
        super(DenseNet121, self).__init__(num_init_features=64, growth_rate=32, block_config=(6, 12, 24, 16))

        self.image_size     = kwargs['image_size']
        self.embedding_size = kwargs['embedding_size']
        self.dropout        = kwargs['dropout']

        self.avgpool = AdaptiveAvgPool2d((self.image_size, self.image_size))
        self.fc = Conv2d(self.image_embedding_size, self.embedding_size, (1, 1))
        self.drop = Dropout(self.dropout)
        self.relu = ReLU(inplace=True)

    def load_state_dict(self, state_dict, strict=False):
        pattern = re.compile(r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')

        _state_dict = {}
        for key in state_dict.keys():
            match = pattern.match(key)
            _key = (match.group(1) + match.group(2)) if match else key

            _state_dict[_key[19:]] = state_dict[key]  # module.densenet121.*

        super(DenseNet121, self).load_state_dict(_state_dict, strict=strict)

    @profile
    def forward(self, batch):
        image = batch['image']

        image = self.features(image)
        image = self.relu(image)

        image = self.avgpool(image)
        image = self.drop(image)
        image = self.fc(image)
        image = self.relu(image)
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

        self.__use_continuous_label = kwargs['__use_continuous_label']
        self.__no_recurrent_label   = kwargs['__no_recurrent_label']

        self.fc_h = Linear(self.embedding_size, self.hidden_size)
        self.fc_m = Linear(self.embedding_size, self.hidden_size)

        self.lstm_sizes = (
            [self.embedding_size] +                                     # image
            [2 * self.view_position_size] +                             # view_position
            [1] +                                                       # begin
            ([] if self.__no_recurrent_label else [self.label_size])    # label
        )
        self.lstm_cell = LSTMCell(sum(self.lstm_sizes), self.hidden_size)
        self.fc_sizes = (
            ([] if self.__no_recurrent_label else [self.label_size]) +  # label
            [self.hidden_size] +                                        # topic
            [1] +                                                       # stop
            [1]                                                         # temp
        )
        self.fc = Linear(self.hidden_size, sum(self.fc_sizes))
        if self.__no_recurrent_label:
            self.fc_label = Linear(self.hidden_size, self.label_size)

        self.drop = Dropout(self.dropout)

    @profile
    def _step(self, batch):
        x = torch.cat((
            [self.drop(batch['image_mean'])] +
            [batch['view_position']] +
            [batch['begin']] +
            ([] if self.__no_recurrent_label else [batch['label']])
        ), 1)

        (h, m) = self.lstm_cell(x, (batch['h'], batch['m']))
        outputs = self.fc(self.drop(h)).split(self.fc_sizes, 1)

        if self.__no_recurrent_label:
            (_topic, _stop, _temp) = outputs
            _label = self.fc_label(self.drop(_topic))
        else:
            (_label, _topic, _stop, _temp) = outputs

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
    def _train(self, batch, length, **kwargs):
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

        teacher_forcing_ratio = kwargs.get('teacher_forcing_ratio', 1.0)

        batch_size = len(image)

        image_mean = image.mean(1)
        h = torch.tanh(self.fc_h(self.drop(image_mean)))
        m = torch.tanh(self.fc_m(self.drop(image_mean)))

        begin = torch.ones((batch_size, 1), dtype=torch.float).cuda()
        this_label = torch.zeros((batch_size, self.label_size), dtype=torch.float).cuda()

        outputs = []
        for t in range(torch.max(length)):
            batch_size_t = torch.sum(length > t).item()

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
                _label if self.__use_continuous_label else (_label > 0.5).float(),
            )

            outputs.append({key: _batch[key] for key in ['_label', '_topic', '_stop', '_temp']})

        outputs = {key: pad_sequence([output[key] for output in outputs]) for key in outputs[0].keys()}

        return outputs

    def _test(self, batch, **kwargs):
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
        h = torch.tanh(self.fc_h(self.drop(image_mean)))[batch_index]
        m = torch.tanh(self.fc_m(self.drop(image_mean)))[batch_index]

        _this_label = torch.zeros((batch_size, self.label_size), dtype=torch.float).cuda()
        _this_topic = torch.zeros((batch_size, self.hidden_size), dtype=torch.float).cuda()

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

            _this_label = _this_label if self.__use_continuous_label else (_this_label > 0.5).float()
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

                (batch_index, h, m, _this_label, _label, _this_topic, _topic, _stop, _temp) = [
                    value[~is_end]
                    for value in [batch_index, h, m, _this_label, _label, _this_topic, _topic, _stop, _temp]
                ]

            if is_end.all():
                break

        outputs = {key: pad_sequence([output[key] for output in outputs], batch_first=True) for key in outputs[0].keys()}

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
        self.word_embedding      = kwargs['word_embedding']
        self.index_to_word       = kwargs['index_to_word']
        self.word_to_index       = kwargs['word_to_index']
        self.label_size          = kwargs['label_size']
        self.embedding_size      = kwargs['embedding_size']
        self.hidden_size         = kwargs['hidden_size']
        self.dropout             = kwargs['dropout']
        self.max_sentence_length = kwargs['max_sentence_length']

        self.__use_continuous_label = kwargs['__use_continuous_label']
        self.__no_recurrent_label   = kwargs['__no_recurrent_label']
        self.__sample_text          = kwargs['__sample_text']
        self.__no_temp              = kwargs['__no_temp']

        self.vocab_size = len(self.word_to_index)

        self.word_embedding = Embedding.from_pretrained(torch.from_numpy(self.word_embedding), freeze=False)
        self.fc_v = Linear(self.embedding_size, self.hidden_size)
        self.fc_h = Linear(self.embedding_size, self.hidden_size)
        self.fc_m = Linear(self.embedding_size, self.hidden_size)
        self.lstm_sizes = (
            [self.embedding_size] +                                     # image
            [2 * self.view_position_size] +                             # view_position
            ([] if self.__no_recurrent_label else [self.label_size]) +  # label
            [self.hidden_size] +                                        # topic
            [self.embedding_size]                                       # text
        )
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
        self.drop = Dropout(self.dropout)

    @profile
    def _step(self, batch):
        text_embedding = self.word_embedding(batch['text'])
        x = torch.cat((
            [self.drop(batch['image_mean'])] +
            [batch['view_position']] +
            ([] if self.__no_recurrent_label else [batch['label']]) +
            [self.drop(batch['topic'])] +
            [self.drop(text_embedding)]
        ), 1)

        (h, m) = self.lstm_cell(x, (batch['h'], batch['m']))
        (hh, s) = F.relu(self.fc(self.drop(h))).unsqueeze(1).split(self.fc_sizes, 2)
        _hh = self.fc_hh(self.drop(hh))
        _s = self.fc_s(self.drop(s))

        z = torch.tanh(torch.cat([batch['v'], _s], 1) + _hh)
        z = self.fc_z(self.drop(z))
        a = F.softmax(z, 1)
        c = torch.sum(a * torch.cat([batch['image'], s], 1), 1)

        _attention = a.squeeze(2)
        _logit = self.fc_p(self.drop(c + hh.squeeze(1)))
        if not self.__no_temp:
            _logit = _logit / batch['temp']
        _log_probability = F.log_softmax(_logit, 1)

        return {
            'h': h,
            'm': m,
            '_attention': _attention,
            '_log_probability': _log_probability,
        }

    @length_sorted_rnn(use_fields=['image', 'view_position', 'text', 'label', '_topic', '_temp'])
    @profile
    def _train(self, batch, length, **kwargs):
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

        teacher_forcing_ratio = kwargs.get('teacher_forcing_ratio', 1.0)

        batch_size = len(image)

        image_mean = image.mean(1)
        v = self.fc_v(self.drop(image))
        h = torch.tanh(self.fc_h(self.drop(image_mean)))
        m = torch.tanh(self.fc_m(self.drop(image_mean)))

        this_text = torch.full((batch_size,), self.word_to_index[Token.bos], dtype=torch.long).cuda()
        _sum_log_probability = torch.zeros((batch_size,), dtype=torch.float).cuda()

        outputs = []
        for t in range(torch.max(length)):
            batch_size_t = torch.sum(length > t).item()

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

            (h, m, _log_probability) = [
                _batch[key]
                for key in ['h', 'm', '_log_probability']
            ]

            categorical = Categorical(logits=_log_probability)

            if self.__sample_text:
                _text = categorical.sample()
            else:
                _text = categorical.argmax()

            this_text = torch.where(
                torch.rand((batch_size_t,), dtype=torch.float).cuda() < teacher_forcing_ratio,
                text[:batch_size_t, t],
                _text,
            )
            _sum_log_probability[:batch_size_t] += categorical.log_prob(_text)

            outputs.append({
                '_attention': _batch['_attention'],
                '_log_probability': _log_probability,
                '_text': _text,
            })

        outputs = {
            key: pad_sequence(
                [output[key] for output in outputs],
                total_length=self.max_sentence_length,
            ) for key in outputs[0].keys()
        }

        outputs['_sum_log_probability'] = _sum_log_probability

        return outputs

    @profile
    def _test(self, batch, probabilistic, **kwargs):
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

        beam_size = kwargs.get('beam_size', 4)
        alpha     = kwargs.get('alpha', 0.65)
        beta      = kwargs.get('beta', 5.0)

        batch_size = len(image)
        batch_index = torch.arange(batch_size, dtype=torch.long).view(-1, 1).expand(-1, beam_size).reshape(-1).cuda()

        image_mean = image.mean(1)
        v = self.fc_v(self.drop(image))
        h = torch.tanh(self.fc_h(self.drop(image_mean)))[batch_index]
        m = torch.tanh(self.fc_m(self.drop(image_mean)))[batch_index]

        _this_text = torch.full((batch_size * beam_size,), self.word_to_index[Token.bos], dtype=torch.long).cuda()

        _text = torch.zeros((batch_size * beam_size, 0), dtype=torch.long).cuda()
        _attention = torch.zeros((batch_size * beam_size, 0, self.image_size * self.image_size + 1), dtype=torch.float).cuda()
        _sum_log_probability = torch.zeros((batch_size * beam_size,), dtype=torch.float).cuda()

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

            (h, m, _this_attention, _log_probability) = [
                _batch[key]
                for key in ['h', 'm', '_attention', '_log_probability']
            ]

            categorical = Categorical(logits=_log_probability)

            if probabilistic:
                _this_text = categorical.sample()
                _sum_log_probability = _sum_log_probability + categorical.log_prob(_this_text)

            else:
                _sum_log_probability = _sum_log_probability.unsqueeze(1) + _log_probability
                _sum_log_probability = pad_packed_sequence(_sum_log_probability, batch_length, padding_value=float('-inf'))
                if t == 0:  # At t = 0, there is only one beam
                    _sum_log_probability = _sum_log_probability.narrow(1, 0, 1)
                (_sum_log_probability, _top_index) = _sum_log_probability.view(batch_size, -1).topk(beam_size, 1)

                _sum_log_probability = pack_padded_sequence(_sum_log_probability, batch_length)
                _index = pack_padded_sequence(_top_index / self.vocab_size + batch_begin.view(-1, 1), batch_length)
                _this_text = pack_padded_sequence(_top_index % self.vocab_size, batch_length)

                _text = _text[_index]
                _attention = _attention[_index]

            if t == self.max_sentence_length - 1:
                _this_text = torch.full((batch_size_t,), self.word_to_index[Token.eos], dtype=torch.long).cuda()

            _text = torch.cat([_text, _this_text.unsqueeze(1)], 1)
            _attention = torch.cat([_attention, _this_attention.unsqueeze(1)], 1)
            _length = torch.full((batch_size_t, 1), t + 1, dtype=torch.long).cuda()

            is_end = (_this_text == self.word_to_index[Token.eos])
            if is_end.any():
                output = {
                    '_index': batch_index[is_end].unsqueeze(1),
                    '_attention': _attention[is_end],
                    '_score': _sum_log_probability[is_end].unsqueeze(1) / ((beta + t) / (beta + 1)) ** alpha,  # https://arxiv.org/pdf/1609.08144.pdf
                    '_sum_log_probability': _sum_log_probability[is_end].unsqueeze(1),
                    '_text': _text[is_end],
                    '_sent_length': _length[is_end],
                }
                outputs.extend([dict(zip(output.keys(), values)) for values in zip(*output.values())])

                (batch_index, h, m, _sum_log_probability, _this_text, _text, _attention) = [
                    value[~is_end]
                    for value in [batch_index, h, m, _sum_log_probability, _this_text, _text, _attention]
                ]

            if is_end.all():
                break

        outputs = {
            key: pad_sequence(
                [output[key] for output in outputs],
                batch_first=True,
                total_length=self.max_sentence_length if key in ['_attention', '_text'] else None,
            ) for key in outputs[0].keys()
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
        outputs['_sum_log_probability'] = outputs['_sum_log_probability'].squeeze(2)
        outputs['_sent_length'] = outputs['_sent_length'].squeeze(2)

        return outputs
