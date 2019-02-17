import logging
import torch
import torch.nn as nn

from api import Mode, Phase
from api.models.base import Module, DataParallelCPU
from api.models.cnn import DenseNet121, ResNet50
from api.models.rnn import ReportDecoder, SentenceDecoder
from api.models.nondiff import SentIndex2Report, CheXpert
from api.utils import profile
from api.utils.rnn import expand_to_sequence, pack_padded_sequence

logger = logging.getLogger(__name__)


class Model(Module):
    def __init__(self, **kwargs):
        super(Model, self).__init__()

        self.mode           = Mode[kwargs['mode']]
        self.embedding_size = kwargs['embedding_size']
        self.label_size     = kwargs['label_size']
        self.dropout        = kwargs['dropout']

        self.__use_densenet = kwargs['__use_densenet']

        if self.__use_densenet:
            self.image_encoder = DenseNet121(**kwargs)
        else:
            self.image_encoder = ResNet50(**kwargs)

        if self.mode & Mode.gen_label_all:
            self.fc_label = nn.Linear(self.embedding_size, self.label_size)
            self.drop = nn.Dropout(self.dropout)

        if self.mode & Mode.gen_label:
            self.report_decoder = ReportDecoder(**kwargs)

        if self.mode & Mode.gen_text:
            self.sentence_decoder = SentenceDecoder(**kwargs)

    @profile
    def forward(self, batch, phase, **kwargs):
        output = batch.copy()

        if phase == Phase.train:
            self.train()

        elif phase in [Phase.val, Phase.test]:
            self.eval()

        output.update(self.image_encoder(output))

        if self.mode & Mode.gen_label_all:
            output['_label'] = torch.sigmoid(self.fc_label(self.drop(output['image'].mean(1))))

        if self.mode & Mode.gen_label:
            if (phase == Phase.train) and (self.mode & Mode.use_self_critical) or (phase in [Phase.val, Phase.test]):
                output.update(self.report_decoder._test(output, **kwargs))
                _text_length = output['_text_length']
            else:
                output.update(self.report_decoder._train(output, length=output['text_length'], **kwargs))
                _text_length = output['text_length']

            for key in ['image', 'view_position']:
                output[key] = expand_to_sequence(output[key], length=torch.max(_text_length))

            for key in ['image', 'view_position', '_label', '_topic', '_stop', '_temp']:
                output[key] = pack_padded_sequence(output[key], length=_text_length)

            if phase in [Phase.train, Phase.val]:
                for key in ['text', 'label', 'stop', 'sent_length']:
                    output[key] = pack_padded_sequence(output[key], length=output['text_length'])

            if (phase == Phase.train) and (self.mode & Mode.use_teacher_forcing):
                output.update(self.sentence_decoder._train(output, length=output['sent_length'], **kwargs))

            elif (phase == Phase.train) and (self.mode & Mode.use_self_critical):
                output.update(self.sentence_decoder._test(output, probabilistic=True, **kwargs))

            elif phase in [Phase.val, Phase.test]:
                output.update(self.sentence_decoder._test(output, probabilistic=False, **kwargs))

        return output