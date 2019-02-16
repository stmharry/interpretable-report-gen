import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
import re

from collections import OrderedDict
from torchvision.models.resnet import ResNet, Bottleneck

from api.utils import profile


class _DenseLayer(nn.Module):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate, efficient=False):
        super(_DenseLayer, self).__init__()

        self.add_module('norm1', nn.BatchNorm2d(num_input_features))
        self.add_module('relu1', nn.ReLU(inplace=True))
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size * growth_rate, kernel_size=1, stride=1, bias=False))
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate))
        self.add_module('relu2', nn.ReLU(inplace=True))
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate, kernel_size=3, stride=1, padding=1, bias=False))

        self.drop_rate = drop_rate
        self.efficient = efficient

    def _conv1_fn(self, *prev_features):
        concated_features = torch.cat(prev_features, 1)
        bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))
        return bottleneck_output

    def forward(self, *prev_features):
        if self.efficient and self.training:
            bottleneck_output = cp.checkpoint(self._conv1_fn, *prev_features)
        else:
            bottleneck_output = self._conv1_fn(*prev_features)

        new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))

        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)

        return new_features


class _DenseBlock(nn.Module):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate, efficient=False):
        super(_DenseBlock, self).__init__()

        for i in range(num_layers):
            layer = _DenseLayer(
                num_input_features + i * growth_rate,
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate,
                efficient=efficient,
            )
            self.add_module('denselayer%d' % (i + 1), layer)

    def forward(self, init_features):
        features = [init_features]
        for name, layer in self.named_children():
            new_features = layer(*features)
            features.append(new_features)

        return torch.cat(features, 1)


class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()

        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features, kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))


class DenseNet(nn.Module):
    def __init__(self,
                 growth_rate=32,
                 block_config=(6, 12, 24, 16),
                 num_init_features=64,
                 compression=0.5,
                 bn_size=4,
                 drop_rate=0,
                 efficient=False):

        super(DenseNet, self).__init__()

        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(3, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))

        num_features = num_init_features
        for (i, num_layers) in enumerate(block_config):
            block = _DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=drop_rate,
                efficient=efficient,
            )
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(
                    num_input_features=num_features,
                    num_output_features=int(num_features * compression),
                )
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = int(num_features * compression)

        self.features.add_module('norm5', nn.BatchNorm2d(num_features))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)


class DenseNet121(DenseNet):
    image_embedding_size = 1024

    def __init__(self, **kwargs):
        super(DenseNet121, self).__init__(num_init_features=64, growth_rate=32, block_config=(6, 12, 24, 16))

        self.image_size     = kwargs['image_size']
        self.embedding_size = kwargs['embedding_size']
        self.dropout        = kwargs['dropout']

        self.avgpool = nn.AdaptiveAvgPool2d((self.image_size, self.image_size))
        self.fc = nn.Conv2d(self.image_embedding_size, self.embedding_size, (1, 1))
        self.drop = nn.Dropout(self.dropout)
        self.relu = nn.ReLU(inplace=True)

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


class ResNet50(ResNet):
    image_embedding_size = 2048

    def __init__(self, **kwargs):
        super(ResNet50, self).__init__(Bottleneck, [3, 4, 6, 3])

        self.image_size     = kwargs['image_size']
        self.embedding_size = kwargs['embedding_size']
        self.dropout        = kwargs['dropout']

        self.__image_encoder_relu = kwargs['__image_encoder_relu']

        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.avgpool = nn.AdaptiveAvgPool2d((self.image_size, self.image_size))
        self.fc = nn.Conv2d(self.image_embedding_size, self.embedding_size, (1, 1))
        self.drop = nn.Dropout(self.dropout)

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
