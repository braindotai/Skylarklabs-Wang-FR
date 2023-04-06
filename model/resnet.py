#!/usr/bin/env python
# encoding: utf-8

import torch
import torch.nn as nn

def ResNet18():
    model = ResNet(BasicBlock, [2, 2, 2, 2])
    return model

def ResNet34():
    model = ResNet(BasicBlock, [3, 4, 6, 3])
    return model

def ResNet50():
    model = ResNet(Bottleneck, [3, 4, 6, 3])
    return model

def ResNet101():
    model = ResNet(Bottleneck, [3, 4, 23, 3])
    return model

def ResNet152():
    model = ResNet(Bottleneck, [3, 8, 36, 3])
    return model

__all__ = ['ResNet', 'ResNet18', 'ResNet34', 'ResNet50', 'ResNet101', 'ResNet152']


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        self.skip_add = nn.quantized.FloatFunctional()
        # Remember to use two independent ReLU for layer fusion.
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        # Use FloatFunctional for addition for quantization compatibility
        # out += identity
        # out = torch.add(identity, out)
        out = self.skip_add.add(identity, out)
        out = self.relu2(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.downsample = downsample
        self.stride = stride
        self.skip_add = nn.quantized.FloatFunctional()
        self.relu3 = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        # out += identity
        # out = torch.add(identity, out)
        out = self.skip_add.add(identity, out)
        out = self.relu3(out)

        return out


class Flatten(nn.Module):
    def forward(self, x):
        # return input.view(input.size(0), -1)
        x = x.reshape(x.size(0), -1)
        return torch.unsqueeze(torch.unsqueeze(x, 2), 3)


class ResNet(nn.Module):

    def __init__(self, block, layers, feature_dim=512, drop_ratio=0.4, zero_init_residual=False):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.output_layer = nn.Sequential(nn.BatchNorm2d(512 * block.expansion),
                                          nn.Dropout(drop_ratio),
                                          Flatten(),
                                          nn.Conv2d(512 * block.expansion * 7 * 7, feature_dim, 1),
                                          nn.BatchNorm2d(feature_dim),
                                          nn.Flatten())

        # self.output_bn2d = nn.BatchNorm2d(512 * block.expansion)
        # self.output_drop = nn.Dropout(drop_ratio)
        # self.output_linear = nn.Linear(512 * block.expansion * 7 * 7, feature_dim)
        # self.output_bn1d = nn.BatchNorm1d(feature_dim)
        #
        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.fc = nn.Linear(512 * block.expansion, feature_dim)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the checkpoints by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.output_layer(x)
        # x = self.output_bn2d(x)
        # x = self.output_drop(x)
        # x = torch.flatten(x, 1)
        # x = self.output_linear(x)
        # x = self.output_bn1d(x)

        return x


if __name__ == "__main__":
    x = torch.Tensor(2, 3, 112, 112)
    net = ResNet50()
    print(net)

    x = net(x)
    print(x.shape)
