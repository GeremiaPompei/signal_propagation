from typing import Callable

import torch
import torch.nn as nn
from torch.nn.functional import relu, avg_pool2d


def conv3x3(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """
    Function able to create a custom convolutional layer with kernel 3x3.
    @param in_planes: Input features.
    @param out_planes: Output features.
    @param stride: Stride.
    @return: Initialized layer.
    """
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False,
    )


class BasicBlock(nn.Module):
    """
    Basic block of ResNet.
    """
    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1):
        """
        ResNet basic block constructor.
        @param in_planes: Input features.
        @param planes: Hidden features.
        @param stride: Stride of first convolutional layer.
        """
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Pytorch forward method.
        @param x: Input to predict.
        @return: Prediction of the input.
        """
        out = relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = relu(out)
        return out


class ResNet(nn.Module):
    """
    ResNet model taken by avalanche library.
    """

    def __init__(self, in_chans: int, block: Callable, num_blocks: list[int], num_classes: int, nf: int):
        """
        ResNet constructor.
        @param block: Function able to construct a ResNet block.
        @param num_blocks: Number of ResNet blocks for each layer.
        @param num_classes: Number of output classes.
        @param nf: Number of features in input of layers.
        """
        super(ResNet, self).__init__()
        self.in_planes = nf
        self.conv1 = conv3x3(in_chans, nf * 1)
        self.bn1 = nn.BatchNorm2d(nf * 1)
        self.layer1 = self._make_layer(block, nf * 1, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, nf * 2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, nf * 4, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, nf * 8, num_blocks[3], stride=2)
        self.avg_pool2d = nn.AvgPool2d(4)
        self.linear = nn.Linear(nf * 8 * block.expansion, num_classes)

    def _make_layer(self, block: Callable, planes: int, num_blocks: int, stride: int) -> nn.Sequential:
        """
        Method able to create a layer of ResNet.
        @param block: Function able to create a block.
        @param planes: Number of hidden features of the block.
        @param num_blocks: Number of blocks.
        @param stride: Stride applied to the first sublayer of the current layer.
        @return: Sequential of sublayers of the current layer.
        """
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def return_hidden(self, x: torch.Tensor) -> torch.Tensor:
        """
        Method able to return the latent representations of some input data.
        @param x: Input data.
        @return: Latent representations.
        """
        bsz = x.size(0)
        out = relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avg_pool2d(out)
        out = out.view(out.size(0), -1)
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Pytorch forward method.
        @param x: Input data.
        @return: Prediction data.
        """
        out = self.return_hidden(x)
        out = self.linear(out)
        return out


def ResNet18(n_classes, in_chans: int = 3):
    """
    Function able to create a ResNet18 model.
    @param n_classes: Number of classes in output layer.
    @return: ResNet18 initialized model.
    """
    return ResNet(in_chans, BasicBlock, [2, 2, 2, 2], n_classes, 20)
