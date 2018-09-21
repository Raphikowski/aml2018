import torch
import torch.nn as nn
import torch.nn.functional as F


class GatingUnit(nn.Module):
    """Gating unit described in 'Convolutional Networks with Adaptive Inference Graphs'."""

    def __init__(self, in_channels, gate_dim):
        super(GatingUnit, self).__init__()

        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.estimate_relevance = nn.Sequential(
            nn.Linear(in_channels, gate_dim),
            nn.ReLU(),
            nn.Linear(gate_dim, 2))

    def forward(self, x):
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.estimate_relevance(x)
        x = F.gumbel_softmax(x, tau=1, hard=True)
        return x


class AdaptiveConv2d(nn.Module):
    """Adaptive Conv2d layer described in 'Convolutional Networks with Adaptive Inference Graphs'."""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, gate_dim=16):
        super(AdaptiveConv2d, self).__init__()

        self.gate = GatingUnit(in_channels, gate_dim)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                              padding, dilation, groups, bias)

    def forward(self, x):
        decision = self.gate(x)
        x = self.conv(x)
        x.transpose_(0, -1).mul_(decision[:, 1]).transpose_(0, -1)
        return x


def conv3x3(in_planes, out_planes, stride=1, adaptive=False):
    """3x3 convolution with padding"""
    if not adaptive:
        return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                         padding=1, bias=False)
    else:
        return AdaptiveConv2d(in_planes, out_planes, kernel_size=3,
                              stride=stride, padding=1, bias=False)


class Block(nn.Module):

    def __init__(self, in_planes, out_planes, stride=1, adaptive=False, batchnorm=False):
        super(Block, self).__init__()

        self.downsample = None

        if batchnorm:
            self.layer1 = nn.Sequential(
                conv3x3(in_planes, out_planes, stride, adaptive),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(out_planes))

            self.layer2 = nn.Sequential(
                conv3x3(out_planes, out_planes),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(out_planes))

            if stride != 1 or in_planes != out_planes:
                self.downsample = nn.Sequential(
                    nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(out_planes))

        else:
            self.layer1 = nn.Sequential(
                conv3x3(in_planes, out_planes, stride, adaptive),
                nn.ReLU(inplace=True))

            self.layer2 = nn.Sequential(
                conv3x3(out_planes, out_planes),
                nn.ReLU(inplace=True))

            if stride != 1 or in_planes != out_planes:
                self.downsample = nn.Conv2d(in_planes, out_planes,
                    kernel_size=1, stride=stride, bias=False)

    def forward(self, x):
        residual = x

        out = self.layer1(x)
        out = self.layer2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        return out


class ResNet(nn.Module):

    def __init__(self, num_classes=10, adaptive=False, batchnorm=False):
        super(ResNet, self).__init__()

        self.block0 = Block(1, 64, stride=2, adaptive=adaptive, batchnorm=batchnorm)
        self.block1 = Block(64, 128, stride=2, adaptive=adaptive, batchnorm=batchnorm)
        self.block2 = Block(128, 256, stride=2, adaptive=adaptive, batchnorm=batchnorm)
        self.block3 = Block(256, 256, stride=2, adaptive=adaptive, batchnorm=batchnorm)
        self.block4 = Block(256, 256, stride=1, adaptive=adaptive, batchnorm=batchnorm)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(256, num_classes)

        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.block0(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x
