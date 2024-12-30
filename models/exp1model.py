import torch
import torch.nn as nn
import torch.nn.functional as F

class DepthwiseSeparableConv2dBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernels_per_layer=1, kernel_size=3, padding=0, stride=1, drop_rate=0.05):
        super(DepthwiseSeparableConv2dBlock, self).__init__()
        self.depthwise = nn.Conv2d(in_channels,
                                   in_channels * kernels_per_layer,
                                   kernel_size=kernel_size,
                                   padding=padding,
                                   stride=stride,
                                   groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels * kernels_per_layer,
                                   out_channels,
                                   kernel_size=1,
                                   padding=padding,
                                   stride=stride)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(drop_rate)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        out = self.dropout(self.bn(self.act(out)))
        return out


class Conv2dBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=0, stride=1, drop_rate=0.05, dilation=1):
        super(Conv2dBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels,
                      out_channels,
                      kernel_size=kernel_size,
                      padding=padding,
                      stride=stride,
                      dilation=dilation,
                      bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(drop_rate),
        )

    def forward(self, x):
        return self.conv(x)


class TransitionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, down_sample=True):
        super().__init__()

        if down_sample:
            self.transition = nn.Sequential(
                # Reducing channels
                nn.Conv2d(in_channels,
                        out_channels,
                        kernel_size=1,
                        bias=False),

                # Replace MaxPool2d with Conv2d
                nn.Conv2d(out_channels,
                        out_channels,
                        kernel_size=2,    # kernel_size,
                        stride=2,         # stride,
                        bias=False),
            )
        else:
            self.transition = nn.Sequential(
                # Reducing channels
                nn.Conv2d(in_channels,
                        out_channels,
                        kernel_size=1,
                        bias=False),
                        )

    def forward(self, x):
        return self.transition(x)


class CifarNet(nn.Module):
    def __init__(self):
        super(CifarNet, self).__init__()
        
        # Initial convolution to increase channels
        self.prep = Conv2dBlock(in_channels=3, out_channels=32, kernel_size=3, padding=1, drop_rate=0.0)
        
        # First dense block
        self.block1 = nn.Sequential(
            DepthwiseSeparableConv2dBlock(in_channels=32, out_channels=64, padding=1, drop_rate=0.1),
            Conv2dBlock(in_channels=64, out_channels=64, kernel_size=3, padding=1, drop_rate=0.1),
            TransitionBlock(in_channels=64, out_channels=32, down_sample=True)  # 32x32 -> 16x16
        )
        
        # Second dense block
        self.block2 = nn.Sequential(
            DepthwiseSeparableConv2dBlock(in_channels=32, out_channels=96, padding=1, drop_rate=0.1),
            DepthwiseSeparableConv2dBlock(in_channels=96, out_channels=96, padding=1, drop_rate=0.1),
            TransitionBlock(in_channels=96, out_channels=48, down_sample=True)  # 16x16 -> 8x8
        )
        
        # Third dense block with dilated convolutions
        self.block3 = nn.Sequential(
            Conv2dBlock(in_channels=48, out_channels=128, kernel_size=3, padding=2, dilation=2, drop_rate=0.1),
            DepthwiseSeparableConv2dBlock(in_channels=128, out_channels=128, padding=1, drop_rate=0.1),
            TransitionBlock(in_channels=128, out_channels=64, down_sample=True)  # 8x8 -> 4x4
        )
        
        # Global average pooling and classifier
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Conv2d(64, 10, kernel_size=1),
            nn.BatchNorm2d(10)
        )

    def forward(self, x):
        x = self.prep(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.global_pool(x)
        x = self.classifier(x)
        x = x.view(x.size(0), -1)
        return F.log_softmax(x, dim=1)

