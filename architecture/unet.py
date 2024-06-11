import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
import glob
import numpy as np
from .core import SpectralFeatureMapper

resnet = torchvision.models.resnet.resnet50(pretrained=True)

class ConvBlock(nn.Module):
    """
    Helper module that consists of a  Conv-> BN -> ReLU
    """

    def __init__(self, in_channels, out_channels, padding=1, kernel_size=3, stride=1, with_nonlinearity=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, padding=padding, kernel_size=kernel_size, stride=stride)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.with_nonlinearity = with_nonlinearity

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.with_nonlinearity:
            x = self.relu(x)
        return x


class Bridge(nn.Module):
    """
    This is the middle layer of the UNet which just consists of some
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.bridge = nn.Sequential(
            ConvBlock(in_channels, out_channels),
            ConvBlock(out_channels, out_channels)
        )

    def forward(self, x):
        return self.bridge(x)


class UpBlockForUNetWithResNet50(nn.Module):
    """
    Up block that encapsulates one up-sampling step which consists of Upsample -> ConvBlock -> ConvBlock
    """

    def __init__(self, in_channels, out_channels, up_conv_in_channels=None, up_conv_out_channels=None,
                 upsampling_method="conv_transpose"):
        super().__init__()

        if up_conv_in_channels == None:
            up_conv_in_channels = in_channels
        if up_conv_out_channels == None:
            up_conv_out_channels = out_channels

        if upsampling_method == "conv_transpose":
            self.upsample = nn.ConvTranspose2d(up_conv_in_channels, up_conv_out_channels, kernel_size=2, stride=2)
        elif upsampling_method == "bilinear":
            self.upsample = nn.Sequential(
                nn.Upsample(mode='bilinear', scale_factor=2),
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
            )
        self.conv_block_1 = ConvBlock(in_channels, out_channels)
        self.conv_block_2 = ConvBlock(out_channels, out_channels)

    def forward(self, up_x, down_x):
        """

        :param up_x: this is the output from the previous up block
        :param down_x: this is the output from the down block
        :return: upsampled feature map
        """
        x = self.upsample(up_x)
        
        if x.shape != down_x.shape:
            if x.shape[2] > down_x.shape[2]:
                x = x[:, :, :down_x.shape[2], :]
            elif x.shape[2] < down_x.shape[2]:
                down_x = down_x[:, :, :x.shape[2], :]
            if x.shape[3] > down_x.shape[3]:
                x = x[:, :, :, :down_x.shape[3]]
            elif x.shape[3] < down_x.shape[3]:
                down_x = down_x[:, :, :, :x.shape[3]]

    
        x = torch.cat([x, down_x], 1)
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        return x


class UNetWithResnet50Encoder(nn.Module):
    DEPTH = 6

    def __init__(self, n_classes=2, multi_output=False, load_weights='', softmax=False):
        super().__init__()
        resnet = torchvision.models.resnet.resnet50(pretrained=True)
        down_blocks = []
        up_blocks = []
        self.softmax = softmax
        self.input_block = nn.Sequential(*list(resnet.children()))[:3]
        self.input_pool = list(resnet.children())[3]
        for bottleneck in list(resnet.children()):
            if isinstance(bottleneck, nn.Sequential):
                down_blocks.append(bottleneck)
        self.down_blocks = nn.ModuleList(down_blocks)
        self.bridge = Bridge(2048, 2048)
        up_blocks.append(UpBlockForUNetWithResNet50(2048, 1024))
        up_blocks.append(UpBlockForUNetWithResNet50(1024, 512))
        up_blocks.append(UpBlockForUNetWithResNet50(512, 256))
        up_blocks.append(UpBlockForUNetWithResNet50(in_channels=128 + 64, out_channels=128,
                                                    up_conv_in_channels=256, up_conv_out_channels=128))
        up_blocks.append(UpBlockForUNetWithResNet50(in_channels=64 + 3, out_channels=64,
                                                    up_conv_in_channels=128, up_conv_out_channels=64))

        self.up_blocks = nn.ModuleList(up_blocks)

        self.out = nn.Conv2d(64, n_classes, kernel_size=1, stride=1)

        self.multi_output = multi_output
        if self.multi_output:
            self.out1 = nn.Sequential(nn.Conv2d(64, 1, kernel_size=1, stride=1), nn.Sigmoid())
            self.out2 = nn.Sequential(nn.Conv2d(64, 1, kernel_size=1, stride=1), nn.Sigmoid())
            self.out3 = nn.Sequential(nn.Conv2d(64, 1, kernel_size=1, stride=1), nn.Sigmoid())
            self.out4 = nn.Sequential(nn.Conv2d(64, 1, kernel_size=1, stride=1), nn.Sigmoid())

        if load_weights != '':
            self.load_state_dict(torch.load(load_weights))
        self.sfm = SpectralFeatureMapper()

        self.sigmoid = nn.Sigmoid()


    def forward(self, x, with_output_feature_map=False):
        pre_pools = dict()
        pre_pools[f"layer_0"] = x
        x = self.input_block(x)
        pre_pools[f"layer_1"] = x
        x = self.input_pool(x)

        for i, block in enumerate(self.down_blocks, 2):
            x = block(x)
            if i == (UNetWithResnet50Encoder.DEPTH - 1):
                continue
            pre_pools[f"layer_{i}"] = x

        x = self.bridge(x)

        for i, block in enumerate(self.up_blocks, 1):
            key = f"layer_{UNetWithResnet50Encoder.DEPTH - 1 - i}"
            x = block(x, pre_pools[key])
        output_feature_map = x

        if not self.multi_output:
            x = self.out(x)
        else:
            x1 = self.out1(x)
            x2 = self.out2(x)
            x3 = self.out3(x)
            x4 = self.out4(x)
            x = torch.cat([x1, x2, x3, x4], dim=1)

        del pre_pools
        if with_output_feature_map:
            if self.softmax:    
                return F.softmax(x, dim=1), output_feature_map
            return x, output_feature_map
        else:
            xsfm = self.sfm(x)
            return xsfm, x

#model = UNetWithResnet50Encoder(4)
#inp = torch.rand((2, 3, 500, 500))
#out = model(inp)
#print(out.shape)