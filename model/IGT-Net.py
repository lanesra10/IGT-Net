# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging

import torch
import torch.nn as nn


from model import seg_configs as configs
from thop import clever_format, profile
from model.Nat import NATLayer
from model.inceptionnext import inceptionnext_base_384
from timm.models.layers import DropPath

logger = logging.getLogger(__name__)


class Embeddings(nn.Module):
    """Construct the embeddings from patch, position embeddings.
    """

    def __init__(self, config, img_size, in_channels=3):
        super(Embeddings, self).__init__()
        self.hybrid = None
        self.config = config

        if config.patches.get("grid") is not None:  # ResNet
            self.hybrid = True
        else:
            self.hybrid = False

        if self.hybrid:
            self.hybrid_model = inceptionnext_base_384()
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, x):
        if self.hybrid:
            x, features = self.hybrid_model(x)
        else:
            features = None
        x = self.up(x)
        return x, features


class SVGA(nn.Module):
    def __init__(self, config, img_size, vis):
        super(SVGA, self).__init__()
        self.embeddings = Embeddings(config, img_size=img_size)
        self.grapherblock = nn.ModuleList([])
        dpr = [x.item() for x in torch.linspace(0, 0.1, 30)]
        for i in range(1):
            dpr_idx = 25
            for j in range(5):
                self.grapherblock += [nn.Sequential(
                    Grapher(1024, drop_path=dpr[dpr_idx], K=2),
                    FFN(1024, 1024 * 4, drop_path=dpr[dpr_idx]))
                ]
                dpr_idx += 1

    def forward(self, input_ids):
        x, features = self.embeddings(input_ids)
        for i in range(len(self.grapherblock)):
            x = self.grapherblock[i](x)
        return x, features


class Conv2dReLU(nn.Sequential):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            stride=1,
            use_batchnorm=True,
    ):
        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=not (use_batchnorm),
        )
        relu = nn.ReLU(inplace=True)

        bn = nn.BatchNorm2d(out_channels)

        super(Conv2dReLU, self).__init__(conv, bn, relu)


class DecoderBlock1(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            skip_channels=0,
            use_batchnorm=True,
    ):
        super().__init__()
        self.conv1 = Conv2dReLU(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.conv2 = Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        self.att = NATLayer((16, 16), 512, 4)

    def forward(self, x, skip):
        x = torch.cat([x, self.att(skip)], dim=1)
        x = self.up(x)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class DecoderBlock2(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            skip_channels=0,
            use_batchnorm=True,
    ):
        super().__init__()
        self.conv1 = Conv2dReLU(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.conv2 = Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)

        self.att = NATLayer((32, 32), 256, 4)

    def forward(self, x, skip):
        x = torch.cat([x, self.att(skip)], dim=1)
        x = self.up(x)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class DecoderBlock3(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            skip_channels=0,
            use_batchnorm=True,
    ):
        super().__init__()
        self.conv1 = Conv2dReLU(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.conv2 = Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        self.att = NATLayer((64, 64), 128, 4)

    def forward(self, x, skip):
        x = torch.cat([x, self.att(skip)], dim=1)
        x = self.up(x)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class DecoderBlock4(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            skip_channels=0,
            use_batchnorm=True,
    ):
        super().__init__()
        self.conv1 = Conv2dReLU(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.conv2 = Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, x, skip):
        x = self.up(x)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class SegmentationHead(nn.Sequential):

    def __init__(self, in_channels, out_channels, kernel_size=3, upsampling=1):
        conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        upsampling = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        super().__init__(conv2d, upsampling)


class MRConv4d(nn.Module):
    """
    Max-Relative Graph Convolution (Paper: https://arxiv.org/abs/1904.03751) for dense data type

    K is the number of superpatches, therefore hops equals res // K.
    """

    def __init__(self, in_channels, out_channels, K=2):
        super(MRConv4d, self).__init__()
        self.nn = nn.Sequential(
            nn.Conv2d(in_channels * 2, out_channels, 1),
            nn.BatchNorm2d(in_channels * 2),
            nn.GELU()
        )
        self.K = K

    def forward(self, x):
        B, C, H, W = x.shape

        x_j = x - x
        for i in range(self.K, H, self.K):
            x_c = x - torch.cat([x[:, :, -i:, :], x[:, :, :-i, :]], dim=2)
            x_j = torch.max(x_j, x_c)
        for i in range(self.K, W, self.K):
            x_r = x - torch.cat([x[:, :, :, -i:], x[:, :, :, :-i]], dim=3)
            x_j = torch.max(x_j, x_r)

        x = torch.cat([x, x_j], dim=1)
        return self.nn(x)


class Grapher(nn.Module):
    """
    Grapher module with graph convolution and fc layers
    """

    def __init__(self, in_channels, drop_path=0.0, K=2):
        super(Grapher, self).__init__()
        self.channels = in_channels
        self.K = K

        self.fc1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1, stride=1, padding=0),
            nn.BatchNorm2d(in_channels),
        )
        self.graph_conv = MRConv4d(in_channels, in_channels * 2, K=self.K)
        self.fc2 = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, 1, stride=1, padding=0),
            nn.BatchNorm2d(in_channels),
        )  # out_channels back to 1x}
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        _tmp = x
        x = self.fc1(x)
        x = self.graph_conv(x)
        x = self.fc2(x)
        x = self.drop_path(x) + _tmp

        return x


class Downsample(nn.Module):
    """ Convolution-based downsample
    """

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_dim),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class FFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop_path=0.0):
        super().__init__()
        out_features = out_features or in_features  # same as input
        hidden_features = hidden_features or in_features  # x4
        self.fc1 = nn.Sequential(
            nn.Conv2d(in_features, hidden_features, 1, stride=1, padding=0),
            nn.BatchNorm2d(hidden_features),
        )
        self.act = nn.GELU()
        self.fc2 = nn.Sequential(
            nn.Conv2d(hidden_features, out_features, 1, stride=1, padding=0),
            nn.BatchNorm2d(out_features),
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        shortcut = x
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop_path(x) + shortcut
        return x


class DecoderCup(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        head_channels = 512
        self.conv_more = Conv2dReLU(
            1024,
            head_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=True,
        )


        if self.config.n_skip != 0:
            skip_channels = self.config.skip_channels
            for i in range(4 - self.config.n_skip):
                skip_channels[3 - i] = 0
        else:
            skip_channels = [0, 0, 0, 0]

        self.decoder1 = DecoderBlock1(in_channels=512, skip_channels=512, out_channels=256)
        self.decoder2 = DecoderBlock2(in_channels=256, skip_channels=256, out_channels=128)
        self.decoder3 = DecoderBlock3(in_channels=128, skip_channels=128, out_channels=64)
        self.decoder4 = DecoderBlock4(in_channels=64, skip_channels=0, out_channels=16)

    def forward(self, hidden_states, features=None):

        x = self.conv_more(hidden_states)
        x = self.decoder1(x, features[0])
        x = self.decoder2(x, features[1])
        x = self.decoder3(x, features[2])
        x = self.decoder4(x, None)
        return x


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


from lib.nn import SynchronizedBatchNorm2d


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = SynchronizedBatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = SynchronizedBatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class IGTNet(nn.Module):
    def __init__(self, config, img_size=256, num_classes=3, zero_head=False, vis=False):
        super(IGTNet, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.bottleneck = SVGA(config, img_size, vis)
        self.decoder = DecoderCup(config)
        self.segmentation_head = SegmentationHead(
            in_channels=config['decoder_channels'][-1],
            out_channels=config['n_classes'],
            kernel_size=3,
        )
        self.config = config

    def forward(self, x):
        if x.size()[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        x, features = self.bottleneck(x)  # (B, n_patch, hidden)
        x = self.decoder(x, features)
        logits = self.segmentation_head(x)
        return logits


CONFIGS = {
    'IGT-Net': configs.get_igt_config()
}


if __name__ == '__main__':
    config_vit = CONFIGS['IGT-Net']
    config_vit.n_skip = 3
    net = IGTNet(config_vit, img_size=256, num_classes=3)
    input = torch.randn(1, 3, 256, 256)
    flops, params = profile(net, inputs=(input,))
    flops, params = clever_format([flops, params], "%.3f")
    print(flops, params)
