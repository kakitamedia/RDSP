import torch
import torch.nn as nn

from .base_networks import *
from .esrt import ESRT
from .imdn import IMDN

def build_sr_model(cfg):
    if not cfg.MODEL.SR.FLAG:
        return nn.Upsample(scale_factor=cfg.MODEL.SR.FACTOR, mode=cfg.MODEL.SR.INTERP_METHOD)

    if cfg.MODEL.SR.ARCITECTURE == 'DBPN':
        model = DBPN(scale_factor=cfg.MODEL.SR.FACTOR)
    elif cfg.MODEL.SR.ARCITECTURE == 'DDBPN':
        model = DDBPN(scale_factor=cfg.MODEL.SR.FACTOR)
    elif cfg.MODEL.SR.ARCITECTURE == 'ESRT':
        model = ESRT(upscale=cfg.MODEL.SR.FACTOR)
    elif cfg.MODEL.SR.ARCITECTURE == 'IMDN':
        model = IMDN(upscale=cfg.MODEL.SR.FACTOR)

    if cfg.MODEL.SR.WEIGHT_FIX:
        for param in model.paramters():
            param.requires_grad = False

    if cfg.MODEL.SR.PRETRAINED_MODEL:
        print(f'Pretrained SR model was loaded from {cfg.MODEL.SR.PRETRAINED_MODEL}')
        model.load_state_dict(torch.load(cfg.MODEL.SR.PRETRAINED_MODEL))

    return model


class DBPN(nn.Module):
    def __init__(self, scale_factor, num_stages=4, input_channels=3, num_channels=64, feat=256, bias=True, activation='prelu', normalization=None):
        super(DBPN, self).__init__()

        params = {
            1: [3, 1, 1],
            2: [6, 2, 2],
            4: [8, 4, 2],
            8: [12, 8, 2],
        }

        kernel_size, stride, padding = params[scale_factor]

        self.init_convs = nn.Sequential(
            ConvBlock(input_channels, feat, kernel_size=3, stride=1, padding=1, bias=bias, activation=activation, normalization=normalization),
            ConvBlock(feat, num_channels, kernel_size=3, stride=1, padding=1, bias=bias, activation=activation, normalization=normalization),
        )

        self.up_layers = nn.ModuleList([
            UpBlock(num_channels, stage=1, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias, activation=activation, normalization=normalization)
            for i in range(1, num_stages+1)
        ])

        self.down_layers = nn.ModuleList([
            DownBlock(num_channels, stage=1, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias, activation=activation, normalization=normalization)
            for i in range(1, num_stages)
        ])

        self.output_conv = ConvBlock(num_channels, input_channels, kernel_size=3, stride=1, padding=1, bias=bias, activation=None, normalization=None)


    def forward(self, x):
        l = self.init_convs(x)

        for i in range(self.num_stages):
            h = self.up_layers[i](l)

            if i == self.num_stages - 1:
                break
            l = self.down_layers[i](h)

        return self.output_conv(h)


class DDBPN(nn.Module):
    def __init__(self, scale_factor, num_stages=4, input_channels=3, num_channels=64, feat=256, bias=True, activation='prelu', normalization=None):
        super(DDBPN, self).__init__()

        params = {
            1: [3, 1, 1],
            2: [6, 2, 2],
            4: [8, 4, 2],
            8: [12, 8, 2],
        }

        self.scale_factor = scale_factor
        self.num_stages = num_stages


        kernel_size, stride, padding = params[scale_factor]

        self.init_convs = nn.Sequential(
            ConvBlock(input_channels, feat, kernel_size=3, stride=1, padding=1, bias=bias, activation=activation, normalization=normalization),
            ConvBlock(feat, num_channels, kernel_size=3, stride=1, padding=1, bias=bias, activation=activation, normalization=normalization),
        )

        self.up_layers = nn.ModuleList([
            UpBlock(num_channels, stage=i, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias, activation=activation, normalization=normalization)
            for i in range(1, num_stages+1)
        ])

        self.down_layers = nn.ModuleList([
            DownBlock(num_channels, stage=i, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias, activation=activation, normalization=normalization)
            for i in range(1, num_stages)
        ])

        self.output_conv = ConvBlock(num_stages*num_channels, input_channels, kernel_size=3, stride=1, padding=1, bias=bias, activation=None, normalization=None)

    def forward(self, x):
        batch, _, height, width = x.shape

        l = self.init_convs(x)

        concat_l = torch.empty((batch, 0, height, width), dtype=x.dtype, device=x.device)
        concat_h = torch.empty((batch, 0, height*self.scale_factor, width*self.scale_factor), dtype=x.dtype, device=x.device)

        for i in range(self.num_stages):
            concat_l = torch.cat((l, concat_l), dim=1)
            h = self.up_layers[i](concat_l)

            concat_h = torch.cat((h, concat_h), dim=1)
            if i == self.num_stages - 1:
                break
            l = self.down_layers[i](concat_h)


        return self.output_conv(concat_h)