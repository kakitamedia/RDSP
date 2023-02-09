import torch
import torch.nn as nn

from .base_networks import *

def build_mask_model(cfg):
    model = MaskWrapper(cfg)
    if cfg.MODEL.MASK.WEIGHT_FIX:
        for param in model.parameters():
            param.requires_grad = False

    return model


class MaskWrapper(nn.Module):
    def __init__(self,cfg):
        super(MaskWrapper, self).__init__()

        input_channel = 3
        self.feat = cfg.MODEL.MASK.FEAT
        self.bias = cfg.MODEL.BIAS
        self.activation = cfg.MODEL.ACTIVATION
        self.normalization = cfg.MODEL.NORMALIZATION

        self.init_layer = ConvBlock(input_channel, self.feat, bias=self.bias, activation=self.activation, normalization=self.normalization)
        self.output_layer = ConvBlock(self.feat, 1, bias=self.bias, activation='sigmoid', normalization=self.normalization)
        self.layers = UNetBlock(self.feat, down_scale=5, num_convs=3, bias=self.bias, activation=self.activation, normalization=self.normalization)

        self.cood_layer = self._make_cood_layers(cfg)
        self.context_convs, self.context_fcs = self._make_context_layers(cfg)

    def forward(self, x, context=None, cood=None):
        assert context is None == self.context_layer is None
        assert cood is None == self.cood_layers is None

        x = self.init_layers(x)
        if cood:
            cood = self.cood_layers(cood)
            x = x + cood

        if context:
            context = self.init_layers(context)
            if cood:
                context = self.cood_layer(self._make_context_cood(context))
                context = context + cood

            context = self.context_convs(context).squeeze(3).squeeze(2)
            context = self.context_fcs(context).unsqueeze(2).unsqueeze(3)
            context = context.expand(*x.shape)
            x = x + context

        x = self.layers(x)
        x = self.output_layers(x)

        return x


    def _make_context_cood(self, context):
        batch, channel, height, width = context.shape
        cood_x = torch.arange(width, dtype=context.dtype, device=context.device).view(1, 1, 1, -1).expand(batch, 1, height, width) / 2048
        cood_y = torch.arange(height, dtype=context.dtype, device=context.device).view(1, 1, -1, 1).expand(batch, 1, height, width) / 2048
        context_cood = torch.cat([cood_x, cood_y], dim=1)

        return context_cood


    def _make_cood_layers(self, cfg):
        if not cfg.MODEL.MASK.COOD.FLAG:
            return None

        convs = [ConvBlock(2, self.feat, kernel_size=1, stride=1, padding=0, bias=self.bias, normalization=self.normalization, activation=self.activation)]
        for _ in range(cfg.MODEL.MASK.COOD.NUM_LAYERS - 1):
            convs.append(ConvBlock(self.feat, self.feat, kernel_size=1, stride=1, padding=0, bias=self.bias, normalization=self.normalization, activation=self.activation))

        return nn.Sequential(*convs)


    def _make_context_layers(self, cfg):
        if not cfg.MODEL.MASK.CONTEXT.FLAG:
            return None

        convs = []
        for _ in range(cfg.MODEL.MASK.CONTEXT.NUM_SCALES):
            convs.extend([
                ConvBlock(self.feat, self.feat, bias=self.bias, activation=self.activation, normalization=self.normalization)
                for _ in range(cfg.MODEL.MASK.CONTEXT.NUM_CONVS)
            ])
            convs.append(nn.MaxPool2d(kernel_size=2, stride=2))
        convs.append(nn.AdaptiveMaxPool2d(1))

        fcs = []
        for _ in range(cfg.MODEL.MASK.CONTEXT.NUM_FCS):
            fcs.append(
                DenseBlock(self.feat, self.feat, bias=self.bias, activation=self.activation, normalization=self.normalization)
            )

        return nn.Sequential(*convs), nn.Sequential(*fcs)


class UNetBlock(nn.Module):
    def __init__(self, base_filter=64, down_scale=3, num_convs=3, bias=True, activation='prelu', normalization=None):
        super(UNetBlock, self).__init__()

        self.num_convs = num_convs

        self.conv_blocks = []
        for _ in range(down_scale):
            self.conv_blocks.append(
                ConvBlock(base_filter, base_filter, kernel_size=2, stride=2, padding=0, activation=activation, normalization=normalization)
            )
            self.conv_blocks.append(
                ConvBlock(base_filter, base_filter, kernel_size=3, stride=1, padding=1, activation=activation, normalization=normalization)
            )
            for _ in range(num_convs-2):
                self.conv_blocks.append(
                    ConvBlock(base_filter, base_filter ,kernel_size=3, stride=1, padding=1, )
                )
        self.conv_blocks = nn.ModuleList(self.conv_blocks)

        self.deconv_blocks = []
        for _ in range(down_scale):
            self.deconv_blocks.append(
                DeconvBlock(base_filter, base_filter, kernel_size=2, stride=2, padding=0, activation=activation, normalization=normalization)
            )
            self.deconv_blocks.append(
                ConvBlock(2 * base_filter, base_filter, kernel_size=3, stride=1, padding=1, activation=activation, normalization=normalization)
            )
            for _ in range(num_convs-2):
                self.deconv_blocks.append(
                    ConvBlock(base_filter, base_filter, kernel_size=3, stride=1, padding=1, activation=activation, normalization=normalization)
                )
        self.deconv_blocks = nn.ModuleList(self.deconv_blocks)


    def forward(self, x):
        sources = []
        for i in range(len(self.conv_blocks)):
            if i % self.num_convs == 0 and i != len(self.conv_blocks)-1 :
                sources.append(x)
            x = self.conv_blocks[i](x)

        for i in range(len(self.deconv_blocks)):
            x = self.deconv_blocks[i](x)
            if i % self.num_convs == 0 and len(sources) != 0:
                x = torch.cat((x, sources.pop(-1)), 1)

        return x