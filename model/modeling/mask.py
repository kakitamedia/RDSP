import torch
import torch.nn as nn

from .base_networks import *

def build_mask_model(cfg):
    if not cfg.MODEL.MASK.FLAG:
        return EmptyMaskModel()

    model = MaskWrapper(cfg)
    if cfg.MODEL.MASK.WEIGHT_FIX:
        for param in model.parameters():
            param.requires_grad = False

    if cfg.MODEL.MASK.PRETRAINED_MODEL:
        print(f'Pretrained mask model was loaded from {cfg.MODEL.MASK.PRETRAINED_MODEL}')
        model.load_state_dict(torch.load(cfg.MODEL.MASK.PRETRAINED_MODEL))

    return model


class MaskWrapper(nn.Module):
    def __init__(self,cfg):
        super(MaskWrapper, self).__init__()

        input_channel = 3
        self.feat = cfg.MODEL.MASK.FEAT
        self.bias = cfg.MODEL.BIAS
        self.activation = cfg.MODEL.ACTIVATION
        self.normalization = cfg.MODEL.NORMALIZATION

        self.init_layers = ConvBlock(input_channel, self.feat, bias=self.bias, activation=self.activation, normalization=self.normalization)
        self.output_layers = ConvBlock(self.feat, 1, bias=self.bias, activation='sigmoid', normalization=self.normalization)
        self.layers = self._make_main_layers(cfg)

        self.cood_layers = self._make_cood_layers(cfg)
        self.context_init, self.context_convs, self.context_fcs = self._make_context_layers(cfg)

        self.cood_type = cfg.MODEL.MASK.COOD.MODE
        self.context_type = cfg.MODEL.MASK.CONTEXT.MODE

    def forward(self, x, context=None, cood=None):
        x = self.init_layers(x)
        if self.cood_layers is not None:
            cood = self.cood_layers(cood)
            if self.cood_type == 'add':
                x = x + cood
            elif self.cood_type == 'mul':
                x = x * cood

        if self.context_convs is not None:
            context = self.context_init(context)
            if self.cood_layers is not None:
                context_cood = self.cood_layers(self._make_context_cood(context))
                if self.cood_type == 'add':
                    context = context + context_cood
                elif self.cood_type == 'mul':
                    context = context * context_cood

            context = self.context_convs(context).squeeze(3).squeeze(2)
            context = self.context_fcs(context).unsqueeze(2).unsqueeze(3)
            context = context.expand(*x.shape)
            if self.context_type == 'add':
                x = x + context
            elif self.context_type == 'mul':
                x = x * context

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
            return None, None, None

        init_conv = [
            ConvBlock(3, self.feat, kernel_size=7, stride=2, padding=3, bias=self.bias, activation=self.activation, normalization=self.normalization),
            nn.MaxPool2d(kernel_size=2, stride=2)
        ]

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

        return nn.Sequential(*init_conv), nn.Sequential(*convs), nn.Sequential(*fcs)


    def _make_main_layers(self, cfg):
        if cfg.MODEL.MASK.ARCITECTURE == 'UNet':
            return UNetBlock(self.feat, down_scale=5, num_convs=3, bias=self.bias, activation=self.activation, normalization=self.normalization)
        elif cfg.MODEL.MASK.ARCITECTURE == 'Flat':
            return FlatBlock(self.feat, down_scale=5, num_convs=3, bias=self.bias, activation=self.activation, normalization=self.normalization)


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
                    ConvBlock(base_filter, base_filter, kernel_size=3, stride=1, padding=1, )
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


class FlatBlock(nn.Module):
    def __init__(self, base_filter=64, down_scale=3, num_convs=3, bias=True, activation='prelu', normalization=None):
        super(FlatBlock, self).__init__()

        self.num_convs = num_convs

        self.conv_blocks = []
        for _ in range(down_scale):
            self.conv_blocks.append(
                ConvBlock(base_filter, base_filter, kernel_size=3, stride=1, padding=1, activation=activation, normalization=normalization)
            )
            self.conv_blocks.append(
                ConvBlock(base_filter, base_filter, kernel_size=3, stride=1, padding=1, activation=activation, normalization=normalization)
            )
            for _ in range(num_convs-2):
                self.conv_blocks.append(
                    ConvBlock(base_filter, base_filter, kernel_size=3, stride=1, padding=1, )
                )
        self.conv_blocks = nn.ModuleList(self.conv_blocks)

        self.deconv_blocks = []
        for _ in range(down_scale):
            self.deconv_blocks.append(
                DeconvBlock(base_filter, base_filter, kernel_size=3, stride=1, padding=1, activation=activation, normalization=normalization)
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

class EmptyMaskModel(nn.Module):
    def __init__(self):
        super(EmptyMaskModel, self).__init__()

    def forward(self, x, context=None, cood=None):
        return torch.ones_like(x)[:, 0:1, :, :]