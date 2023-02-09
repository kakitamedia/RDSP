import math

import torch.nn as nn
import torch.nn.functional as F

class BaseBlock(nn.Module):
    def __init__(self, output_dim, bias, activation, normalization):
        super(BaseBlock, self).__init__()
        self.output_dim = output_dim
        self.bias = bias
        self.activation = activation
        self.normalization = normalization

    def create_block(self):
        ### Nomalizing layer
        if self.normalization =='batch':
            self.norm = nn.BatchNorm2d(self.output_dim)
        elif self.normalization == 'instance':
            self.norm = nn.InstanceNorm2d(self.output_dim)
        elif self.normalization == 'group':
            self.norm = nn.GroupNorm(32, self.output_dim)
        elif self.normalization == 'layer':
            self.norm = nn.LayerNorm(self.output_dim)
        elif self.normalization == 'spectral':
            self.norm = None
            self.layer = nn.utils.spectral_norm(self.layer)
        elif self.normalization == None:
            self.norm = None

        ### Activation layer
        if self.activation == 'relu':
            self.act = nn.ReLU(True)
        elif self.activation == 'relu6':
            self.act = nn.Hardtanh(0, 6, inplace=True)
        elif self.activation == 'relu1':
            self.act = nn.Hardtanh(0, 1, inplace=True)
        elif self.activation == 'prelu':
            self.act = nn.PReLU(init=0.01)
        elif self.activation == 'lrelu':
            self.act = nn.LeakyReLU(0.01, True)
        elif self.activation == 'tanh':
            self.act = nn.Tanh()
        elif self.activation == 'sigmoid':
            self.act = nn.Sigmoid()
        elif self.activation == None:
            self.act = None

        ### Initialize weights
        if self.activation == 'relu':
            nn.init.kaiming_normal_(self.layer.weight, nonlinearity='relu')
        elif self.activation == 'prelu' or self.activation == 'lrelu':
            nn.init.kaiming_normal_(self.layer.weight, a=0.01, nonlinearity='leaky_relu')
        elif self.activation == 'tanh':
            nn.init.xavier_normal_(self.layer.weight, gain=5/3)
        else:
            nn.init.xavier_normal_(self.layer.weight, gain=1)

        if self.bias:
            nn.init.zeros_(self.layer.bias)


    def forward(self, x):
        x = self.layer(x)

        if self.norm is not None:
            x = self.norm(x)

        if self.act is not None:
            x = self.act(x)

        return x


class DenseBlock(BaseBlock):
    def __init__(self, input_dim, output_dim, bias=False, activation='relu', normalization='batch'):
        super().__init__(output_dim, bias, activation, normalization)
        self.layer = nn.Linear(input_dim, output_dim, bias=bias)
        self.create_block()

        ### Overwrite normalizing layer for 1D version
        self.normalization = normalization
        if self.normalization == 'batch':
            self.norm = nn.BatchNorm1d(output_dim)
        elif self.normalization == 'instance':
            self.norm = nn.InstanceNorm1d(output_dim)


class ConvBlock(BaseBlock):
    def __init__(self, input_dim, output_dim, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=False, activation='relu', normalization='batch'):
        super().__init__(output_dim, bias, activation, normalization)
        self.layer = nn.Conv2d(input_dim, output_dim, kernel_size, stride, padding, dilation=dilation, groups=groups, bias=bias)
        self.create_block()


class DeconvBlock(BaseBlock):
    def __init__(self, input_dim, output_dim, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=False, activation='relu', normalization='batch'):
        super().__init__(output_dim, bias, activation, normalization)
        self.layer = nn.ConvTranspose2d(input_dim, output_dim, kernel_size, stride, padding, dilation=dilation, groups=groups, bias=bias)
        self.create_block()


class BackProjectionBlock(nn.Module):
    def __init__(self):
        super(BackProjectionBlock, self).__init__()

    def forward(self, x):
        if self.stage > 1:
            x = self.init_conv(x)
        h0 = self.conv1(x)
        l0 = self.conv2(h0)
        h1 = self.conv3(l0 - x)

        return h1 + h0


class UpBlock(BackProjectionBlock):
    def __init__(self, num_channels, stage=1, kernel_size=8, stride=4, padding=2, bias=True, activation='prelu', normalization=None):
        super().__init__()

        self.stage = stage

        if self.stage > 1:
            self.init_conv = ConvBlock(num_channels*stage, num_channels, kernel_size=1, stride=1, padding=0, bias=bias, activation=activation, normalization=normalization)

        self.conv1 = DeconvBlock(num_channels, num_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias, activation=activation, normalization=normalization)
        self.conv2 = ConvBlock(num_channels, num_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias, activation=activation, normalization=normalization)
        self.conv3 = DeconvBlock(num_channels, num_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias, activation=activation, normalization=normalization)


class DownBlock(BackProjectionBlock):
    def __init__(self, num_channels, stage=1, kernel_size=8, stride=4, padding=2, bias=True, activation='prelu', normalization=None):
        super().__init__()

        self.stage = stage

        if self.stage > 1:
            self.init_conv = ConvBlock(num_channels*stage, num_channels, kernel_size=1, stride=1, padding=0, bias=bias, activation=activation, normalization=normalization)

        self.conv1 = ConvBlock(num_channels, num_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias, activation=activation, normalization=normalization)
        self.conv2 = DeconvBlock(num_channels, num_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias, activation=activation, normalization=normalization)
        self.conv3 = ConvBlock(num_channels, num_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias, activation=activation, normalization=normalization)


class ResidualBlock(nn.Module):
    def __init__(self, input_dim, output_dim, num_convs=2, kernel_size=3, stride=1, padding=1, bias=False, activation='relu', normalization='batch'):
        super(ResidualBlock, self).__init__()

        self.num_convs = num_convs

        self.layers, self.norms, self.acts = [], [], []
        self.layers.append(nn.Conv2d(input_dim, output_dim, kernel_size, stride, padding, bias=bias))
        for _ in range(num_convs - 1):
            self.layers.append(nn.Conv2d(output_dim, output_dim, kernel_size, 1, padding, bias=bias))

        #TODO: Initialization for skip_layer is not implemented
        if input_dim != output_dim or stride != 1:
            self.skip_layer = [nn.Conv2d(input_dim, output_dim, 1, stride, 0, bias=bias)]
            if bias:
                nn.init.zeros_(self.skip_layer[0].bias)
        else:
            self.skip_layer = None


        for i in range(num_convs):
            ### Normalizing layer
            if normalization == 'batch':
                self.norms.append(nn.BatchNorm2d(output_dim))
                if self.skip_layer is not None:
                    self.skip_layer.append(nn.BatchNorm2d(output_dim))
            elif normalization == 'instance':
                self.norms.append(nn.InstansNorm2d(output_dim))
                if self.skip_layer is not None:
                    self.skip_layer.append(nn.InstansNorm2d(output_dim))
            elif normalization == 'group':
                self.norms.append(nn.GroupNorm(32, output_dim))
                if self.skip_layer is not None:
                    self.skip_layer.append(nn.GroupNorm(32, output_dim))
            elif normalization == 'spectral':
                self.norms.append(None)
                self.layers[i] = nn.utils.spectral_norm(self.layers[i])
                if self.skip_layer is not None:
                    self.skip_layer[0] = nn.utils.spectral_norm(self.skip_layer[0])
            elif normalization == None:
                self.norms.append(None)
            else:
                raise Exception('normalization={} is not implemented.'.format(normalization))

            ### Activation layer
            if activation == 'relu':
                self.acts.append(nn.ReLU(True))
            elif activation == 'lrelu':
                self.acts.append(nn.LeakyReLU(0.01, True))
            elif activation == 'prelu':
                self.acts.append(nn.PReLU(init=0.01))
            elif activation == 'tanh':
                self.acts.append(nn.Tanh())
            elif activation == 'sigmoid':
                self.acts.append(nn.Sigmoid())
            elif activation == None:
                self.acts.append(None)
            else:
                raise Exception('activation={} is not implemented.'.format(activation))

            ### Initialize weights
            if activation == 'relu':
                nn.init.kaiming_normal_(self.layers[i].weight, nonlinearity='relu')
            elif activation == 'lrelu' or activation == 'prelu':
                nn.init.kaiming_normal_(self.layers[i].weight, a=0.01, nonlinearity='leaky_relu')
            elif activation == 'tanh':
                nn.init.xavier_normal_(self.layers[i].weight, gain=5/3)
            elif activation == 'sigmoid':
                nn.init.xavier_normal_(self.layers[i].weight, gain=1)
            elif activation == None:
                nn.init.xaview_normal_(self.layers[i].weight, gain=1)
            else:
                raise Exception('activation={} is not implemented.'.format(activation))

            if bias:
                nn.init.zeros_(self.layers[i].bias)

        self.layers = nn.ModuleList(self.layers)
        self.norms = nn.ModuleList(self.norms)
        self.acts = nn.ModuleList(self.acts)
        if self.skip_layer is not None:
            self.skip_layer = nn.Sequential(*self.skip_layer)

        self.cutoff = (math.floor(kernel_size/2) - padding) * num_convs

    def forward(self, x):
        if self.skip_layer is not None:
            residual = self.skip_layer(x)
        else:
            residual = x

        for i in range(self.num_convs):
            x = self.layers[i](x)

            if self.norms[i] is not None:
                x = self.norms[i](x)

            if i == self.num_convs - 1:
                if self.cutoff == 0:
                    x = x + residual
                else:
                    x = x + residual[:, :, self.cutoff:-self.cutoff, self.cutoff:-self.cutoff]

            if self.acts[i] is not None:
                x = self.acts[i](x)

        return x