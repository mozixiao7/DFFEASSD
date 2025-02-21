import os
import sys
# sys.path.append('../')
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

import yaml
from torch import Tensor
import scipy.io as sio
from torch.utils import data
from collections import OrderedDict
from torch.nn.parameter import Parameter
from torch.autograd import Variable
import random
# from rawformer.models.classifier import SequencePooling
# from FcaNet.model.layer import MultiSpectralAttentionLayer
from FcaNet.model.layer import MultiSpectralAttentionLayer

class ELA(nn.Module):
    def __init__(self,channel,kernel_size=3):
        super(ELA, self).__init__()
        self.pad=kernel_size//2
        self.conv=nn.Conv1d(channel,channel,kernel_size=kernel_size,padding=self.pad,groups=channel,bias=False)
        self.gn = nn.GroupNorm(16,channel)
        self.sig = nn.Sigmoid()
    def forward(self,x):
        identify = x
        b,c,h,w=x.size()
        x_h=torch.mean(x,dim=3,keepdim=True).view(b,c,h)
        x_w = torch.mean(x,dim=2,keepdim=True).view(b,c,w)
        x_h=self.sig(self.gn(self.conv(x_h))).view(b,c,h,1)
        x_w = self.sig(self.gn(self.conv(x_w))).view(b,c,1,w)
        return x*x_h*x_w+identify
class FCA_ELA(nn.Module):
    def __init__(self,channels):
        super(FCA_ELA, self).__init__()
        c2wh = dict([(64, 56), (128, 28), (256, 14), (512, 7),(32,112)])
        self.fca = MultiSpectralAttentionLayer(channels,c2wh[channels], c2wh[channels],  reduction=16, freq_sel_method = 'top16')
        self.ela = ELA(channels)
        self.conv = nn.Conv2d(channels*2, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(channels)
        self.act = nn.ReLU()
    def forward(self,x):
        fca = self.fca(x)
        ela = self.ela(x)
        concantx = torch.cat((fca,ela),1)
        out = self.act(self.bn(self.conv(concantx)))
        return out
class CONV(nn.Module):
    @staticmethod
    def to_mel(hz):
        return 2595 * np.log10(1 + hz / 700)

    @staticmethod
    def to_hz(mel):
        return 700 * (10 ** (mel / 2595) - 1)

    def __init__(self, device, out_channels, kernel_size, in_channels=1, sample_rate=16000,
                 stride=1, padding=0, dilation=1, bias=False, groups=1, mask=False):
        super(CONV, self).__init__()
        if in_channels != 1:
            msg = "SincConv only support one input channel (here, in_channels = {%i})" % (in_channels)
            raise ValueError(msg)

        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.sample_rate = sample_rate

        # Forcing the filters to be odd (i.e, perfectly symmetrics)
        if kernel_size % 2 == 0:
            self.kernel_size = self.kernel_size + 1

        self.device = device
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.mask = mask
        if bias:
            raise ValueError('SincConv does not support bias.')
        if groups > 1:
            raise ValueError('SincConv does not support groups.')
        self.device = device

        NFFT = 512
        f = int(self.sample_rate / 2) * np.linspace(0, 1, int(NFFT / 2) + 1)
        fmel = self.to_mel(f)
        fmelmax = np.max(fmel)
        fmelmin = np.min(fmel)
        filbandwidthsmel = np.linspace(fmelmin, fmelmax, self.out_channels + 1)
        filbandwidthsf = self.to_hz(filbandwidthsmel)

        self.mel = filbandwidthsf
        self.hsupp = torch.arange(-(self.kernel_size - 1) / 2, (self.kernel_size - 1) / 2 + 1)
        self.band_pass = torch.zeros(self.out_channels, self.kernel_size)

    def forward(self, x, mask=False):
        for i in range(len(self.mel) - 1):
            fmin = self.mel[i]
            fmax = self.mel[i + 1]
            hHigh = (2 * fmax / self.sample_rate) * np.sinc(2 * fmax * self.hsupp / self.sample_rate)
            hLow = (2 * fmin / self.sample_rate) * np.sinc(2 * fmin * self.hsupp / self.sample_rate)
            hideal = hHigh - hLow

            self.band_pass[i, :] = Tensor(np.hamming(self.kernel_size)) * Tensor(hideal)

        band_pass_filter = self.band_pass.to(self.device)

        # Frequency masking: We randomly mask (1/5)th of no. of sinc filters channels (70)
        if (mask == True):
            for i1 in range(1):
                A = np.random.uniform(0, 14)
                A = int(A)
                A0 = random.randint(0, band_pass_filter.shape[0] - A)
                band_pass_filter[A0:A0 + A, :] = 0
        else:
            band_pass_filter = band_pass_filter

        self.filters = (band_pass_filter).view(self.out_channels, 1, self.kernel_size)

        return F.conv1d(x, self.filters, stride=self.stride,
                        padding=self.padding, dilation=self.dilation,
                        bias=None, groups=1)


class Residual_block(nn.Module):
    def __init__(self, nb_filts, first=False):
        super(Residual_block, self).__init__()
        self.first = first
        c2wh = dict([(32,112),(64, 56), (128, 28), (256, 14), (512, 7)])

        if not self.first:
            self.bn1 = nn.BatchNorm2d(num_features=nb_filts[0])
            self.conv1 = nn.Conv2d(in_channels=nb_filts[0],
                                   out_channels=nb_filts[1],
                                   kernel_size=(2, 3),
                                   padding=(1, 1),
                                   stride=1)
        self.selu = nn.SELU(inplace=True)

        self.conv_1 = nn.Conv2d(in_channels=1,
                                out_channels=nb_filts[1],
                                kernel_size=(2, 3),
                                padding=(1, 1),
                                stride=1)
        self.bn2 = nn.BatchNorm2d(num_features=nb_filts[1])
        self.conv2 = nn.Conv2d(in_channels=nb_filts[1],
                               out_channels=nb_filts[1],

                               kernel_size=(2, 3),
                               padding=(0, 1),
                               stride=1)
        # self.fca = MultiSpectralAttentionLayer(channel= nb_filts[1],dct_h=c2wh[nb_filts[1]], dct_w=c2wh[nb_filts[1]],reduction=16, freq_sel_method='top16')
        if nb_filts[0] != nb_filts[1]:
            self.downsample = True
            self.conv_downsample = nn.Conv2d(in_channels=nb_filts[0],
                                             out_channels=nb_filts[1],
                                             padding=(0, 1),
                                             kernel_size=(1, 3),
                                             stride=1)

        else:
            self.downsample = False
        self.mp = nn.MaxPool2d((1, 3))

    def forward(self, x):
        identity = x

        if not self.first:
            out = self.bn1(x)
            out = self.selu(out)
            out = self.conv1(x)
        else:
            x = x
            out = self.conv_1(x)

        out = self.bn2(out)
        out = self.selu(out)
        out = self.conv2(out)
        # out = self.fca(out)

        if self.downsample:
            identity = self.conv_downsample(identity)

        out += identity
        out = self.mp(out)
        print('res:',out.shape)
        return out

class Raw3D(nn.Module):
    def __init__(self, d_args, device):
        super(Raw3D, self).__init__()
        self.device = device

        '''
        Sinc conv. layer
        '''
        self.conv_time = CONV(device=self.device,
                              out_channels=70,
                              kernel_size=d_args['first_conv'],
                              in_channels=d_args['in_channels']
                              )

        self.first_bn = nn.BatchNorm2d(num_features=1)

        self.selu = nn.SELU(inplace=True)

        # Note that here you can also use only one encoder to reduce the network parameters which is jsut half of the 0.44M (mentioned in the paper). I was doing some subband analysis and forget to remove the use of two encoders.  I also checked with one encoder and found same results.

        c2wh = dict([(32, 112), (64, 56), (128, 28), (256, 14), (512, 7)])
        self.encoder = nn.Sequential(
            nn.Sequential(Residual_block(nb_filts=d_args['filts1'][1], first=True)),
            # MultiSpectralAttentionLayer(channel= 32,dct_h=c2wh[32], dct_w=c2wh[32],reduction=16, freq_sel_method='top16'),
            # ELA(channel=32),
            FCA_ELA(32),
            nn.Sequential(Residual_block(nb_filts=d_args['filts1'][1])),
            # MultiSpectralAttentionLayer(channel=32, dct_h=c2wh[32], dct_w=c2wh[32], reduction=16,freq_sel_method='top16'),
            # ELA(channel=32),
            FCA_ELA(32),
            nn.Sequential(Residual_block(nb_filts=d_args['filts1'][2])),
            # MultiSpectralAttentionLayer(channel=64, dct_h=c2wh[64], dct_w=c2wh[64], reduction=16,freq_sel_method='top16'),
            # ELA(channel=64),
            FCA_ELA(64),
            nn.Sequential(Residual_block(nb_filts=d_args['filts1'][3])),
            # MultiSpectralAttentionLayer(channel=64, dct_h=c2wh[64], dct_w=c2wh[64], reduction=16,freq_sel_method='top16'),
            # ELA(channel=64),
            FCA_ELA(64),
            nn.Sequential(Residual_block(nb_filts=d_args['filts1'][3])),
            # MultiSpectralAttentionLayer(channel=64, dct_h=c2wh[64], dct_w=c2wh[64], reduction=16,freq_sel_method='top16'),
            # ELA(channel=64),
            FCA_ELA(64),
            nn.Sequential(Residual_block(nb_filts=d_args['filts1'][3])),
            # MultiSpectralAttentionLayer(channel=64, dct_h=c2wh[64], dct_w=c2wh[64], reduction=16,freq_sel_method='top16'),
            # ELA(channel=64)
            FCA_ELA(64)
        )

        self.bn_before_gru = nn.BatchNorm2d(num_features=d_args['filts1'][3][-1])
        self.gru = nn.GRU(input_size=d_args['filts1'][3][-1], hidden_size=d_args['gru_node'],
                          num_layers=d_args['nb_gru_layer'], batch_first=True)
        self.fc1_gru = nn.Linear(in_features=d_args['gru_node'], out_features=d_args['nb_fc_node'])
        self.fc2_gru = nn.Linear(in_features=d_args['nb_fc_node'], out_features=d_args['nb_classes'], bias=True)
        self.sig = nn.Sigmoid()
        # self.sequence = SequencePooling(64)
        # self.cfp = EVCBlock(64,64)

        # classifier layer with nclass=2 and 7 is number of nodes remaining after pooling layer in Spectro-temporal graph attention layer
        # self.proj_node = nn.Linear(7, 2)
    def forward(self, x, Freq_aug=False):
        """
        x= (#bs,samples)
        """

        # follow sincNet recipe
        # nb_samp = x.shape[0]
        # len_seq = x.shape[1]
        # x = x.view(nb_samp, 1, len_seq)
        # Freq masking during training only

        if (Freq_aug == True):
            x = self.conv_time(x, mask=True)  # (#bs,sinc_filt(70),64472)

        else:
            x = self.conv_time(x, mask=False)
        # print('con_time',x.shape)

        """
        Different with the our RawNet2 model, we interpret the output of sinc-convolution layer as 2-dimensional image with one channel (like 2-D representation).
        """
        x = x.unsqueeze(dim=1)  # 2-D (#bs,1,sinc-filt(70),64472)

        x = F.max_pool2d(torch.abs(x), (3, 3))  # [#bs, C(1),F(23),T(21490)]
        # print('x.maxpool:',x.shape)

        x = self.first_bn(x)
        x = self.selu(x)

        # encoder structure for spectral GAT
        x = self.encoder(x)  # [#bs, C(64), F(23), T(29)]
        # print('encoder:',x.shape)

        # x = self.cfp(x)[0]
        # print(x.shape)

        x = self.bn_before_gru(x)
        # print('bn_before_gru',x.shape) #torch.Size([16, 64, 23, 29])
        x = self.selu(x)
        # print('selu',x.shape) #torch.Size([16, 64, 23, 29])

        nb_samp = x.shape[0]
        ch_seq = x.shape[1]
        x = x.view(nb_samp, ch_seq, -1)
        # print('x.view', x.shape)
        x = x.permute(0,2,1)  # (batch, filt, time) >> (batch, time, filt)
        self.gru.flatten_parameters()
        x, _ = self.gru(x)
        # print('gru',x.shape) #torch.Size([16, 29, 1024])
        x = x[:, -1, :]
        # print('x[:.-1,:]',x.shape) #torch.Size([16, 1024])
        x = self.fc1_gru(x)
        # print('fc1',x.shape) #torch.Size([16, 1024])
        x = self.fc2_gru(x)
        # print('fc2',x.shape) #torch.Size([16, 2])
        # x = self.sequence(x)

        return x


        # x_pool3 = self.pool3(x_gat3)
        #
        # out_proj = self.proj(x_pool3).flatten(1)  # (#bs,#nodes) --> [#bs, 7]
        #
        # output = self.proj_node(out_proj)  # (#bs, output node(no. of classes)) ---> [#bs,2]
        #
        # return output

    def _make_layer(self, nb_blocks, nb_filts, first=False):
        layers = []
        # def __init__(self, nb_filts, first = False):
        for i in range(nb_blocks):
            first = first if i == 0 else False
            layers.append(Residual_block(nb_filts=nb_filts,
                                         first=first))
            if i == 0: nb_filts[0] = nb_filts[1]

        return nn.Sequential(*layers)
# class Raw3Dtf(nn.Module):
#     def __init__(self, d_args, device):
#         super(Raw3Dtf, self).__init__()
#         self.device = device
#
#         '''
#         Sinc conv. layer
#         '''
#         self.conv_time = CONV(device=self.device,
#                               out_channels=70,
#                               kernel_size=d_args['first_conv'],
#                               in_channels=d_args['in_channels']
#                               )
#
#         self.first_bn = nn.BatchNorm2d(num_features=1)
#
#         self.selu = nn.SELU(inplace=True)
#
#         # Note that here you can also use only one encoder to reduce the network parameters which is jsut half of the 0.44M (mentioned in the paper). I was doing some subband analysis and forget to remove the use of two encoders.  I also checked with one encoder and found same results.
#
#         c2wh = dict([(32, 112), (64, 56), (128, 28), (256, 14), (512, 7)])
#         self.encoder = nn.Sequential(
#             nn.Sequential(Residual_block(nb_filts=d_args['filts1'][1], first=True)),
#             MultiSpectralAttentionLayer(channel= 32,dct_h=c2wh[32], dct_w=c2wh[32],reduction=16, freq_sel_method='top16'),
#             ELA(channel=32),
#             nn.Sequential(Residual_block(nb_filts=d_args['filts1'][1])),
#             MultiSpectralAttentionLayer(channel=32, dct_h=c2wh[32], dct_w=c2wh[32], reduction=16,freq_sel_method='top16'),
#             ELA(channel=32),
#             nn.Sequential(Residual_block(nb_filts=d_args['filts1'][2])),
#             MultiSpectralAttentionLayer(channel=64, dct_h=c2wh[64], dct_w=c2wh[64], reduction=16,freq_sel_method='top16'),
#             ELA(channel=64),
#             nn.Sequential(Residual_block(nb_filts=d_args['filts1'][3])),
#             MultiSpectralAttentionLayer(channel=64, dct_h=c2wh[64], dct_w=c2wh[64], reduction=16,freq_sel_method='top16'),
#             ELA(channel=64),
#             nn.Sequential(Residual_block(nb_filts=d_args['filts1'][3])),
#             MultiSpectralAttentionLayer(channel=64, dct_h=c2wh[64], dct_w=c2wh[64], reduction=16,freq_sel_method='top16'),
#             ELA(channel=64),
#             nn.Sequential(Residual_block(nb_filts=d_args['filts1'][3])),
#             MultiSpectralAttentionLayer(channel=64, dct_h=c2wh[64], dct_w=c2wh[64], reduction=16,freq_sel_method='top16'),
#             ELA(channel=64)
#         )
#         # Projection layers
#         # self.proj1 = nn.Linear(14, 12)
#         # self.proj2 = nn.Linear(23, 12)
#         # self.proj = nn.Linear(16, 1)
#         self.bn_before_gru = nn.BatchNorm2d(num_features=d_args['filts1'][3][-1])
#         # self.gru = nn.GRU(input_size=d_args['filts1'][3][-1], hidden_size=d_args['gru_node'],
#         #                   num_layers=d_args['nb_gru_layer'], batch_first=True)
#         # self.fc1_gru = nn.Linear(in_features=d_args['gru_node'], out_features=d_args['nb_fc_node'])
#         # self.fc2_gru = nn.Linear(in_features=d_args['nb_fc_node'], out_features=d_args['nb_classes'], bias=True)
#         self.sig = nn.Sigmoid()
#         self.sequence = SequencePooling(64)
#
#         # classifier layer with nclass=2 and 7 is number of nodes remaining after pooling layer in Spectro-temporal graph attention layer
#         # self.proj_node = nn.Linear(7, 2)
#     def forward(self, x, Freq_aug=False):
#         """
#         x= (#bs,samples)
#         """
#
#         # follow sincNet recipe
#         # nb_samp = x.shape[0]
#         # len_seq = x.shape[1]
#         # x = x.view(nb_samp, 1, len_seq)
#         # Freq masking during training only
#
#         if (Freq_aug == True):
#             x = self.conv_time(x, mask=True)  # (#bs,sinc_filt(70),64472)
#
#         else:
#             x = self.conv_time(x, mask=False)
#         # print('con_time',x.shape)
#
#         """
#         Different with the our RawNet2 model, we interpret the output of sinc-convolution layer as 2-dimensional image with one channel (like 2-D representation).
#         """
#         x = x.unsqueeze(dim=1)  # 2-D (#bs,1,sinc-filt(70),64472)
#
#         x = F.max_pool2d(torch.abs(x), (3, 3))  # [#bs, C(1),F(23),T(21490)]
#
#         x = self.first_bn(x)
#         x = self.selu(x)
#
#         # encoder structure for spectral GAT
#         x = self.encoder(x)  # [#bs, C(64), F(23), T(29)]
#         # print('encoder:',x.shape)
#         x = self.bn_before_gru(x)
#
#         # print('bn_before_gru',x.shape) #torch.Size([16, 512, 29])
#         x = self.selu(x)
#         # print('selu',x.shape) #torch.Size([16, 512, 29])
#         nb_samp = x.shape[0]
#         ch_seq = x.shape[1]
#         x = x.view(nb_samp, ch_seq, -1)
#         # print('x.view', x.shape)
#         x = x.permute(0,2,1)  # (batch, filt, time) >> (batch, time, filt)
#         # self.gru.flatten_parameters()
#         # x, _ = self.gru(x)
#         # # print('gru',x.shape) #torch.Size([16, 29, 1024])
#         # x = x[:, -1, :]
#         # # print('x[:.-1,:]',x.shape) #torch.Size([16, 1024])
#         # x = self.fc1_gru(x)
#         # # print('fc1',x.shape) #torch.Size([16, 1024])
#         # x = self.fc2_gru(x)
#         # # print('fc2',x.shape) #torch.Size([16, 2])
#         x = self.sequence(x)
#         return x
#
#
#         # x_pool3 = self.pool3(x_gat3)
#         #
#         # out_proj = self.proj(x_pool3).flatten(1)  # (#bs,#nodes) --> [#bs, 7]
#         #
#         # output = self.proj_node(out_proj)  # (#bs, output node(no. of classes)) ---> [#bs,2]
#         #
#         # return output
#
#     def _make_layer(self, nb_blocks, nb_filts, first=False):
#         layers = []
#         # def __init__(self, nb_filts, first = False):
#         for i in range(nb_blocks):
#             first = first if i == 0 else False
#             layers.append(Residual_block(nb_filts=nb_filts,
#                                          first=first))
#             if i == 0: nb_filts[0] = nb_filts[1]
#
#         return nn.Sequential(*layers)
if __name__ == '__main__':
    print(torch.cuda.device_count())
    device = torch.device('cuda:0')
    dir_yaml = os.path.splitext('model_config_RawNet2')[0] + '.yaml'
    print(dir_yaml)
    with open(dir_yaml, 'r') as f_yaml:
        parser1 = yaml.load(f_yaml, Loader=yaml.FullLoader)
    model = Raw3D(parser1['model'], device)
    # model = nn.DataParallel(model,device_ids=[11,12,13,14])
    model = model.to(device)
    # print(model)
    x = torch.randn(16, 1, 64000).to(device)
    x = model(x)
    print(x.shape)
