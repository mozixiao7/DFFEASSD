import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import os
import random
import numpy as np
from torch.autograd import Function
import pickle
import math
from FcaNet.model.layer import MultiSpectralAttentionLayer

## Adapted from https://github.com/joaomonteirof/e2e_antispoofing
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
        return x * x_h* x_w+x
class FCA_ELA(nn.Module):
    def __init__(self,channels):
        super(FCA_ELA, self).__init__()
        c2wh = dict([(64, 56), (128, 28), (256, 14), (512, 7)])
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

class SelfAttention(nn.Module):
    def __init__(self, hidden_size, mean_only=False):
        super(SelfAttention, self).__init__()

        #self.output_size = output_size
        self.hidden_size = hidden_size
        self.att_weights = nn.Parameter(torch.Tensor(1, hidden_size),requires_grad=True)

        self.mean_only = mean_only

        init.kaiming_uniform_(self.att_weights)

    def forward(self, inputs):

        batch_size = inputs.size(0)
        weights = torch.bmm(inputs, self.att_weights.permute(1, 0).unsqueeze(0).repeat(batch_size, 1, 1))

        if inputs.size(0)==1:
            attentions = F.softmax(torch.tanh(weights),dim=1)
            weighted = torch.mul(inputs, attentions.expand_as(inputs))
        else:
            attentions = F.softmax(torch.tanh(weights.squeeze()),dim=1)
            weighted = torch.mul(inputs, attentions.unsqueeze(2).expand_as(inputs))

        if self.mean_only:
            return weighted.sum(1)
        else:
            noise = 1e-5*torch.randn(weighted.size())

            if inputs.is_cuda:
                noise = noise.to(inputs.device)
            avg_repr, std_repr = weighted.sum(1), (weighted+noise).std(1)

            representations = torch.cat((avg_repr,std_repr),1)

            return representations


class PreActBlock(nn.Module):
    '''Pre-activation version of the BasicBlock.'''
    expansion = 1

    def __init__(self, in_planes, planes, stride, *args, **kwargs):
        super(PreActBlock, self).__init__()
        c2wh = dict([(64, 56), (128, 28), (256, 14), (512, 7)])
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        # self.eca = ECA(planes)
        # self.ca = CoordAtt(planes,planes)
        # self.sa = sa_layer(planes)
        # self.fca = MultiSpectralAttentionLayer(planes,c2wh[planes], c2wh[planes],  reduction=8, freq_sel_method = 'top16')
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False))

    def forward(self, x):
        out = F.relu(self.bn1(x))
        # print('F.relu.bu:{}'.format(out.shape))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        # print('conv1_out.shape:{}'.format(out.shape))
        out = self.conv2(F.relu(self.bn2(out)))
        # print('conv2_out.shape:{}'.format(out.shape))

        # out = self.eca(out)
        # print('eca_out.shape:{}'.format(out.shape))
        # out = self.ca(out)
        # print('ca_out.shape:{}'.format(out.shape))
        # out = self.sa(out)
        # print('sa_out.shape{}'.format(out.shape))
        # out = self.fca(out)


        out += shortcut
        return out


class PreActBottleneck(nn.Module):
    '''Pre-activation version of the original Bottleneck module.'''
    expansion = 1

    def __init__(self, in_planes, planes, stride, *args, **kwargs):
        super(PreActBottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)


        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False))

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out = self.conv3(F.relu(self.bn3(out)))
        out += shortcut
        return out

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

RESNET_CONFIGS = {'18': [[2, 2, 2, 2], PreActBlock],
                  '28': [[3, 4, 6, 3], PreActBlock],
                  '34': [[3, 4, 6, 3], PreActBlock],
                  '50': [[3, 4, 6, 3], PreActBottleneck],
                  '101': [[3, 4, 23, 3], PreActBottleneck]
                  }

def setup_seed(random_seed, cudnn_deterministic=True):
    # initialization
    torch.manual_seed(random_seed)
    random.seed(random_seed)
    np.random.seed(random_seed)
    os.environ['PYTHONHASHSEED'] = str(random_seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)
        torch.manual_seed(random_seed)
        torch.backends.cudnn.deterministic = cudnn_deterministic
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.enabled=False

class ResNet_FCA(nn.Module):
    def __init__(self, num_nodes, enc_dim, resnet_type='18', nclasses=2):
        self.in_planes = 16
        super(ResNet_FCA, self).__init__()

        layers, block = RESNET_CONFIGS[resnet_type]
        c2wh = dict([(64, 56), (128, 28), (256, 14), (512, 7)])

        self._norm_layer = nn.BatchNorm2d

        self.conv1 = nn.Conv2d(1, 16, kernel_size=(9, 3), stride=(3, 1), padding=(1, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.activation = nn.ReLU()

        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.fca1 = MultiSpectralAttentionLayer(64,c2wh[64], c2wh[64],  reduction=16, freq_sel_method = 'top16')

        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.fca2 = MultiSpectralAttentionLayer(128,c2wh[128], c2wh[128],  reduction=16, freq_sel_method = 'top16')

        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.fca3 = MultiSpectralAttentionLayer(256,c2wh[256], c2wh[256],  reduction=16, freq_sel_method = 'top16')

        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.fca = MultiSpectralAttentionLayer(512,c2wh[512], c2wh[512],  reduction=16, freq_sel_method = 'top16')


        # self.conv5 = nn.Conv2d(512 * block.expansion, 256, kernel_size=(num_nodes, 3), stride=(3, 1), padding=(0, 1),bias=False)
        self.conv5 = nn.Conv2d(512 * block.expansion, 256, kernel_size=(num_nodes, 3), stride=(1, 1), padding=(0, 1),bias=False)
        self.bn5 = nn.BatchNorm2d(256)
        self.fc = nn.Linear(256 * 2, enc_dim)
        self.fc_mu = nn.Linear(enc_dim, nclasses) if nclasses >= 2 else nn.Linear(enc_dim, 1)

        self.initialize_params()
        self.attention = SelfAttention(256)

        # self.aff = AFF(channels=1,r=1/4)

    def initialize_params(self):
        for layer in self.modules():
            if isinstance(layer, torch.nn.Conv2d):
                init.kaiming_normal_(layer.weight, a=0, mode='fan_out')
            elif isinstance(layer, torch.nn.Linear):
                init.kaiming_uniform_(layer.weight)
            elif isinstance(layer, torch.nn.BatchNorm2d) or isinstance(layer, torch.nn.BatchNorm1d):
                layer.weight.data.fill_(1)
                layer.bias.data.zero_()

    def _make_layer(self, block, planes, num_blocks, stride=1):
        norm_layer = self._norm_layer
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = nn.Sequential(conv1x1(self.in_planes, planes * block.expansion, stride),
                                       norm_layer(planes * block.expansion))
        layers = []
        layers.append(block(self.in_planes, planes, stride, downsample, 1, 64, 1, norm_layer))
        self.in_planes = planes * block.expansion
        for _ in range(1, num_blocks):
            layers.append(
                block(self.in_planes, planes, 1, groups=1, base_width=64, dilation=False, norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.activation(self.bn1(x))
        x = self.layer1(x)
        x = self.fca1(x)
        x = self.layer2(x)
        x = self.fca2(x)
        x = self.layer3(x)
        x = self.fca3(x)
        x = self.layer4(x)
        x = self.fca(x)
        x = self.conv5(x)
        x = self.activation(self.bn5(x)).squeeze(2)
        # print('attention_x.shape',x.shape)

        stats = self.attention(x.permute(0, 2, 1).contiguous())

        feat = self.fc(stats)

        mu = self.fc_mu(feat)


        # x = self.conv1(x)
        # print('conv1',x.shape) #conv1 torch.Size([32, 16, 18, 750])
        # x = self.activation(self.bn1(x))
        # print('activation',x.shape)
        # x = self.layer1(x)
        # print('layer1',x.shape) #layer1 torch.Size([32, 64, 18, 750])
        # x = self.layer2(x)
        # print('layer2',x.shape) #layer2 torch.Size([32, 128, 9, 375])
        # x = self.layer3(x)
        # print('layer3',x.shape) #layer3 torch.Size([32, 256, 5, 188])
        # x = self.layer4(x)
        # print('layer4',x.shape) #layer4 torch.Size([32, 512, 3, 94])
        # x = self.conv5(x)
        # print('conv5',x.shape) #conv5 torch.Size([32, 256, 1, 94])
        # x = self.activation(self.bn5(x)).squeeze(2)
        # print('activation',x.shape)
        #
        # stats = self.attention(x.permute(0, 2, 1).contiguous())
        # print('attention',stats.shape) #attention torch.Size([32, 512])
        #
        # feat = self.fc(stats)
        # print('fc',feat.shape)
        #
        # mu = self.fc_mu(feat)
        # print('fc_mu',mu.shape)

        return feat, mu

class ResNet_FCA_ELA(nn.Module):
    def __init__(self, num_nodes, enc_dim, resnet_type='18', nclasses=2):
        self.in_planes = 16
        super(ResNet_FCA_ELA, self).__init__()

        layers, block = RESNET_CONFIGS[resnet_type]
        c2wh = dict([(64, 56), (128, 28), (256, 14), (512, 7)])

        self._norm_layer = nn.BatchNorm2d

        self.conv1 = nn.Conv2d(1, 16, kernel_size=(9, 3), stride=(3, 1), padding=(1, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.activation = nn.ReLU()

        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        # self.fca1 = MultiSpectralAttentionLayer(64,c2wh[64], c2wh[64],  reduction=16, freq_sel_method = 'top16')
        # self.ela1 = ELA(64)
        # self.fcaela1 = FCA_ELA(64)

        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        # self.fca2 = MultiSpectralAttentionLayer(128,c2wh[128], c2wh[128],  reduction=16, freq_sel_method = 'top16')
        # self.ela2 = ELA(128)
        # self.fcaela2 = FCA_ELA(128)

        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        # self.fca3 = MultiSpectralAttentionLayer(256,c2wh[256], c2wh[256],  reduction=16, freq_sel_method = 'top16')
        # self.ela3 = ELA(256)
        # self.fcaela3 = FCA_ELA(256)

        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        # self.fca4 = MultiSpectralAttentionLayer(512,c2wh[512], c2wh[512],  reduction=16, freq_sel_method = 'top16')
        # self.ela4 = ELA(512)
        self.fcaela4 = FCA_ELA(512)

        # self.conv5 = nn.Conv2d(512 * block.expansion, 256, kernel_size=(num_nodes, 3), stride=(3, 1), padding=(0, 1),bias=False)
        self.conv5 = nn.Conv2d(512 * block.expansion, 256, kernel_size=(num_nodes, 3), stride=(1, 1), padding=(0, 1),bias=False)
        self.bn5 = nn.BatchNorm2d(256)
        self.fc = nn.Linear(256 * 2, enc_dim)
        self.fc_mu = nn.Linear(enc_dim, nclasses) if nclasses >= 2 else nn.Linear(enc_dim, 1)

        self.initialize_params()
        self.attention = SelfAttention(256)

        # self.aff = AFF(channels=1,r=1/4)

    def initialize_params(self):
        for layer in self.modules():
            if isinstance(layer, torch.nn.Conv2d):
                init.kaiming_normal_(layer.weight, a=0, mode='fan_out')
            elif isinstance(layer, torch.nn.Linear):
                init.kaiming_uniform_(layer.weight)
            elif isinstance(layer, torch.nn.BatchNorm2d) or isinstance(layer, torch.nn.BatchNorm1d):
                layer.weight.data.fill_(1)
                layer.bias.data.zero_()

    def _make_layer(self, block, planes, num_blocks, stride=1):
        norm_layer = self._norm_layer
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = nn.Sequential(conv1x1(self.in_planes, planes * block.expansion, stride),
                                       norm_layer(planes * block.expansion))
        layers = []
        layers.append(block(self.in_planes, planes, stride, downsample, 1, 64, 1, norm_layer))
        self.in_planes = planes * block.expansion
        for _ in range(1, num_blocks):
            layers.append(
                block(self.in_planes, planes, 1, groups=1, base_width=64, dilation=False, norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.activation(self.bn1(x))
        x = self.layer1(x)
        # x = self.fca1(x)
        # x = self.ela1(x)
        # x = self.fcaela1(x)
        x = self.layer2(x)
        # x = self.fca2(x)
        # x = self.ela2(x)
        # x = self.fcaela2(x)
        x = self.layer3(x)
        # x = self.fca3(x)
        # x = self.ela3(x)
        # x = self.fcaela3(x)
        x = self.layer4(x)
        # x = self.fca4(x)
        # x = self.ela4(x)
        x = self.fcaela4(x)
        x = self.conv5(x)
        x = self.activation(self.bn5(x)).squeeze(2)
        # print('attention_x.shape',x.shape)

        stats = self.attention(x.permute(0, 2, 1).contiguous())

        feat = self.fc(stats)

        mu = self.fc_mu(feat)


        # x = self.conv1(x)
        # print('conv1',x.shape) #conv1 torch.Size([32, 16, 18, 750])
        # x = self.activation(self.bn1(x))
        # print('activation',x.shape)
        # x = self.layer1(x)
        # print('layer1',x.shape) #layer1 torch.Size([32, 64, 18, 750])
        # x = self.layer2(x)
        # print('layer2',x.shape) #layer2 torch.Size([32, 128, 9, 375])
        # x = self.layer3(x)
        # print('layer3',x.shape) #layer3 torch.Size([32, 256, 5, 188])
        # x = self.layer4(x)
        # print('layer4',x.shape) #layer4 torch.Size([32, 512, 3, 94])
        # x = self.conv5(x)
        # print('conv5',x.shape) #conv5 torch.Size([32, 256, 1, 94])
        # x = self.activation(self.bn5(x)).squeeze(2)
        # print('activation',x.shape)
        #
        # stats = self.attention(x.permute(0, 2, 1).contiguous())
        # print('attention',stats.shape) #attention torch.Size([32, 512])
        #
        # feat = self.fc(stats)
        # print('fc',feat.shape)
        #
        # mu = self.fc_mu(feat)
        # print('fc_mu',mu.shape)

        return feat, mu
if __name__ == '__main__':
    device = torch.device('cuda:0')
    # model = ResNet(3,256,resnet_type='18', nclasses=2).to(device)
    # model = SelfAttention(256)
    model = FCA_ELA(512).to(device)
    # print(model)
    x = torch.rand(32,512,3,94).to(device)
    # label = torch.randint(2,(32,)).to(device)
    # print(label.shape)
    out = model(x)
    print(out.shape)
    # score = F.softmax(mu, dim=1)[:, 0]
    # print(score.shape)
    # print(mu,score)
    # ocsoftmax = AMSoftmax(2,256, s=20,m=0.9).to(device)
    # out1,out2 = ocsoftmax(feat,label)
    # print(out1,out2)
    # print(out1.shape,out2.shape)

    # print('x(1)',x[0].shape)
    # print('x(2)',x[1].shape)
    # xcqt = np.load('./features_cqt/train/LA_T_1138215.npy')
    # for i in xcqt:
    #     print(i)
    # print(xcqt.shape)
    # with open('./features_lfcc/train/LA_T_1000824LFCC.pkl','rb') as f:
    #     feat_mat = pickle.load(f)
    # print(feat_mat.shape)
    # xcqcc = np.load('./features_cqcc/train/LA_T_6306026.npy')
    # xcqcc = np.load('LA_T_1276960.npy')
    # xtcqcc = xcqcc.transpose()
    # print(xcqcc.shape)
    # print(xtcqcc.shape)


    # x1 = torch.rand(32,1,256).to(device)
    # x2 = torch.rand(32,1,256).to(device)
    # # x3 = torch.cat((x1,x2),dim=1)
    # # print(x3.shape)
    # model = AFF(channels=1).to(device)
    # print(model)
    # out = model(x1,x2)
    # print(out.shape)
