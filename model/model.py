#!/usr/bin/env python
# coding: utf-8

import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.activation import Sigmoid


class Dataset(torch.utils.data.Dataset):
    def __init__(self, data, target, transform=None):
        self.transform = transform
        self.data = data
        self.target = target
   
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        out_data = self.data[index]
        out_target = self.target[index]
        
        return out_data, out_target



class Variable(nn.Module):
    def __init__(self, emb_dim=128, num_layer=8, num_filters=128, kernel_sizes=5):
        super(Variable, self).__init__()
        self.filter = num_filters
        self.embedding = nn.Embedding(6, emb_dim)

        self.convs = nn.ModuleList()
        self.convs.append(conv1DBatchNormMish(in_channels=emb_dim, out_channels=num_filters,
                         kernel_size=kernel_sizes, padding=kernel_sizes//2, stride=1))
        self.convs.append(scSE(channels=num_filters))
        for i in range(num_layer):
            self.convs.append(conv1DBatchNormMish(in_channels=num_filters, out_channels=num_filters,
                                                    kernel_size=kernel_sizes, padding=(kernel_sizes//2)*1, dilation=1))
            self.convs.append(conv1DBatchNormMish(in_channels=num_filters, out_channels=num_filters,
                                                    kernel_size=kernel_sizes, padding=(kernel_sizes//2)*3, dilation=3))
            self.convs.append(conv1DBatchNormMish(in_channels=num_filters, out_channels=num_filters,
                                                    kernel_size=kernel_sizes, padding=(kernel_sizes//2)*1, dilation=1))
            self.convs.append(conv1DBatchNormMish(in_channels=num_filters, out_channels=num_filters,
                                                    kernel_size=kernel_sizes, padding=(kernel_sizes//2)*5, dilation=5))



        self.convs.append(conv1DBatchNormMish(in_channels=num_filters, out_channels=num_filters, kernel_size=kernel_sizes, padding=kernel_sizes//2))
        self.convs.append(scSE(channels=num_filters))
        self.convs.append(conv1DBatchNorm(in_channels=num_filters, out_channels=1, kernel_size=5))

    def forward(self, x):
        x = self.embedding(x.long())
        x = torch.transpose(x, 1, 2)
        for i, l in enumerate(self.convs):
            x = l(x)

        x = x.view(x.shape[0], -1)
        return x

class dilation1(nn.Module):
    def __init__(self, emb_dim=128, num_layer=8, num_filters=32, kernel_sizes=5):
        super(dilation1, self).__init__()
        self.filter = num_filters
        self.embedding = nn.Embedding(6, emb_dim)

        self.convs = nn.ModuleList()
        self.convs.append(conv1DBatchNorm(in_channels=emb_dim, out_channels=num_filters,
                         kernel_size=kernel_sizes, padding=kernel_sizes//2, stride=1))
        self.convs.append(nn.Mish())
        # self.convs.append(scSE(channels=num_filters))
        for i in range(1, num_layer):
            self.convs.append(conv1DBatchNormRelu(in_channels=num_filters, out_channels=num_filters, 
                                                  kernel_size=kernel_sizes, padding=(kernel_sizes//2)*1, stride=1, dilation=1))
            self.convs.append(conv1DBatchNormRelu(in_channels=num_filters, out_channels=num_filters, 
                                                  kernel_size=kernel_sizes, padding=(kernel_sizes//2)*3, stride=1, dilation=3))
            self.convs.append(conv1DBatchNormRelu(in_channels=num_filters, out_channels=num_filters, 
                                                  kernel_size=kernel_sizes, padding=(kernel_sizes//2)*5, stride=1, dilation=5))
            self.convs.append(conv1DBatchNormRelu(in_channels=num_filters, out_channels=num_filters, 
                                                  kernel_size=kernel_sizes, padding=(kernel_sizes//2)*7, stride=1, dilation=7))

        self.convs.append(conv1DBatchNorm(in_channels=num_filters, out_channels=num_filters, kernel_size=kernel_sizes, padding=kernel_sizes//2))
        self.convs.append(nn.Mish())
        # self.convs.append(scSE(channels=num_filters))
        self.convs.append(conv1DBatchNorm(in_channels=num_filters, out_channels=1, kernel_size=5))        

    def forward(self, x):
        x = self.embedding(x.long())
        x = torch.transpose(x, 1, 2)
        for i, l in enumerate(self.convs):
            x = l(x)

#デコード
        x = x.view(x.shape[0], -1)
        return x



class conv1DBatchNormRelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1):
        super(conv1DBatchNormRelu, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias=False)
        self.batchnorm = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.batchnorm(x)
        output = self.relu(x)

        return output

class conv1DBatchNormMish(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1):
        super(conv1DBatchNormMish, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias=False)
        self.batchnorm = nn.BatchNorm1d(out_channels)
        self.mish = nn.Mish(inplace=True)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.batchnorm(x)
        output = self.mish(x)

        return output

class Tconv1DBatchNormRelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1):
        super(Tconv1DBatchNormRelu, self).__init__()
        self.conv = nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias=False)
        self.batchnorm = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.batchnorm(x)
        output = self.relu(x)

        return output

class conv1DBatchNorm(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1):
        super(conv1DBatchNorm, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias=False)
        self.batchnorm = nn.BatchNorm1d(out_channels)
        
    def forward(self, x):
        x = self.conv(x)
        output = self.batchnorm(x)

        return output


class DSconv1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, SE=False):
        super(DSconv1D, self).__init__()
        self.depthwise = conv1DBatchNorm(in_channels, in_channels, kernel_size, stride, padding, dilation, groups=in_channels)
        self.pointwise = conv1DBatchNorm(in_channels, out_channels, kernel_size=1, padding=0)
        if (SE==True):
            self.activate = nn.Hardswish(inplace=True)
        else:
            self.activate = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.activate(self.pointwise(self.depthwise(x)))

        return x


class InvertedBottleNeck(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0, dilation=1, expand_ratio=6, SE=False):
        super(InvertedBottleNeck, self).__init__()
        
        mid_channels = in_channels*expand_ratio
        self.pw1 = conv1DBatchNormRelu(in_channels, mid_channels, kernel_size=1, padding=0)
        self.dw = conv1DBatchNorm(mid_channels, mid_channels, kernel_size, padding=padding, groups=mid_channels)
        self.se = scSE(mid_channels)
        self.pw2 = conv1DBatchNorm(mid_channels, out_channels, kernel_size=1, padding=0)
        
    def forward(self, x):
        conv = self.dw(self.pw1(x))
        if (self.SE==True):
            conv = self.se(conv)
        conv = self.pw2(conv)
        return F.relu(x + conv)


class scSE(nn.Module):
    def __init__(self, channels, reduction=4):
        super(scSE, self).__init__()
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(channels, channels//reduction, bias=False)
        self.fc2 = nn.Linear(channels//reduction, channels, bias=False)
        
        self.conv = nn.Conv1d(channels, 1, kernel_size=1)

        self.sig = nn.Sigmoid()

    def forward(self, x):
        batch, channel, _ = x.size()
        c = self.gap(x).view(batch, channel)
        c = self.sig(self.fc2(F.relu(self.fc1(c)))).view(batch, channel, 1)
        c = x * c
        
        s = self.sig(self.conv(x))
        s = x * s
        return c + s

# He重みの初期化
def weight_init(m):
    if isinstance(m, nn.Conv1d):
        nn.init.kaiming_normal_(m.weight.data)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)