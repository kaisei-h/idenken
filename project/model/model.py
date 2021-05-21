#!/usr/bin/env python
# coding: utf-8

import torch
import math
import torch.nn as nn
import torch.nn.functional as F


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


class Fixed(nn.Module):
    def __init__(self, emb_dim=5, num_layer=8, mid_units=400, num_filters=128, kernel_sizes=7, length=256):
        super(Fixed, self).__init__()
        self.filter = num_filters
        self.max_length = length
        self.embedding = nn.Embedding(5, emb_dim)

        self.convs = nn.ModuleList()
        for i in range(num_layer):
            if (i==0):
                self.convs.append(conv1DBatchNormRelu(in_channels=emb_dim, out_channels=num_filters,
                                                      kernel_size=kernel_sizes, padding=kernel_sizes//2))
            else:
                self.convs.append(conv1DBatchNormRelu(in_channels=num_filters, out_channels=num_filters,
                                                      kernel_size=kernel_sizes, padding=kernel_sizes//2))
        
#線形層
        self.mid = nn.Linear(self.max_length*num_filters, mid_units)
        self.fc = nn.Linear(mid_units, self.max_length-4)

    def forward(self, x):
        x = self.embedding(x.long())
        x = torch.transpose(x, 1, 2)
        for i, l in enumerate(self.convs):
            x = l(x)

#線形層
        x = x.view(-1, self.max_length*self.filter)
        x = F.relu(self.mid(x))
        x = self.fc(x)
        
        return x

class Variable(nn.Module):
    def __init__(self, emb_dim=5, num_layer=8, num_filters=128, kernel_sizes=7):
        super(Variable, self).__init__()
        self.filter = num_filters
        self.embedding = nn.Embedding(5, emb_dim)

        self.convs = nn.ModuleList()
        for i in range(num_layer):
            if (i==0):
                self.convs.append(conv1DBatchNormRelu(in_channels=emb_dim, out_channels=num_filters,
                                                      kernel_size=kernel_sizes, padding=kernel_sizes//2))
            else:
                self.convs.append(conv1DBatchNormRelu(in_channels=num_filters, out_channels=num_filters,
                                                      kernel_size=kernel_sizes, padding=kernel_sizes//2))
        
        self.fc = nn.Conv1d(in_channels=num_filters, out_channels=1, kernel_size=5)

    def forward(self, x):
        x = self.embedding(x.long())
        x = torch.transpose(x, 1, 2)
        for i, l in enumerate(self.convs):
            x = l(x)

#デコード
        x = self.fc(x)
        x = x.view(x.shape[0], -1)
        return x

class new_Variable(nn.Module):
    def __init__(self, emb_dim=5, num_layer=8, num_filters=64, kernel_sizes=7):
        super(new_Variable, self).__init__()
        self.filter = num_filters
        self.embedding = nn.Embedding(5, emb_dim)

        self.convs = nn.ModuleList()
        self.convs.append(conv1DBatchNormRelu(in_channels=emb_dim, out_channels=num_filters,
                         kernel_size=kernel_sizes, padding=kernel_sizes//2, stride=1))
        for i in range(1, num_layer):
            self.convs.append(conv1DBatchNormRelu(in_channels=num_filters*i, out_channels=num_filters*(i+1), 
                                                  kernel_size=kernel_sizes, padding=kernel_sizes//2, stride=1))
            self.convs.append(conv1DBatchNorm(in_channels=num_filters*(i+1), out_channels=num_filters*(i+1), 
                                                  kernel_size=kernel_sizes, padding=kernel_sizes//2, stride=1))
            self.convs.append(nn.MaxPool1d(kernel_size=2))

        for i in range(num_layer, 1, -1):
            self.convs.append(Tconv1DBatchNormRelu(in_channels=num_filters*i, out_channels=num_filters*(i-1), 
                                                  kernel_size=kernel_sizes, padding=kernel_sizes//2, stride=2))
        #     self.convs.append(conv1DBatchNorm(in_channels=num_filters*(i-1), out_channels=num_filters*(i-1), 
        #                                           kernel_size=kernel_sizes, padding=kernel_sizes//2, stride=1))
        
        self.fc = nn.Conv1d(in_channels=num_filters, out_channels=1, kernel_size=5)

    def forward(self, x):
        x = self.embedding(x.long())
        x = torch.transpose(x, 1, 2)
        for i, l in enumerate(self.convs):
            x = l(x)

#デコード
        x = self.fc(x)
        x = x.view(x.shape[0], -1)
        return x


class new_Variable_2(nn.Module):
    def __init__(self, emb_dim=5, num_layer=8, num_filters=64, kernel_sizes=7):
        super(new_Variable_2, self).__init__()
        self.filter = num_filters
        self.embedding = nn.Embedding(5, emb_dim)

        self.convs = nn.ModuleList()
        self.convs.append(conv1DBatchNormRelu(in_channels=emb_dim, out_channels=num_filters,
                         kernel_size=kernel_sizes, padding=kernel_sizes//2, stride=1))
        for i in range(1, num_layer):
            self.convs.append(conv1DBatchNormRelu(in_channels=num_filters*2**(i-1), out_channels=num_filters*2**i, 
                                                  kernel_size=kernel_sizes, padding=kernel_sizes//2, stride=1))
            self.convs.append(conv1DBatchNorm(in_channels=num_filters*2**i, out_channels=num_filters*2**i, 
                                                  kernel_size=kernel_sizes, padding=kernel_sizes//2, stride=1))
            self.convs.append(nn.AvgPool1d(2))

        for i in range(num_layer, 1, -1):
            self.convs.append(nn.Upsample(scale_factor=2, mode='linear'))
            self.convs.append(conv1DBatchNormRelu(in_channels=num_filters*2**(i-1), out_channels=num_filters*2**(i-2), 
                                                  kernel_size=kernel_sizes, padding=kernel_sizes//2, stride=1))
        #     self.convs.append(conv1DBatchNorm(in_channels=num_filters*(i-1), out_channels=num_filters*(i-1), 
        #                                           kernel_size=kernel_sizes, padding=kernel_sizes//2, stride=1))
        
        self.fc = nn.Conv1d(in_channels=num_filters, out_channels=1, kernel_size=5)

    def forward(self, x):
        x = self.embedding(x.long())
        x = torch.transpose(x, 1, 2)
        for i, l in enumerate(self.convs):
            x = l(x)

#デコード
        x = self.fc(x)
        x = x.view(x.shape[0], -1)
        return x


class new_Variable_3(nn.Module):
    def __init__(self, emb_dim=5, num_layer=8, num_filters=64, kernel_sizes=7):
        super(new_Variable_3, self).__init__()
        self.filter = num_filters
        self.embedding = nn.Embedding(5, emb_dim)

        self.convs = nn.ModuleList()
        self.convs.append(conv1DBatchNormRelu(in_channels=emb_dim, out_channels=16,
                         kernel_size=kernel_sizes, padding=kernel_sizes//2, stride=1))
        
        self.convs.append(conv1DBatchNormRelu(in_channels=16, out_channels=16, 
                                                kernel_size=kernel_sizes, padding=kernel_sizes//2, stride=1))
        self.convs.append(conv1DBatchNormRelu(in_channels=16, out_channels=32, 
                                                kernel_size=kernel_sizes, padding=kernel_sizes//2, stride=1))
        self.convs.append(nn.MaxPool1d(2))
        self.convs.append(conv1DBatchNormRelu(in_channels=32, out_channels=32, 
                                                kernel_size=kernel_sizes, padding=kernel_sizes//2, stride=1))
        self.convs.append(conv1DBatchNormRelu(in_channels=32, out_channels=64, 
                                                kernel_size=kernel_sizes, padding=kernel_sizes//2, stride=1))
        self.convs.append(nn.MaxPool1d(2))
        self.convs.append(conv1DBatchNormRelu(in_channels=64, out_channels=64, 
                                                kernel_size=kernel_sizes, padding=kernel_sizes//2, stride=1))
        self.convs.append(conv1DBatchNormRelu(in_channels=64, out_channels=128, 
                                                kernel_size=kernel_sizes, padding=kernel_sizes//2, stride=1))
        self.convs.append(nn.MaxPool1d(2))
        self.convs.append(conv1DBatchNormRelu(in_channels=128, out_channels=128, 
                                                kernel_size=kernel_sizes, padding=kernel_sizes//2, stride=1))
        self.convs.append(conv1DBatchNormRelu(in_channels=128, out_channels=128, 
                                                kernel_size=kernel_sizes, padding=kernel_sizes//2, stride=1))
        self.convs.append(conv1DBatchNormRelu(in_channels=128, out_channels=256, 
                                                kernel_size=kernel_sizes, padding=kernel_sizes//2, stride=1))
        self.convs.append(nn.MaxPool1d(2))
        self.convs.append(conv1DBatchNormRelu(in_channels=256, out_channels=256, 
                                                kernel_size=kernel_sizes, padding=kernel_sizes//2, stride=1))
        self.convs.append(conv1DBatchNormRelu(in_channels=256, out_channels=256, 
                                                kernel_size=kernel_sizes, padding=kernel_sizes//2, stride=1))
        self.convs.append(conv1DBatchNormRelu(in_channels=256, out_channels=512, 
                                                kernel_size=kernel_sizes, padding=kernel_sizes//2, stride=1))
        self.convs.append(nn.MaxPool1d(2))
        self.convs.append(conv1DBatchNormRelu(in_channels=512, out_channels=512, 
                                                kernel_size=kernel_sizes, padding=kernel_sizes//2, stride=1))
        self.convs.append(conv1DBatchNormRelu(in_channels=512, out_channels=512, 
                                                kernel_size=kernel_sizes, padding=kernel_sizes//2, stride=1))
        self.convs.append(conv1DBatchNormRelu(in_channels=512, out_channels=1024, 
                                                kernel_size=kernel_sizes, padding=kernel_sizes//2, stride=1))
        self.convs.append(nn.MaxPool1d(2))
        self.convs.append(conv1DBatchNormRelu(in_channels=1024, out_channels=1024, 
                                                kernel_size=kernel_sizes, padding=kernel_sizes//2, stride=1))
        self.convs.append(conv1DBatchNormRelu(in_channels=1024, out_channels=1024, 
                                                kernel_size=kernel_sizes, padding=kernel_sizes//2, stride=1))
        self.convs.append(conv1DBatchNormRelu(in_channels=1024, out_channels=2048, 
                                                kernel_size=kernel_sizes, padding=kernel_sizes//2, stride=1))
        self.convs.append(nn.MaxPool1d(2))


        self.convs.append(nn.Upsample(scale_factor=2, mode='linear'))
        self.convs.append(conv1DBatchNormRelu(in_channels=2048, out_channels=1024, 
                                                kernel_size=kernel_sizes, padding=kernel_sizes//2, stride=1))
        self.convs.append(conv1DBatchNormRelu(in_channels=1024, out_channels=1024, 
                                                kernel_size=kernel_sizes, padding=kernel_sizes//2, stride=1))
        self.convs.append(nn.Upsample(scale_factor=2, mode='linear'))
        self.convs.append(conv1DBatchNormRelu(in_channels=1024, out_channels=512, 
                                                kernel_size=kernel_sizes, padding=kernel_sizes//2, stride=1))
        self.convs.append(conv1DBatchNormRelu(in_channels=512, out_channels=512, 
                                                kernel_size=kernel_sizes, padding=kernel_sizes//2, stride=1))
        self.convs.append(nn.Upsample(scale_factor=2, mode='linear'))
        self.convs.append(conv1DBatchNormRelu(in_channels=512, out_channels=256, 
                                                kernel_size=kernel_sizes, padding=kernel_sizes//2, stride=1))
        self.convs.append(conv1DBatchNormRelu(in_channels=256, out_channels=256, 
                                                kernel_size=kernel_sizes, padding=kernel_sizes//2, stride=1))
        self.convs.append(nn.Upsample(scale_factor=2, mode='linear'))
        self.convs.append(conv1DBatchNormRelu(in_channels=256, out_channels=128, 
                                                kernel_size=kernel_sizes, padding=kernel_sizes//2, stride=1))
        self.convs.append(conv1DBatchNormRelu(in_channels=128, out_channels=128, 
                                                kernel_size=kernel_sizes, padding=kernel_sizes//2, stride=1))
        self.convs.append(nn.Upsample(scale_factor=2, mode='linear'))
        self.convs.append(conv1DBatchNormRelu(in_channels=128, out_channels=64, 
                                                kernel_size=kernel_sizes, padding=kernel_sizes//2, stride=1))
        self.convs.append(nn.Upsample(scale_factor=2, mode='linear'))
        self.convs.append(conv1DBatchNormRelu(in_channels=64, out_channels=32, 
                                                kernel_size=kernel_sizes, padding=kernel_sizes//2, stride=1))
        self.convs.append(nn.Upsample(scale_factor=2, mode='linear'))
        self.convs.append(conv1DBatchNormRelu(in_channels=32, out_channels=16, 
                                                kernel_size=kernel_sizes, padding=kernel_sizes//2, stride=1))

        self.fc = nn.Conv1d(in_channels=16, out_channels=1, kernel_size=5)
    def forward(self, x):
        x = self.embedding(x.long())
        x = torch.transpose(x, 1, 2)
        for i, l in enumerate(self.convs):
            x = l(x)

#デコード
        x = self.fc(x)
        x = x.view(x.shape[0], -1)
        return x



# class SENet(nn.Module):
#     def __init__(self):
#         super(SENet, self).__init__()

#         self.avg_pool = nn.AdaptiveAvgPool1d(1)
#         self.fc = nn.Sequential(
#             nn.Linear(channel, channel//reduction, bias=False),
#             nn.ReLU(inplace=True),
#             nn.Linear(channel//reduction, channel, bias=False),
#             nn.Sigmoid()
#         )

#     def forward(self, x):
#         b, c, _ = x.size()
#         y = self.avg_pool(x).view(b)
#         y = self.fc(y).view(b, c, 1)
#         return x * y.expand_as(x)

class PSPNet(nn.Module):
    def __init__(self):
        super(PSPNet, self).__init__()
        
        block_config = [1, 2, 4, 1]
        length = 256
        length_8 = 13
        emb_dim = 5
        n_classes = length-4
        self.embedding = nn.Embedding(5, emb_dim)
        
        self.feature_conv = FeatureMap_convolution(in_channels=emb_dim, out_channels=128) #1/4の長さ
        self.feature_res_1 = ResidualBlockPSP(n_blocks=block_config[0],
                                             in_channels=128, mid_channels=64, out_channels=256) #変化なし
        self.feature_res_2 = ResidualBlockPSP(n_blocks=block_config[1],
                                             in_channels=256, mid_channels=128, out_channels=512, stride=2) #半分の長さ
        self.feature_dilated_res_1 = ResidualBlockPSP(n_blocks=block_config[2],
                                             in_channels=512, mid_channels=256, out_channels=1024) #変化なし
        self.feature_dilated_res_2 = ResidualBlockPSP(n_blocks=block_config[3],
                                             in_channels=1024, mid_channels=512, out_channels=2048, dilation=3) #変化なし
        
        self.pyramid_pooling = PyramidPooling(in_channels=2048, pool_sizes=[6,3,2,1], length=length_8) #チャンネル倍
        
        self.decode_feature = DecodePSPFeature(in_channels=4096, length=1, n_classes=n_classes)
        
        self.aux = AuxiliaryPSPlayers(in_channels=1024, length=1, n_classes=n_classes)
        
    def forward(self, x):
        x = self.embedding(x.long())
        x = torch.transpose(x, 1, 2)
        x = self.feature_conv(x)
        x = self.feature_res_1(x)
        x = self.feature_res_2(x)
        x = self.feature_dilated_res_1(x)
        output_aux = self.aux(x)
        output_aux = output_aux.view(output_aux.shape[0],-1)
        x = self.feature_dilated_res_2(x)
        x = self.pyramid_pooling(x)
        output = self.decode_feature(x)
        output = F.relu(output.view(output.shape[0],-1))
        return (output, output_aux)


class conv1DBatchNormRelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1):
        super(conv1DBatchNormRelu, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups)
        self.batchnorm = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.batchnorm(x)
        output = self.relu(x)

        return output

class Tconv1DBatchNormRelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1):
        super(Tconv1DBatchNormRelu, self).__init__()
        self.conv = nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups)
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
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups)
        self.batchnorm = nn.BatchNorm1d(out_channels)
        
    def forward(self, x):
        x = self.conv(x)
        output = self.batchnorm(x)

        return output


class DSconv1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1):
        super(DSconv1D, self).__init__()
        self.depthwise = nn.Conv1d(in_channels, in_channels, kernel_size, stride, padding, dilation, groups=in_channels)
        self.bn_d = nn.BatchNorm1d(in_channels)
        self.pointwise = nn.Conv1d(in_channels, out_channels, kernel_size=1, padding=0)
        self.bn_p = nn.BatchNorm1d(out_channels)
        self.relu_p = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.bn_d(self.depthwise(x))
        output = self.relu_p(self.bn_p(self.pointwise(x)))

        return output


class BottleNeck(nn.Module):
    def __init__(self, in_channels, mid_channels, kernel_size, stride=1, padding=0, dilation=1):
        super(bottleNeckIdentifyPSP, self).__init__()
        
        self.bn = nn.BatchNorm1d(in_channels)
        self.cbr_1 = conv1DBatchNormRelu(in_channels, mid_channels, kernel_size=1)
        self.cb_2 = conv1DBatchNorm(mid_channels, mid_channels, kernel_size=kernel_size, padding=padding)
        self.cb_3 = conv1DBatchNorm(mid_channels, in_channels, kernel_size=1)
        
    def forward(self, x):
        conv = self.cb_3(self.cb_2(self.cbr_1(self.nb(x))))
        residual = x
        return self.relu(conv + residual)


class InvertedBottleNeck(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1):
        super(InvertedBottleNeck, self).__init__()
        
        mid_channels = in_channels*6
        self.bn = nn.BatchNorm1d(in_channels)
        self.pw1 = conv1DBatchNormRelu(in_channels, mid_channels, kernel_size=1, padding=0)
        self.dw = conv1DBatchNorm(mid_channels, mid_channels, kernel_size, padding=padding, groups=mid_channels)
        self.pw2 = conv1DBatchNorm(mid_channels, out_channels, kernel_size=1, padding=0)
        
    def forward(self, x):
        conv = self.pw2(self.dw(self.pw1(self.bn(x))))
        residual = x
        return self.relu(conv + residual)


        
class FeatureMap_convolution(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FeatureMap_convolution, self).__init__()
        self.cbnr_1 = conv1DBatchNormRelu(in_channels=in_channels, out_channels=64, kernel_size=5, stride=2, padding=2)
        
        self.cbnr_2 = conv1DBatchNormRelu(in_channels=64, out_channels=64, kernel_size=5, padding=2)
        
        self.cbnr_3 = conv1DBatchNormRelu(in_channels=64, out_channels=out_channels, kernel_size=5, padding=2)
        
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        
    def forward(self, x):
        x = self.cbnr_1(x) #半分
        x = self.cbnr_2(x)
        x = self.cbnr_3(x)
        output = self.maxpool(x) #半分
        
        return output


class ResidualBlockPSP(nn.Sequential):
    def __init__(self, n_blocks, in_channels, mid_channels, out_channels, stride=1, dilation=1):
        super(ResidualBlockPSP, self).__init__()
        
        self.add_module('block',
                       bottleNeckPSP(in_channels, mid_channels, out_channels, stride, dilation))
        for i in range(n_blocks-1):
            self.add_module('block' + str(i+2),
                           bottleNeckIdentifyPSP(out_channels, mid_channels, stride, dilation))
            

class bottleNeckPSP(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, stride=1, dilation=1):
        super(bottleNeckPSP, self).__init__()
        
        self.cbr_1 = conv1DBatchNormRelu(in_channels, mid_channels, kernel_size=1)
        self.cbr_2 = conv1DBatchNormRelu(mid_channels, mid_channels,
                                         kernel_size=3, stride=stride, padding=dilation, dilation=dilation)
        self.cb_3 = conv1DBatchNorm(mid_channels, out_channels, kernel_size=1)
        
        self.cb_residual = conv1DBatchNorm(in_channels, out_channels,
                                           kernel_size=1, stride=stride)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        conv = self.cb_3(self.cbr_2(self.cbr_1(x)))
        residual = self.cb_residual(x)
        return self.relu(conv + residual)


class bottleNeckIdentifyPSP(nn.Module):
    def __init__(self, in_channels, mid_channels, stride=1, dilation=1):
        super(bottleNeckIdentifyPSP, self).__init__()
        
        self.cbr_1 = conv1DBatchNormRelu(in_channels, mid_channels, kernel_size=1)
        self.cbr_2 = conv1DBatchNormRelu(mid_channels, mid_channels,
                                         kernel_size=3, padding=dilation, dilation=dilation)
        self.cb_3 = conv1DBatchNorm(mid_channels, in_channels, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        conv = self.cb_3(self.cbr_2(self.cbr_1(x)))
        residual = x
        return self.relu(conv + residual)


class PyramidPooling(nn.Module):
    def __init__(self, in_channels, pool_sizes:list, length):
        super(PyramidPooling, self).__init__()
        
        self.length = length
        out_channels = int(in_channels / len(pool_sizes))
        module_list = []
        
        for i in pool_sizes:
            module_list.append(nn.AdaptiveAvgPool1d(output_size=i))
            module_list.append(conv1DBatchNormRelu(in_channels, out_channels, kernel_size=1))
        
        self.pools = nn.ModuleList(module_list)
        
    def forward(self, x):
        out_list = []
        out_list.append(x)
        for i, l in enumerate(self.pools):
            out = l(x)
            if i%2==1:
                out = F.interpolate(out, size=self.length, mode='linear', align_corners=True)
                out_list.append(out)
        output = torch.cat(out_list, dim=1)
        return output
    

class DecodePSPFeature(nn.Module):
    def __init__(self, in_channels, length, n_classes):
        super(DecodePSPFeature, self).__init__()
        
        self.length = length
        
        self.cbr = conv1DBatchNormRelu(in_channels=in_channels, out_channels=1024, kernel_size=5, padding=2)
        self.dropout = nn.Dropout(p=0.1)
        self.regression = nn.Conv1d(in_channels=1024, out_channels=n_classes, kernel_size=1)
        
    def forward(self, x):
        x = self.cbr(x)
        x = self.dropout(x)
        x = self.regression(x)
        output = F.interpolate(x, size=self.length, mode='linear', align_corners=True)
        return output
    

class AuxiliaryPSPlayers(nn.Module):
    def __init__(self, in_channels, length, n_classes):
        super(AuxiliaryPSPlayers, self).__init__()
        
        self.length = length
        
        self.cbr = conv1DBatchNormRelu(in_channels=in_channels, out_channels=256, kernel_size=3, padding=1)
        self.dropout = nn.Dropout(p=0.1)
        self.regression = nn.Conv1d(in_channels=256, out_channels=n_classes, kernel_size=1)
        
    def forward(self, x):
        x = self.cbr(x)
        x = self.dropout(x)
        x = self.regression(x)
        output = F.interpolate(x, size=self.length, mode='linear', align_corners=True)
        
        return output


class Self_Attention(nn.Module):
    def __init__(self, in_dim):
        super(Self_Attention, self).__init__()
        
        self.query_conv = nn.Conv1d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = nn.Conv1d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = nn.Conv1d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        
        self.softmax = nn.Softmax(dim=-1)
        
        self.gamma = nn.Parameter(torch.zeros(1))
        
    def forward(self, x):
        X = x
        
        proj_query = self.query_conv(X).permute(0, 2, 1)
        proj_key = self.key_conv(X)
        S = torch.bmm(proj_query, proj_key)

        attention_map = self.softmax(S)
        attention_map = attention_map.permute(0, 2, 1)

        proj_value = self.value_conv(X)
        o = torch.bmm(proj_value, attention_map.permute(0, 2, 1))

        o = o.view(X.shape[0], X.shape[1], X.shape[2])
        
        out = x + self.gamma*o
        return out, attention_map


# He重みの初期化
def weight_init(m):
    if isinstance(m, nn.Conv1d):
        nn.init.kaiming_normal_(m.weight.data)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)


class PSPLoss(nn.Module):
    # PSPNetの損失関数
    def __init__(self, aux_weight=0.4):
        super(PSPLoss, self).__init__()
        self.aux_weight = aux_weight
        
    def forward(self, outputs, targets):
        loss = F.mse_loss(outputs[0], targets, reduction='mean')
        loss_aux = F.mse_loss(outputs[1], targets, reduction='mean')
        
        return loss+self.aux_weight*loss_aux