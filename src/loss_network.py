from collections import namedtuple

import torch
from torch import nn
from torchvision import models

"""
class VGG16(nn.Module):
    def __init__(self, requires_grad=False):
        super(VGG16, self).__init__()

        vgg_pretrained_features = models.vgg16(pretrained=True).features
        self.slice1 = nn.Sequential()
        self.slice2 = nn.Sequential()
        self.slice3 = nn.Sequential()
        self.slice4 = nn.Sequential()
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h = self.slice1(X)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        vgg_outputs = namedtuple(
            'VGGOutputs', ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3'])
        out = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3)
        return out
"""

"""
class VGG19(nn.Module):
    def __init__(self, requires_grad=False):
        super(VGG19, self).__init__()

        self.features = models.vgg19(pretrained=True).features
        self.layers = {'1': 'relu1_1', '6': 'relu2_1',
                       '11': 'relu3_1', '20': 'relu4_1', '29': 'relu5_1'}
        # Gets largest layer number
        self.largest_layer = max([int(key) for key in self.layers.keys()])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        features = {}
        for name, layer in self.features._modules.items():
            X = layer(X)
            if name in self.layers:
                features[self.layers[name]] = X
                if(name == self.largest_layer):
                    break
        return features
"""

class VGG16(nn.Module):
    def __init__(self, requires_grad=False):
        super(VGG16, self).__init__()

        self.features = models.vgg16(pretrained=True).features
        self.layers = {'3': 'relu1_2', '8': 'relu2_2',
                       '15': 'relu3_3', '22': 'relu4_3'}
        # Gets largest layer number
        self.largest_layer = max([int(key) for key in self.layers.keys()])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        features = {}
        for name, layer in self.features._modules.items():
            X = layer(X)
            if name in self.layers:
                features[self.layers[name]] = X
                if(name == self.largest_layer):
                    break
        return features


class TVLoss(nn.Module):
    """Adapted from https://github.com/jxgu1016/Total_Variation_Loss.pytorch"""

    def __init__(self, TVLoss_weight=1):
        super(TVLoss, self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:, :, 1:, :])
        count_w = self._tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :]-x[:, :, :h_x-1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:]-x[:, :, :, :w_x-1]), 2).sum()
        return self.TVLoss_weight*2*(h_tv/count_h+w_tv/count_w)/batch_size

    def _tensor_size(self, t):
        return t.size()[1]*t.size()[2]*t.size()[3]
