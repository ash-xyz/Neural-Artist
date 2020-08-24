import torch
from torch import nn
from torchvision import models

VGG19_LAYERS = {'0': 'conv1_1', '1': 'relu1_1',
                '2': 'conv1_2', '3': 'relu1_2',
                '5': 'conv2_1', '6': 'relu2_1',
                '7': 'conv2_2', '8': 'relu2_2',
                '10': 'conv3_1', '11': 'relu3_1',
                '12': 'conv3_2', '13': 'relu3_2',
                '14': 'conv3_3', '15': 'relu3_3',
                '16': 'conv3_4', '17': 'relu3_4',
                '19': 'conv4_1', '20': 'relu4_1',
                '21': 'conv4_2', '22': 'relu4_2',
                '23': 'conv4_3', '24': 'relu4_3',
                '25': 'conv4_4', '26': 'relu4_4',
                '28': 'conv5_1', '29': 'relu5_1'}

VGG16_LAYERS = {'0': 'conv1_1', '1': 'relu1_1',
                '2': 'conv1_2', '3': 'relu1_2',
                '5': 'conv2_1', '6': 'relu2_1',
                '7': 'conv2_2', '8': 'relu2_2',
                '10': 'conv3_1', '11': 'relu3_1',
                '12': 'conv3_2', '13': 'relu3_2',
                '14': 'conv3_3', '15': 'relu3_3',
                '17': 'conv4_1', '18': 'relu4_1',
                '19': 'conv4_2', '20': 'relu4_2',
                '21': 'conv4_3', '22': 'relu4_3',
                '24': 'conv5_1', '25': 'relu5_1',
                '26': 'conv5_2', '27': 'relu5_2'}


class VGG19(nn.Module):
    def __init__(self, style_layers, content_layer, requires_grad=False):
        super(VGG19, self).__init__()

        self.features = models.vgg19(pretrained=True).features

        # Gets largest layer number
        self.largest_layer = max([int(key) for key in VGG16_LAYERS.keys(
        ) if VGG19_LAYERS[key] in style_layers or VGG19_LAYERS[key] == content_layer])

        self.content_layer = content_layer
        self.style_layers = style_layers

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        style_features = {}
        content_feature = {}
        features = {'content': 'hello', 'style': {}}
        for name, layer in self.features._modules.items():
            X = layer(X)
            if name in VGG19_LAYERS and VGG19_LAYERS[name] in self.style_layers:
                style_features[VGG19_LAYERS[name]] = X
            if name in VGG19_LAYERS and VGG19_LAYERS[name] == self.content_layer:
                content_feature[VGG19_LAYERS[name]] = X
            if(int(name) == self.largest_layer):
                break
        return style_features, content_feature


class VGG16(nn.Module):
    def __init__(self, style_layers, content_layer, requires_grad=False):
        super(VGG16, self).__init__()

        self.features = models.vgg19(pretrained=True).features

        # Gets largest layer number
        self.largest_layer = max([int(key) for key in VGG16_LAYERS.keys(
        ) if VGG16_LAYERS[key] in style_layers or VGG16_LAYERS[key] == content_layer])

        self.content_layer = content_layer
        self.style_layers = style_layers

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        style_features = {}
        content_feature = {}
        for name, layer in self.features._modules.items():
            X = layer(X)
            if name in VGG16_LAYERS and VGG16_LAYERS[name] in self.style_layers:
                style_features[VGG16_LAYERS[name]] = X
            if name in VGG16_LAYERS and VGG16_LAYERS[name] == self.content_layer:
                content_feature[VGG16_LAYERS[name]] = X
            if(int(name) == self.largest_layer):
                break
        return style_features, content_feature


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
