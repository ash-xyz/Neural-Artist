import torchvision.models as models
import torch
from torchvision.transforms import transforms


class LossNet(torch.nn.Module):
    def __init__(self, style_layers=[], content_layers=[], requires_grad=False):
        # Inspired by https://github.com/pytorch/examples/tree/master/fast_neural_style/vgg.py Changes: Replaced VGG16 with 19, replaced relu with conv, allow you to choose your output layers
        super(LossNet, self).__init__()
        vgg = models.vgg19(pretrained=True).features.eval()

        self.style_layers = style_layers
        self.content_layers = content_layers

        norm = Normalization()
        self.layers = ['conv1_1', 'conv1_2', 'conv2_1', 'conv2_2', 'conv3_1',
                       'conv3_2', 'conv3_3', 'conv3_4', 'conv4_1', 'conv4_2', 'conv4_3', 'conv4_4']

        # if you use a regular dict, it won't be properly registered
        self.model = torch.nn.ModuleDict({})
        self.model[self.layers[0]] = torch.nn.Sequential(norm)
        self.model[self.layers[1]] = torch.nn.Sequential()

        self.model[self.layers[2]] = torch.nn.Sequential()
        self.model[self.layers[3]] = torch.nn.Sequential()

        self.model[self.layers[4]] = torch.nn.Sequential()
        self.model[self.layers[5]] = torch.nn.Sequential()
        self.model[self.layers[6]] = torch.nn.Sequential()
        self.model[self.layers[7]] = torch.nn.Sequential()

        self.model[self.layers[8]] = torch.nn.Sequential()
        self.model[self.layers[9]] = torch.nn.Sequential()
        self.model[self.layers[10]] = torch.nn.Sequential()
        self.model[self.layers[11]] = torch.nn.Sequential()

        # Block 1
        self.model[self.layers[0]].add_module(str(0), vgg[0])
        for x in range(1, 3):
            self.model[self.layers[1]].add_module(str(x), vgg[x])

        # Block 2
        for x in range(3, 6):
            self.model[self.layers[2]].add_module(str(x), vgg[x])
        for x in range(6, 8):
            self.model[self.layers[3]].add_module(str(x), vgg[x])

        # Block 3
        for x in range(8, 11):
            self.model[self.layers[4]].add_module(str(x), vgg[x])
        for x in range(11, 13):
            self.model[self.layers[5]].add_module(str(x), vgg[x])
        for x in range(13, 15):
            self.model[self.layers[6]].add_module(str(x), vgg[x])
        for x in range(15, 17):
            self.model[self.layers[7]].add_module(str(x), vgg[x])

        # Block 4
        for x in range(17, 20):
            self.model[self.layers[8]].add_module(str(x), vgg[x])
        for x in range(20, 22):
            self.model[self.layers[9]].add_module(str(x), vgg[x])
        for x in range(22, 24):
            self.model[self.layers[10]].add_module(str(x), vgg[x])
        for x in range(24, 26):
            self.model[self.layers[11]].add_module(str(x), vgg[x])

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, input_image, request_style=True, request_content=True):
        """Does a forward pass through the VGG network
        Args:
            input_image: image to be passed through the network
            request_style: Returns a style output if true
            request_content: Returns a content output if true
        Returns:
            style_outputs: A dictionary of the outputs of the style layers, empty if not requested
            content_outputs: A dictionary of the outputs of the content layers, empty if not requested
        """
        style_outputs = {}
        content_outputs = {}

        h = input_image
        # Loops through styles
        for i in range(len(self.layers)):
            h = self.model[self.layers[i]](h)
            if(request_style and self.layers[i] in self.style_layers):
                style_outputs[self.layers[i]] = h
            if(request_content and self.layers[i] in self.content_layers):
                content_outputs[self.layers[i]] = h

        return style_outputs, content_outputs


class Normalization(torch.nn.Module):
    #"""Normalizes a batch with imagenet means"""
    # Taken and adapted from the pytorch Neural Style Tutorial: https://pytorch.org/tutorials/advanced/neural_style_tutorial.html

    def __init__(self):
        super(Normalization, self).__init__()
        self.mean = torch.tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
        self.std = torch.tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)

    def forward(self, img):
        # Forward Pass that normalizes images
        return (img - self.mean) / self.std
