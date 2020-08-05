import torchvision.models as models
import torch
from torchvision.transforms import transforms


class LossNet(torch.nn.Module):
    def __init__(self, requires_grad=False):
        # Inspired by https://github.com/pytorch/examples/tree/master/fast_neural_style/vgg.py Changes: Replaced VGG16 with 19, replaced relu with conv
        super(LossNet, self).__init__()
        vgg = models.vgg19(pretrained=True).features
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        for x in range(3):
            self.slice1.add_module(str(x), vgg[x])
        for x in range(3, 7):
            self.slice2.add_module(str(x), vgg[x])
        for x in range(9, 15):
            self.slice3.add_module(str(x), vgg[x])
        for x in range(15, 36):
            self.slice4.add_module(str(x), vgg[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, input_image):
        # Image net normalization
        normalized_input = self.normalize(input_image)
