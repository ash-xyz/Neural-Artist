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
        self.model = {}
        self.model['conv1_1'] = torch.nn.Sequential()
        self.model['conv1_2'] = torch.nn.Sequential()

        self.model['conv2_1'] = torch.nn.Sequential()
        self.model['conv2_2'] = torch.nn.Sequential()

        self.model['conv3_1'] = torch.nn.Sequential()
        self.model['conv3_2'] = torch.nn.Sequential()
        self.model['conv3_3'] = torch.nn.Sequential()
        self.model['conv3_4'] = torch.nn.Sequential()

        self.model['conv4_1'] = torch.nn.Sequential()
        self.model['conv4_2'] = torch.nn.Sequential()
        self.model['conv4_3'] = torch.nn.Sequential()
        self.model['conv4_4'] = torch.nn.Sequential()

        self.model['conv1_1'].add_module(str(0), vgg[0])
        for x in range(1, 3):
            self.model['conv1_2'].add_module(str(x), vgg[x])

        for x in range(3, 6):
            self.model['conv2_1'].add_module(str(x), vgg[x])
        for x in range(6, 8):
            self.model['conv2_2'].add_module(str(x), vgg[x])

        for x in range(8, 11):
            self.model['conv3_1'].add_module(str(x), vgg[x])
        for x in range(11, 13):
            self.model['conv3_2'].add_module(str(x), vgg[x])
        for x in range(13, 15):
            self.model['conv3_3'].add_module(str(x), vgg[x])
        for x in range(15, 17):
            self.model['conv3_4'].add_module(str(x), vgg[x])

        for x in range(17, 20):
            self.model['conv4_1'].add_module(str(x), vgg[x])
        for x in range(20, 22):
            self.model['conv4_2'].add_module(str(x), vgg[x])
        for x in range(22, 24):
            self.model['conv4_3'].add_module(str(x), vgg[x])
        for x in range(24, 26):
            self.model['conv4_4'].add_module(str(x), vgg[x])

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, input_image):
        # Image net normalization
        normalized_input = self.normalize(input_image)

        outputs = {}

        h = self.model['conv1_1'](normalized_input)
        outputs['conv1_1'] = h
        h = self.model['conv1_2'](h)
        outputs['conv1_2'] = h

        h = self.model['conv2_1'](h)
        outputs['conv2_1'] = h
        h = self.model['conv2_2'](h)
        outputs['conv2_2'] = h

        h = self.model['conv3_1'](h)
        outputs['conv3_1'] = h
        h = self.model['conv3_2'](h)
        outputs['conv3_2'] = h
        h = self.model['conv3_3'](h)
        outputs['conv3_3'] = h
        h = self.model['conv3_4'](h)
        outputs['conv3_4'] = h

        h = self.model['conv4_1'](h)
        outputs['conv4_1'] = h
        h = self.model['conv4_2'](h)
        outputs['conv4_2'] = h
        h = self.model['conv4_3'](h)
        outputs['conv4_3'] = h
        h = self.model['conv4_4'](h)
        outputs['conv4_4'] = h

        return outputs
