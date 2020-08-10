import torch
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


def show_image(tensor, title=None):
    """ Plots a tensor image using pyplot
    Args:
        tensor: A tensor image of shape (1,ch,h,w)
        title: (optional) Title of the plot
    """
    image = tensor.clone()
    image = image.squeeze(0)
    image = transforms.ToPILImage()(image)
    plt.imshow(image)
    if title:
        plt.title(title)
    plt.pause(0.01)
