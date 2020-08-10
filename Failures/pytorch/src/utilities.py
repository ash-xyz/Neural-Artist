import torch
from PIL import Image
import torchvision.transforms as transforms
import numpy


def load_image(image_path, size=None, scale=None, unsqueeze=True):
    """Load an image into a pytorch tensor
    Args:
        image_path: path to the image
        size: Resizes the image
        scale: scales the image
        unsqueeze: Add a 4th dimension, required for feeding it into a network
    Returns:
        image: An image tensor
    """
    img = Image.open(image_path)
    if size is not None:
        img = img.resize((size, size), Image.ANTIALIAS)
    elif scale is not None:
        img = img.resize(
            (int(img.size[0] / scale), int(img.size[1] / scale)), Image.ANTIALIAS)
    loader = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])
    if unsqueeze == True:
        img = loader(img).unsqueeze(0)
    else:
        img = loader(img)
    return img


def save_image(path, image):
    img = image[0].clone().clamp(0, 255).numpy()
    img = img.transpose(1, 2, 0).astype("uint8")
    img = Image.fromarray(img)
    img.save(path)


def gram_matrix(tensor):
    """Computes the gram matrix for a tensor
    Args:
        tensor: A tensor of shape b, ch, h, w; Where b is the batch size
    Returns:
        gram: The Calculated gram matrix of shape b, ch, h*w
    """
    (b, ch, h, w) = tensor.size()
    features = tensor.reshape(b, ch, h*w)
    gram = features.bmm(features.transpose(1, 2))/(ch*h*w)
    return gram
