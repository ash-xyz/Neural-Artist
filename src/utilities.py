import torch
from PIL import Image
import torchvision.transforms as transforms


def load_image(image_path):
    """Load a PIL image 
    Args:
        image_path: path to the image
    Returns:
        image: An image tensor
    """
    image = Image.open(image_path)
    loader = transforms.Compose([
        transforms.ToTensor()])
    image = loader(image).unsqueeze(0)
    return image


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
