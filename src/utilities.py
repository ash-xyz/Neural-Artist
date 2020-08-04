import torch
from PIL import Image


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
