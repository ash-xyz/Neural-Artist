import torch

def normalize_batch(batch):
    """Normalize batch using ImageNet mean and std
    Args:
        batch: batch of images, shape: b, ch, h, w
    Returns:
        Normalized batch
    """
    mean = batch.new_tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
    std = batch.new_tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
    batch = batch.div_(255.0)

    return (batch - mean) / std


def gram_matrix(tensor):
    """Computes the gram matrix for a tensor
    Args:
        tensor: A tensor of shape b, ch, h, w; Where b is the batch size
    Returns:
        gram: The Calculated gram matrix of shape b, ch, h*w
    """
    (batch, channel, height, width) = tensor.size()
    features = tensor.view(batch, channel, height * width)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t) / (channel * height * width)
    return gram