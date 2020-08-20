import torch
from torchvision import transforms
from PIL import Image


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


def load_image(image_path, batch_size=1):
    """Loads an image
    Args:
        image_path: location and name of the image
        batch_size: Creates a batch images consisting of the same image, defaults to 1 with a shape 1,C,H,W
    Returns:
        image: A Tensor of size B,C,H,W
    """
    image = Image.open(image_path)
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Lambda(lambda x: x.mul(255))])
    image = transform(image)
    image = image.repeat(batch_size, 1, 1, 1)
    return image


def save_image(out, image_path):
    """Turns a tensor into an image and saves it
    Args:
        out: tensor of shape: B, C, H, W
        image_path: directory and name of the image with an appropriate file extension(e.g .jpg or .png)
    """
    img = out[0].clone().clamp(0, 255).numpy()
    img = img.transpose(1, 2, 0).astype('uint8')
    img = Image.fromarray(img)
    img.save(image_path)


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
