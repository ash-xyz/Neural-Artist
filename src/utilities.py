import tensorflow as tf
from PIL import Image
import numpy as np


def load_image(image_path):
    """Loads and turns an image into a tensor

    Args:
        image_path: Path of the image to be loaded

    Returns:
        Tensor image of shape [1,H,W,3]
    """
    image = tf.io.read_file(image_path)
    image = tf.image.decode_image(image, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.expand_dims(image, 0)
    return image


def save_image(tensor, name):
    """Turns a tensor into an image

    Args:
        tensor: A tensor to be turned into an image
        name: The name of the file

    Returns:
        An image in the source directory
    """
    tensor = tensor*255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor) > 3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return Image.fromarray(tensor).save(name)
