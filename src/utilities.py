import tensorflow as tf
from PIL import Image
import numpy as np


def load_image(image_path):
    """Loads images into a tensor model

    Args:
        image_path: Path of the image to be loaded

    Returns:
        Tensor of the image
    """
    image = tf.io.read_file(image_path)
    image = tf.image.decode_image(image, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    return image


def save_image(tensor, name):
    """Takes the tensor and returns an image

    Args:
        tensor:
        name: name of the image to store

    Returns:
        Tensor of the image
    """
    print(tensor)
    tensor = tensor*255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor) > 3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    tensor_to_image(PIL.Image.fromarray(tensor)).save(save_path)


a = load_image('test.png')
save_image(a, 'result.png')
