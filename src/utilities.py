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


def load_and_process_image(path_to_image):
    image = load_image(path_to_image)
    image = tf.keras.applications.vgg19.preprocess_input(image)
    return image


def deprocess_image(processed_image):
    x = processed_image.copy()
    if len(x.shape) == 4:
        x = np.squeeze(x, 0)
    assert len(x.shape) == 3, ("Input to deprocess image must be an image of "
                               "dimension [1, height, width, channel] or [height, width, channel]")
    if len(x.shape) != 3:
        raise ValueError("Invalid input to deprocessing image")

    # perform the inverse of the preprocessing step
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    x = x[:, :, ::-1]

    x = np.clip(x, 0, 255).astype('uint8')
    return x


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


class ContentLoss:
    def __init__(self, target):
        super().__init__()
        self.target = target

    def forward(self, input):
        self.loss = 0
        for i in range(len(input)):
            self.loss += tf.reduce_mean(
                tf.square(input[i]-self.target[i]))
        return self.loss


def gram_matrix(features):
    """Computes the Gram matrix from features

    Args:
        features: A list of Tensors of shape [1, H, W, C]

    Returns:
        Gram matrix: A list of Tensors of shape [C, C]
    """
    gram = []
    for i in range(len(features)):
        _, H, W, C = tf.shape(features[i])
        gram.append(tf.linalg.einsum(
            'bijc,bijd->bcd', features[i], features[i])/tf.cast(C, tf.float32))
    return gram


class StyleLoss:
    def __init__(self, target):
        super().__init__()
        gram = gram_matrix(target)
        self.target_gram = gram

    def forward(self, input):
        input_gram = gram_matrix(input)
        self.loss = 0
        for i in range(len(input_gram)):
            self.loss += tf.reduce_mean(
                tf.square(input_gram[i]-self.target_gram[i]))
        return self.loss/len(input)
