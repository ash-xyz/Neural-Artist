import tensorflow as tf
from .hyperparameters import CONTENT_LAYERS, STYLE_LAYERS
"""
File that contains the loss and style networks
"""


def loss_network():
    """Generates a loss network

    Returns:
        model: keras functional api based on VGG19 with content and style layers outputs
    """
    cnn = tf.keras.applications.VGG19(
        include_top=False, weights='imagenet')
    layers = CONTENT_LAYERS+STYLE_LAYERS
    outputs = [cnn.get_layer(name).output for name in layers]
    model = tf.keras.Model(inputs=cnn.input, outputs=outputs)
    return model
