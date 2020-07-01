from utilities import *
import tensorflow as tf


def gram_matrix(features):
    """Computes the Gram matrix from features

    Args:
        features: Tensor of shape [1, H, W, C]

    Returns:
        Gram matrix: Tensor of shape [C, C]
    """
    _, H, W, C = tf.shape(features)
    gram = tf.linalg.einsum('bijc,bijd->bcd', features, features)
    return gram/tf.cast(H*W, tf.float32)


def loss_network():
    """Generates a loss network

    Returns:
        model: keras Model based on VGG 19 that returns multiple outputs
        divider: numerical divider for the output list, [:divider] are the content layers, [divider:] are the style layers
        weights: Weights for each layer in style
    """
    cnn = tf.keras.applications.VGG19(
        include_top=False, weights='imagenet', pooling="avg")
    content_layers = ['block4_conv2']
    style_layers = [
        'block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1'
    ]
    weights = [0.2, 0.2, 0.2, 0.2, 0.2]
    layers = content_layers+style_layers
    outputs = [cnn.get_layer(name).output for name in layers]
    model = tf.keras.Model(inputs=cnn.input, outputs=outputs)
    divider = len(content_layers)
    return model, divider, weights