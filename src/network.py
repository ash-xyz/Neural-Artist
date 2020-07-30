import tensorflow as tf
"""
File that contains the loss and style networks
"""


def loss_network(content_layers, style_layers):
    """Generates a loss network based on vgg 19

    Args:
        content_layers: layers wanted for the content
        style_layers: layers wanted for the style

    Returns:
        model: keras functional api based on VGG19 with content and style layers outputs
    """
    cnn = tf.keras.applications.VGG19(
        include_top=False, weights='imagenet')
    layers = content_layers+style_layers
    outputs = [cnn.get_layer(name).output for name in layers]
    model = tf.keras.Model(inputs=cnn.input, outputs=outputs)
    return model
