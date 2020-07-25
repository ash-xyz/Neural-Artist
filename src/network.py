import tensorflow as tf
"""
File that contains the loss and style networks
"""


def loss_network():
    """Generates a loss network

    Returns:
        model: keras functional api based on VGG19
        len_content: length of the content layers
        len_style: length of the style layers
    """
    cnn = tf.keras.applications.VGG19(
        include_top=False, weights='imagenet')
    content_layers = ['block3_conv2']
    style_layers = [
        'block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1'
    ]
    layers = content_layers+style_layers
    outputs = [cnn.get_layer(name).output for name in layers]
    model = tf.keras.Model(inputs=cnn.input, outputs=outputs)
    len_content = len(content_layers)
    len_styles = len(style_layers)
    return model, len_content, len_styles
