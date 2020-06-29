from utilities import *
import tensorflow as tf

# TODO: Generalize to include lists


def compute_content_loss(content, pastiche):
    """ Computes the loss of the pastiche with respect to the content

    Args:
        content: features of the content image, Tensor of shape [1, H, W, C]
        pastiche: features of the pastische image, Tensor of shape [1, H, W, C]

    Returns:
        Scalar loss
    """
    return tf.reduce_sum(tf.math.abs(content - pastiche))


def gram_matrix(features):
    """Computes the Gram matrix from features

    Args:
        features: Tensor of shape [1, H, W, C]

    Returns:
        Gram matrix: Tensor of shape [C, C]
    """
    _, H, W, C = tf.shape(features)
    f = tf.reshape(features, [H*W, C])
    gram = tf.linalg.matmul(f, f, transpose_a=True)/tf.cast(H*W*C, dtype=tf.float32)
    return gram


def compute_style_loss(style_gram, pastiche_features, weights):
    """Computes the loss of the pastische with respect to the style

    Args:
        style_gram: A list of Gram tensors of shape [C,C]
        pastische_gram: A list of Gram tensors of shape [C,C]
        weights: A list of size len(style_gram) containing the proportional weight of each layer

    Returns:
        Scalar loss
    """
    loss = 0
    for i in range(len(pastiche_features)):
        loss += weights[i] * \
            tf.reduce_sum(tf.math.abs(
                style_gram[i] - gram_matrix(pastiche_features[i])))
    return loss


def loss_network():
    """Generates a loss network

    Returns:
        model: keras Model based on VGG 19 that returns multiple outputs
        content_layers: specifies the outputs which are used for content
        styler_layers: specifies the outputs which are used for style
        weights: Weights for each layer in style
    """
    cnn = tf.keras.applications.VGG19(
        include_top=False, weights='imagenet', pooling="avg")
    content_layers = ['block5_conv2']
    style_layers = [
        'block1_conv1',
        'block3_conv1',
        'block5_conv1'
    ]
    weights = [200000, 800, 15]
    layers = content_layers+style_layers
    outputs = [cnn.get_layer(name).output for name in layers]
    model = tf.keras.Model(inputs=cnn.input, outputs=outputs)

    return model, content_layers, style_layers, weights


def pastische_generator(image_path, style_path):

    network, content_layers, style_layers, weights = loss_network()

    # load image
    content_image = load_image(image_path)
    style_image = load_image(style_path)

    # Initialize content and style features
    content_features = network(content_image)[:len(content_layers)]
    style_gram = network(style_image)[len(content_layers):]
    for i in range(len(style_gram)):
        style_gram[i] = gram_matrix(style_gram[i])

    # Initialize optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

    # Initialize Pastische as the content image
    pastische = tf.Variable(content_image)

    for i in range(200):
        with tf.GradientTape() as tape:
            tape.watch(pastische)
            pastische_features = network(pastische)
            # Compute Loss
            content_loss = compute_content_loss(
                content_features[0], pastische_features[0])
            style_loss = compute_style_loss(
                style_gram, pastische_features[len(content_layers):], weights)
            loss = content_loss + style_loss
        # Apply gradient decent
        grad = tape.gradient(loss, pastische)
        optimizer.apply_gradients([(grad, pastische)])

        if i % 10 == 0:
            print(f"Number of Iterations: {i}")
    return pastische


pastische = pastische_generator("obrien.jpg", "style.jpg")
save_image(pastische, "pastische.jpg")
