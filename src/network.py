from utilities import *
import tensorflow as tf


def compute_content_loss(content, pastiche):
    """ Computes the loss of the pastiche with respect to the content

    Args:
        content: A list of features of the content image, Tensor of shape [1, H, W, C]
        pastiche: A list of features of the pastische image, Tensor of shape [1, H, W, C]

    Returns:
        Scalar loss
    """
    loss = 0
    for i in range(len(content)):
        loss += tf.reduce_sum(
            tf.math.squared_difference(content[i], pastiche[i]))
    return loss


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


def compute_style_loss(style_gram, pastiche_features, weights):
    """Computes the loss of the pastische with respect to the style

    Args:
        style_gram: A list of Gram tensors of shape [C,C]
        pastische_features: A list of features with Tensorshape 1, H, W, C
        weights: A list of size len(style_gram) containing the proportional weight of each layer

    Returns:
        Scalar loss
    """
    loss = 0
    for i in range(len(pastiche_features)):
        gram = gram_matrix(pastiche_features[i])
        loss += weights[i] * \
            tf.reduce_sum(tf.math.squared_difference(
                style_gram[i], gram))
    return loss


def tv_loss(img):
    """
    Compute total variation loss.
    Inputs:
    - img: Tensor of shape (1, H, W, 3) holding an input image.
    Returns:
    - loss: Tensor holding a scalar giving the total variation loss
      for img weighted by tv_weight.
    """
    N, H, W, C = img.shape
    sub_down = img[:, :H-1, :, :] - img[:, 1:H, :, :]
    sub_right = img[:, :, :W-1, :] - img[:, :, 1:W, :]
    return (tf.reduce_sum(tf.square(sub_down))) + tf.reduce_sum(tf.square(sub_right))


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


def pastische_generator(image_path, style_path):

    network, divider, weights = loss_network()

    # load image
    content_image = load_image(image_path)
    style_image = load_image(style_path)

    # Initialize content and style features
    content_features = network(content_image)[:divider]
    style_gram = network(style_image)[divider:]
    for i in range(len(style_gram)):
        style_gram[i] = gram_matrix(style_gram[i])

    # Set up optimization hyperparameters
    initial_lr = 3.0
    decayed_lr = 0.1
    decay_lr_at = 180
    max_iter = 1000

    step = tf.Variable(0, trainable=False)
    boundaries = [decay_lr_at]
    values = [initial_lr, decayed_lr]
    learning_rate_fn = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
        boundaries, values)

    # Initialize optimizer
    learning_rate = learning_rate_fn(step)
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-2)

    # Initialize Pastische as the content image
    pastische = tf.Variable(content_image)

    for i in range(max_iter):
        with tf.GradientTape() as tape:
            tape.watch(pastische)
            pastische_features = network(pastische)
            # Compute Loss
            content_loss = compute_content_loss(
                content_features, pastische_features)
            style_loss = compute_style_loss(
                style_gram, pastische_features[divider:], weights)
            #tv = tv_loss(pastische)
            loss = 1e4*content_loss + 1e-2*style_loss #+ 1e-3*tv
        # Apply gradient decent
        grad = tape.gradient(loss, pastische)
        optimizer.apply_gradients([(grad, pastische)])

        if i % 10 == 0:
            print(f"Number of Iterations: {i}")
    return pastische


pastische = pastische_generator("obrien.jpg", "style.jpg")
save_image(pastische, "pastische.jpg")
