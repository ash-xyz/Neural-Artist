import tensorflow as tf
from src.utilities import load_image, gram_matrix
from src.network import loss_network


def visualize_style_layer(image_style_path, style_layer):
    """Visualizes the style layers for a function
    """
    image = load_image(image_style_path)
    net = loss_network([], [style_layer])

    style = net(image)
    style_gram = gram_matrix([style])

    optimizer = tf.optimizers.Adam(
        learning_rate=0.02, beta_1=0.99, epsilon=1e-1)

    initializer = tf.random_uniform_initializer(0, 1)
    img = initializer(shape=image.shape)
    img_var = tf.Variable(img)

    ITERATIONS = 1000

    for i in range(ITERATIONS):
        with tf.GradientTape() as tape:
            tape.watch(img_var)
            img_style = net(img_var)
            img_gram = gram_matrix([img_style])

            loss = tf.reduce_mean(tf.square(img_gram[0]-style_gram[0]))

        grad = tape.gradient(loss, img_var)
        optimizer.apply_gradients([(grad, img_var)])
        img_var.assign(tf.clip_by_value(
            img_var, clip_value_min=0.0, clip_value_max=1.0))
        if(i % 50 == 0 or i == ITERATIONS-1):
            print(f"{i} - Loss: {loss}")
    return img_var
