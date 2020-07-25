from utilities import *
from network import *
import tensorflow as tf


def generate_pastische(style_path, content_path):
    model, len_content, _ = loss_network()
    content_image = load_image(content_path)
    content_loss = ContentLoss(
        model(content_image)[:len_content])
    style_loss = StyleLoss(
        model(load_image(style_path))[len_content:])

    num_iter = 1000
    content_weight = 5e0
    style_weight = 1e3
    tv_weight = 1e-2
    learning_rate = 1e-3
    norm_means = np.array([103.939, 116.779, 123.68])
    min_vals = -1 * norm_means
    max_vals = 255 - norm_means
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    pastische = tf.Variable(content_image)
    best_img = tf.Variable(content_image)
    for i in range(num_iter):
        best_loss = 1e50
        with tf.GradientTape() as tape:
            tape.watch(pastische)
            features = model(pastische)
            content_features = features[:len_content]
            style_features = features[len_content:]
            c_loss = content_loss.forward(content_features)
            s_loss = style_loss.forward(style_features)
            tv_loss = tf.image.total_variation(pastische)
            loss = content_weight * c_loss+style_weight*s_loss + tv_weight*tv_loss
        grad = tape.gradient(loss, pastische)
        optimizer.apply_gradients([(grad, pastische)])
        clipped = tf.clip_by_value(pastische, min_vals, max_vals)
        pastische.assign(clipped)
        if(loss < best_loss):
            best_img = pastische
        if(i % 50 == 0 or i == 999):
            print(
                f"Iterations: {i} Style Loss {style_loss.loss} Content Loss {content_loss.loss}")
    return best_img