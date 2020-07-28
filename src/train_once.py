import tensorflow as tf
import numpy as np
from src.utilities import *
from src.network import loss_network
from src.config import CONTENT_LAYERS, ITERATIONS, CONTENT_WEIGHT, STYLE_WEIGHT, TV_WEIGHT
from src.visualization.visualize import show_image


def generate_pastische(content_path, style_path):
    content_image = load_image(content_path)
    preprocessed_content = tf.keras.applications.vgg19.preprocess_input(
        content_image)
    style_image = load_image(style_path)
    preprocessed_style = tf.keras.applications.vgg19.preprocess_input(
        style_image)

    lossNetwork = loss_network()
    content_features = lossNetwork(preprocessed_content)[:len(CONTENT_LAYERS)]
    style_features = lossNetwork(preprocessed_style)[len(CONTENT_LAYERS):]
    content_loss = ContentLoss(content_features)
    style_loss = StyleLoss(style_features)

    optimizer = tf.optimizers.Adam(
        learning_rate=0.02, beta_1=0.99, epsilon=1e-1)

    pastische = tf.Variable(content_image)
    for i in range(ITERATIONS):
        with tf.GradientTape() as tape:
            tape.watch(pastische)
            pastische_processed = tf.keras.applications.vgg19.preprocess_input(
                pastische)
            pastische_features = lossNetwork(pastische_processed)
            pastische_content = pastische_features[:len(CONTENT_LAYERS)]
            pastische_style = pastische_features[len(CONTENT_LAYERS):]
            c_loss = CONTENT_WEIGHT * content_loss.forward(pastische_content)
            s_loss = STYLE_WEIGHT * style_loss.forward(pastische_style)
            tv_loss = TV_WEIGHT * tf.image.total_variation(pastische)
            total_loss = c_loss + s_loss + tv_loss
        grad = tape.gradient(total_loss, pastische)
        optimizer.apply_gradients([(grad, pastische)])
        pastische.assign(clip_0_1(pastische))
        if(i % 50 == 0 or i == ITERATIONS-1):
            print(f"{i} - Content Loss: {c_loss} Style_loss: {s_loss}")
            show_image(pastische)
    return pastische


def clip_0_1(image):
    return tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)
