"""
Scripts to create exploratory and results oriented visualizations
"""
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf


def graph_history(content_history, style_history):
    """Graphs history using matplotlib and seaborn

    Requirements:
        len(content_history) = len(style_history)
    Args:
        content_history: history of loss for the content
        style_history: history of loss for the style
    """
    sns.set(style='darkgrid')
    epoch = range(1, len(content_history) + 1)
    plt.plot(epoch, content_history)
    plt.plot(epoch, style_history)
    plt.legend(['Content Loss', 'Style Loss'])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show


def show_image(image, title=None):
    """Shows image using matplotlib

    Args:
        image: Tensor with a shape greater than 3
        title: Name of the image
    """
    if(len(image.shape) > 3):
        image = tf.squeeze(image, axis=0)
    plt.imshow(image)
    if title:
        plt.title(title)
