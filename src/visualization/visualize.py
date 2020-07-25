"""
Scripts to create exploratory and results oriented visualizations
"""
import matplotlib.pyplot as plt
import seaborn as sns


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
