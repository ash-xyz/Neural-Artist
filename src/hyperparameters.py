"""
File that includes the Hyperparameters for the project
"""

CONTENT_WEIGHT = 5e0
STYLE_WEIGHT = 1e2
TV_WEIGHT = 1e-3
ITERATIONS = 1000
""" Choose between "content_image", "style_image" & "random" """
INITIALIZER = "content_image"
CONTENT_LAYERS = [
    'block4_conv2'
]
STYLE_LAYERS = ['block1_conv1',
                'block2_conv1',
                'block3_conv1',
                'block4_conv1',
                'block5_conv1'
                ]
