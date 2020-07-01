from utilities import *
import tensorflow as tf


class ContentLoss:
    def __init__(self, target):
        super().__init__()
        self.target = target

    def forward(self, input):
        self.loss = 0
        for i in range(len(input)):
            self.loss += tf.reduce_mean(
                tf.square(input[i]-self.target[i]))
        return self.loss


def gram_matrix(features):
    """Computes the Gram matrix from features

    Args:
        features: A list of Tensors of shape [1, H, W, C]

    Returns:
        Gram matrix: A list of Tensors of shape [C, C]
    """
    gram = []
    for i in range(len(features)):
        _, H, W, C = tf.shape(features[i])
        gram.append(tf.linalg.einsum(
            'bijc,bijd->bcd', features[i], features[i])/tf.cast(C, tf.float32))
    return gram


class StyleLoss:
    def __init__(self, target):
        super().__init__()
        gram = gram_matrix(target)
        self.target_gram = gram

    def forward(self, input):
        input_gram = gram_matrix(input)
        self.loss = 0
        for i in range(len(input_gram)):
            self.loss += tf.reduce_mean(
                tf.square(input_gram[i]-self.target_gram[i]))
        return self.loss/len(input)


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


class normalization:
    def __init__(self, mean, std):
        super().__init__()
        self.mean = mean
        self.std = std

    def forward(self, image):
        return (image - self.mean)/self.std


def generate_pastische(style_path, content_path):
    model, len_content, _ = loss_network()
    content_image = load_image(content_path)
    instance = normalization([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    content_loss = ContentLoss(
        model(instance.forward(content_image))[:len_content])
    style_loss = StyleLoss(
        model(instance.forward(load_image(style_path)))[len_content:])

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
            features = model(instance.forward(pastische))
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


pastische = generate_pastische("style.jpg", "obrien.jpg")
save_image(pastische, "pastische.jpg")
"""Style test
net, len_content, len_styles = loss_network()
image = load_image("style.jpg")
image2 = load_image("obrien.jpg")
result = net(image2)[len_content:]
result2 = net(image2)[len_content:]
style_loss = StyleLoss(result)
print(style_loss.forward(result2))
"""
"""Content test
net, len_content, len_styles = loss_network()
image = load_image("style.jpg")
image2 = load_image("obrien.jpg")
result = net(image)[len_content:]
result2 = net(image2)[len_content:]
content_loss = ContentLoss(result)
print(content_loss.forward(result2))
"""
