import torch
from src.network import LossNet
from src.utilities import gram_matrix, load_image
from src.visualization.visualize import show_image


def visualize_style(style_layer, style_img_path):
    """Visualizes a style layer
    Args:
        style_layer: A single layer that you want to visualize the style of
        style_img_path: Path to the image you want to visualize the style for
    Returns:
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    net = LossNet(style_layers=[style_layer]).to(device)
    style_img = load_image(style_img_path).to(device)

    style_outputs, _ = net(style_img)
    style_gram = gram_matrix(style_outputs[style_layer])
    ITERATIONS = 200

    style = load_image("../images/content/obrien_small.jpg").to(device)
    #style = torch.rand(style_img.shape).to(device)

    optimizer = torch.optim.Adam([style.requires_grad_()], lr=1e1)

    mse = torch.nn.MSELoss()
    for i in range(ITERATIONS):
        optimizer.zero_grad()
        style.data.clamp_(0, 255)

        style_outputs, _ = net(style)
        y_gram = gram_matrix(style_outputs[style_layer])

        loss = 1e10* mse(y_gram, style_gram)

        loss.backward()
        optimizer.step()
        with torch.no_grad():
            if(i % 50 == 0 or i == ITERATIONS-1):
                style.data.clamp_(0, 255)
                print(f"Epoch: {i} Loss: {loss}")
                image = style.cpu()
                image/=255.0
                show_image(image)
    return style