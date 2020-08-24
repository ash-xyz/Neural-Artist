import torch

from src.transformer_net import TransformerNet
from src.utils import save_image, load_image


def stylize(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Loads image
    content = load_image(args.content).to(device)

    with torch.no_grad():
        # Load Transformer net
        style_model = TransformerNet()
        state_dict = torch.load(args.model)

        # Load Model Weights
        style_model.load_state_dict(state_dict)
        style_model.eval().to(device)

        # Forward through Image Transformation Network
        out = style_model(content).cpu()

    # Save result image
    save_image(out, args.out)
