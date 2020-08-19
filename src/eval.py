import argparse

from PIL import Image
import torch
from torchvision import transforms

from src.transformer_net import TransformerNet


def save_image(out, filename):
    img = out.clone().clamp(0, 255).numpy()
    # transpose (C, H, W) -> (H, W, C)
    img = img.transpose(1, 2, 0).astype('uint8')
    img = Image.fromarray(img)
    img.save(filename)


def stylize(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    content_image = Image.open(args.content)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])
    content = transform(content_image)
    content = content.unsqueeze(0).to(device)

    with torch.no_grad():
        style_model = TransformerNet()
        state_dict = torch.load(args.model)

        style_model.load_state_dict(state_dict)
        style_model.to(device)

        # Forward through Image Transformation Network
        out = style_model(content).cpu()

    # Save result image
    save_image(out[0], args.out)
