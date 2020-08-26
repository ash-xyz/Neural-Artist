import os
import io

import torch
from torchvision import transforms
from PIL import Image
from flask import Flask, request, jsonify
from transformer_net import TransformerNet
import base64

app = Flask(__name__)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

MODEL_PATH = {'candy': 'models/candy.pth',
              'cubist': 'models/cubist.pth',
              'rain_princess': 'models/rain_princess.pth',
              'udnie': 'models/udnie.pth',
              'udnie_small': 'models/udnie_small.pth',
              'wave': 'models/wave.pth'}


def load_image(image_bytes):
    """Returns an image ready to be put through the model
    Args:
        image_bytes: a binary image input
    Returns:
        A transformed image of shape 1,C,H,W
    """
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Lambda(lambda x: x.mul(255))])
    image = Image.open(io.BytesIO(image_bytes))
    return transform(image).unsqueeze(0).to(device)


def get_model(model_name):
    """Returns model
    Args:
        model_name: name of the model to load
    Returns:
        pytorch model
    """
    style_model = TransformerNet()
    state_dict = torch.load(MODEL_PATH[model_name])
    style_model.load_state_dict(state_dict)
    return style_model.eval().to(device)


def image_to_byte(image):
    """Returns bytes
    Args:
        image: tensor of shape 1,c,h,w
    Returns:
        img_base64: base64 encoding of images
    """
    image = image.to("cpu").clone().detach()
    image = image.numpy().squeeze()
    image = image.transpose(1,2,0)
    image = image.clip(0,255)
    image = Image.fromarray(image.astype("uint8"))

    rawBytes = io.BytesIO()
    image.save(rawBytes, "JPEG")
    rawBytes.seek(0)
    img_base64 = base64.b64encode(rawBytes.read())
    return img_base64


def eval_image(image_bytes, model_name):
    """Returns stylized image
    Args:
        image_bytes: a binary image input
        model_name: name of the model
    Returns:
        A Stylized image of shape C,H,W in Bytes
    """
    image = load_image(image_bytes)
    model = get_model(model_name)
    image = model(image).cpu()
    return image_to_byte(image)


@app.route('/style', methods=['POST'])
def convert():
    if request.method == 'POST':
        file = request.files['image'].read()  # Byte File
        model_name = request.form['style']
        img_base64 = eval_image(file, model_name)
        return jsonify({'status': str(img_base64)})


if __name__ == '__main__':
    app.run(debug=True)
