import argparse
import os

import numpy
import torch
from torch import optim
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from tqdm import tqdm

from src.transformer_net import TransformerNet
from src.loss_network import VGG16, VGG19, TVLoss
from src.utils import gram_matrix, normalize_batch, load_image

LOSS_NETWORK = 'vgg16'
#LOSS_NETWORK = 'vgg19'

# Make sure style layers and content layers exist in the Loss Network
STYLE_LAYERS = ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3']
CONTENT_LAYER = 'relu3_3'


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    transform = transforms.Compose([
        transforms.Resize(args.image_size),
        transforms.CenterCrop(args.image_size),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])
    train_dataset = datasets.ImageFolder(args.dataset, transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size)

    # Loads transformer, vgg
    transformer = TransformerNet().to(device)
    mse_loss = torch.nn.MSELoss()
    tv_loss = TVLoss(args.tv_weight).to(device)
    optimizer = optim.Adam(transformer.parameters(), args.learning_rate)
    if LOSS_NETWORK == 'vgg16':
        vgg = VGG16(STYLE_LAYERS, CONTENT_LAYER,
                    requires_grad=False).to(device)
    else:
        vgg = VGG19(STYLE_LAYERS, CONTENT_LAYER,
                    requires_grad=False).to(device)

    # Loads style image
    style = load_image(args.style, args.batch_size).to(device)

    # Computes style
    features_style, _ = vgg(normalize_batch(style, args.norm_range))
    gram_style = [gram_matrix(y)
                  for _, y in features_style.items()]

    for epoch in range(args.epochs):
        transformer.train()
        agg_content_loss = 0.
        agg_style_loss = 0.

        for batch_id, (x, _) in tqdm(enumerate(train_loader), unit=' batches'):
            x = x.to(device)
            n_batch = len(x)

            optimizer.zero_grad()

            pred = transformer(x)
            y = normalize_batch(pred, args.norm_range)
            x = normalize_batch(x, args.norm_range)

            features_y, content_y = vgg(y)
            features_x, content_x = vgg(x)

            # Content Loss
            content_loss = args.content_weight * \
                mse_loss(content_y[CONTENT_LAYER], content_x[CONTENT_LAYER])

            # Style Loss
            style_loss = 0.0
            features_y = [feature for _,
                          feature in features_y.items()]
            for ft_y, gm_s in zip(features_y, gram_style):
                gm_y = gram_matrix(ft_y)
                style_loss += mse_loss(gm_y, gm_s[:n_batch, :, :])
            style_loss *= args.style_weight

            # Tv Loss
            tv = tv_loss(pred)

            total_loss = content_loss + style_loss + tv
            total_loss.backward()
            optimizer.step()

            agg_content_loss += content_loss.item()
            agg_style_loss += style_loss.item()

            if (batch_id + 1) % args.log_interval == 0:
                tqdm.write('[{}] ({})\t'
                           'content: {:.6f}\t'
                           'style: {:.6f}\t'
                           'total: {:.6f}'.format(epoch+1, batch_id+1,
                                                  agg_content_loss /
                                                  (batch_id + 1),
                                                  agg_style_loss /
                                                  (batch_id + 1),
                                                  (agg_content_loss + agg_style_loss) / (batch_id + 1)))

            if (batch_id + 1) % args.save_interval == 0:
                transformer.eval().cpu()
                style_name = args.style.split('/')[-1].split('.')[0]
                checkpoint_file = os.path.join(args.checkpoint_dir,
                                               '{}.pth'.format(style_name))

                tqdm.write('Checkpoint {}'.format(checkpoint_file))
                torch.save(transformer.state_dict(), checkpoint_file)

                transformer.to(device).train()
