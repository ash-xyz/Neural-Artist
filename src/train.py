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
from src.loss_network import VGG16, TVLoss
from src.utils import gram_matrix, normalize_batch, load_image


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
    vgg = VGG16(requires_grad=False).to(device)

    # Loads style image
    style = load_image(args.style, args.batch_size).to(device)

    # Computes style
    features_style = vgg(normalize_batch(style))
    gram_style = [gram_matrix(y) for y in features_style]

    for epoch in range(args.epochs):
        transformer.train()
        agg_content_loss = 0.
        agg_style_loss = 0.

        for batch_id, (x, _) in tqdm(enumerate(train_loader), unit='batch'):
            x = x.to(device)
            n_batch = len(x)

            optimizer.zero_grad()

            pred = transformer(x)
            y = normalize_batch(pred)
            x = normalize_batch(x)

            features_y = vgg(y)
            features_x = vgg(x)

            # Content Loss
            content_loss = args.content_weight * \
                mse_loss(features_y.relu2_2, features_x.relu2_2)

            # Style Loss
            style_loss = 0.
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
