import argparse
import os

from PIL import Image
from tqdm import tqdm
import torch
from torch import optim
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms

from src.transformer_net import TransformerNet
from src.vgg import VGG16
from src.utils import gram_matrix, normalize_batch


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

    transformer = TransformerNet().to(device)
    mse_loss = torch.nn.MSELoss()
    optimizer = optim.Adam(transformer.parameters(), args.learning_rate)

    vgg = VGG16(requires_grad=False).to(device)

    style_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])

    style = Image.open(args.style)
    style = style_transform(style)
    style = style.repeat(args.batch_size, 1, 1, 1).to(device)

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

            y = transformer(x)
            y = normalize_batch(y)
            x = normalize_batch(x)

            features_y = vgg(y)
            features_x = vgg(x)

            content_loss = args.content_weight * \
                mse_loss(features_y.relu3_3, features_x.relu3_3)

            style_loss = 0.
            for ft_y, gm_s in zip(features_y, gram_style):
                gm_y = gram_matrix(ft_y)
                style_loss += mse_loss(gm_y, gm_s[:n_batch, :, :])
            style_loss *= args.style_weight

            total_loss = content_loss + style_loss
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
