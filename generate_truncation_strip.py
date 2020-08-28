import argparse
import math
import torch
import os
import random

from torchvision import utils
from model import Generator
from tqdm import tqdm


def generate_truncation_strip(args, generator, latent, noise):
    with torch.no_grad():
        generator.eval()
        slice_latent = latent[0,:].unsqueeze(0)
        
        images = []
        for i in range(0,6):
            image, _ = generator([slice_latent], input_is_latent=True, noise=noise, truncation=i*0.2, truncation_latent=mean_latent)
            if not os.path.exists('sample'):
                os.makedirs('sample')
            images.append(image)

        image_out = torch.cat(images)
        utils.save_image(
            image_out,
            f'sample/truncation_strip.png',
            nrow=8,
            normalize=True,
            range=(-1, 1)
        )

if __name__ == '__main__':
    device = 'cuda'

    parser = argparse.ArgumentParser()
    parser.add_argument('--size', type=int, default=1024)
    parser.add_argument('--channel_multiplier', type=int, default=2)
    parser.add_argument('--ckpt', type=str, default="ckpt/stylegan2-ffhq.pt")
    parser.add_argument('--latent', type=str, default="") 
    parser.add_argument('--truncation_mean', type=int, default=4096)

    args = parser.parse_args()

    args.latent_dim = 512
    args.n_mlp = 8

    generator = Generator(
        args.size, args.latent_dim, args.n_mlp, channel_multiplier=args.channel_multiplier
    ).to(device)
    checkpoint = torch.load(args.ckpt)
    generator.load_state_dict(checkpoint['g_ema'])
 
    latent = torch.load(args.latent)['latent']
    noise = torch.load(args.latent)['noises']
    
    with torch.no_grad():
        mean_latent = generator.mean_latent(args.truncation_mean)
    
    generate_truncation_strip(args, generator, latent, noise)
