import argparse
import math
import torch
import os
import random
import numpy as np

from torchvision import utils
from model import Generator
from tqdm import tqdm

from util import get_slerp_loop


def interpolate_loop_z(args, generator):
    with torch.no_grad():
        generator.eval()
        random_latent = np.random.randn(512)
        slerp_loop = get_slerp_loop(args.nb_latent, args.nb_interp, random_latent)
        for i in tqdm(range(len(slerp_loop))):
            input = torch.tensor(slerp_loop[i])
            input = input.view(1,512)
            input = input.to('cuda')
            image, _ = generator([input], truncation=args.truncation, truncation_latent=mean_latent)
            
            if not os.path.exists('interpolation_loop_z'):
                os.makedirs('interpolation_loop_z')

            utils.save_image(
                image,
                f'interpolation_loop_z/{str(i).zfill(6)}.png',
                nrow=1,
                normalize=True,
                range=(-1, 1),
            )


if __name__ == '__main__':
    device = 'cuda'

    parser = argparse.ArgumentParser()
    parser.add_argument('--size', type=int, default=1024)
    parser.add_argument('--channel_multiplier', type=int, default=2)
    parser.add_argument('--truncation', type=float, default=0.5)
    parser.add_argument('--truncation_mean', type=int, default=4096)
    parser.add_argument('--ckpt', type=str, default="ckpt/stylegan2-ffhq.pt")
    parser.add_argument('--nb_latent', type=int, default=3)
    parser.add_argument('--nb_interp', type=int, default=30)

    args = parser.parse_args()

    args.latent_dim = 512
    args.n_mlp = 8

    generator = Generator(
        args.size, args.latent_dim, args.n_mlp, channel_multiplier=args.channel_multiplier
    ).to(device)
    checkpoint = torch.load(args.ckpt)
    generator.load_state_dict(checkpoint['g_ema'])

    if args.truncation < 1:
        with torch.no_grad():
            mean_latent = generator.mean_latent(args.truncation_mean)
    else:
        mean_latent = None
 
    interpolate_loop_z(args, generator)
