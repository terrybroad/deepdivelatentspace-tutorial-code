import argparse
import math
import torch
import os
import random

from torchvision import utils
from model import Generator
from tqdm import tqdm
import numpy as np

def slerp(val, low, high):
    '''
    original: Animating Rotation with Quaternion Curves, Ken Shoemake
    
    https://arxiv.org/abs/1609.04468
    Code: https://github.com/soumith/dcgan.torch/issues/14, Tom White
    '''
    omega = np.arccos(np.dot(low/np.linalg.norm(low), high/np.linalg.norm(high)))
    so = np.sin(omega)
    return np.sin((1.0-val)*omega) / so * low + np.sin(val*omega)/so * high

def get_slerp_loop(nb_latents, nb_interp, start_latent):
        low = start_latent
        og_low = low
        latent_interps = np.empty(shape=(0, 512), dtype=np.float32)
        for _ in range(nb_latents):
                high = np.random.normal(0.5, 0.7, 512)#low + np.random.randn(512) * 0.7

                interp_vals = np.linspace(1./nb_interp, 1, num=nb_interp)
                latent_interp = np.array([slerp(v, low, high) for v in interp_vals],
                                                dtype=np.float32)

                latent_interps = np.vstack((latent_interps, latent_interp))
                low = high
        
        interp_vals = np.linspace(1./nb_interp, 1, num=nb_interp)
        latent_interp = np.array([slerp(v, low, og_low) for v in interp_vals],
                                                dtype=np.float32)
        latent_interps = np.vstack((latent_interps, latent_interp))
        return latent_interps


def interpolate(args, generator, latent, noise):
    with torch.no_grad():
        generator.eval()
        slice_latent = latent[0,:]
        slerp_loop = get_slerp_loop(args.nb_latent, args.nb_interp, slice_latent.cpu().numpy())
        for i in range(len(slerp_loop)):
            input = torch.tensor(slerp_loop[i])
            input = input.view(1,512)
            input = input.to('cuda')
            image, _ = generator([input], input_is_latent=True, noise = noise)
            
            if not os.path.exists('circular_interpolate'):
                os.makedirs('circular_interpolate')

            utils.save_image(
                image,
                f'circular_interpolate/{str(i).zfill(6)}.png',
                nrow=1,
                normalize=True,
                range=(-1, 1),
            )


if __name__ == '__main__':
    device = 'cuda'

    parser = argparse.ArgumentParser()
    parser.add_argument('--size', type=int, default=1024)
    parser.add_argument('--channel_multiplier', type=int, default=2)
    parser.add_argument('--n_frames', type=int, default=100)
    parser.add_argument('--ckpt', type=str, default="models/stylegan2-ffhq-config-f.pt")
    parser.add_argument('--latent', type=str, default="") 
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
 
    latent = torch.load(args.latent)['latent']
    noise = torch.load(args.latent)['noises']

    interpolate(args, generator, latent, noise)
