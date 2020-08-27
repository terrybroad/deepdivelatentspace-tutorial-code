import argparse
import math
import torch
import os
import random

from torchvision import utils
from model import Generator
from tqdm import tqdm


def mix_styles(args, generator, l1, l2, n1, n2):
    with torch.no_grad():
        generator.eval()
        slice_l1 = l1[0,:].unsqueeze(0)
        slice_l2 = l2[0,:].unsqueeze(0)
        
        images = []
        for i in range(1,16,2):
            image, _ = generator([slice_l1,slice_l2], input_is_latent=True, inject_index=i)
            if not os.path.exists('sample'):
                os.makedirs('sample')
            images.append(image)

        image_out = torch.cat(images)
        utils.save_image(
            image_out,
            f'sample/style_mix_'+args.label+'.png',
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
    parser.add_argument('--latent1', type=str, default="") 
    parser.add_argument('--latent2', type=str, default="")
    parser.add_argument('--label', type=str, default="fig1") 

    args = parser.parse_args()

    args.latent = 512
    args.n_mlp = 8

    generator = Generator(
        args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier
    ).to(device)
    checkpoint = torch.load(args.ckpt)
    generator.load_state_dict(checkpoint['g_ema'])
 
    l1 = torch.load(args.latent1)['latent']
    l2 = torch.load(args.latent2)['latent']
    
    n1 = torch.load(args.latent1)['noises']
    n2 = torch.load(args.latent2)['noises']

    mix_styles(args, generator, l1, l2, n1, n2)
