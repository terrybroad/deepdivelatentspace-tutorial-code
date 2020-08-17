import argparse
from pathlib import Path
import os

import torch
from torchvision import utils
from model import Generator
from tqdm import tqdm

def mix_styles(args, generator, l1, l2):
    with torch.no_grad():
        generator.eval()
        slice_l1 = l1[0,:].unsqueeze(0)
        slice_l2 = l2[0,:].unsqueeze(0)

        for i in tqdm(range(1, generator.n_latent-1)):
            image, _ = generator([slice_l1,slice_l2],
                input_is_latent=args.w_space,
                inject_index=i)

            save_dir = Path(args.save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)

            utils.save_image(
                image,
                save_dir/f'style_mix_{str(i).zfill(2)}.png',
                nrow=1,
                normalize=True,
                range=(-1, 1))

if __name__ == '__main__':
    device = 'cuda'

    parser = argparse.ArgumentParser()
    parser.add_argument('--size', type=int, default=1024)
    parser.add_argument('--channel_multiplier', type=int, default=2)
    parser.add_argument('--ckpt', type=str, default="models/stylegan2-ffhq-config-f.pt")
    parser.add_argument('--latent1', type=str)
    parser.add_argument('--latent2', type=str)
    parser.add_argument('--w_space', action='store_true'
        help="Indicate given latents are from w space. "\
             "Otherwise latents are expected to be from z space")
    parser.add_argument('--save_dir', type=str, default="sample/")

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

    mix_styles(args, generator, l1, l2)
