import argparse
from pathlib import Path

import torch
from torchvision import utils
from tqdm import tqdm

from model import Generator


def generate(args, g_ema):
    with torch.no_grad():
        g_ema.eval()

        mean_latent = g_ema.mean_latent(args.truncation_mean)

        for i in tqdm(range(args.pics)):
            sample_z = torch.randn(args.sample, args.latent, device=args.device)
            sample, _ = g_ema([sample_z], truncation=args.truncation, truncation_latent=mean_latent)

            # Create directory if it does not exist
            save_dir = Path(args.save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            name = str(i).zfill(6)
            utils.save_image(
                sample,
                save_dir/f'image_{name}.png',
                nrow=1,
                normalize=True,
                range=(-1, 1),
            )

            if args.save_latent:
                latents = {'latent': sample_z}
                torch.save(latents, save_dir/f'latent_{name}.pt')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--size', type=int, default=1024)
    parser.add_argument('--sample', type=int, default=1)
    parser.add_argument('--pics', type=int, default=20)
    parser.add_argument('--truncation', type=float, default=1)
    parser.add_argument('--truncation_mean', type=int, default=4096)
    parser.add_argument('--ckpt', type=str, default="stylegan2-ffhq-config-f.pt")
    parser.add_argument('--channel_multiplier', type=int, default=2)
    parser.add_argument('--save_dir', type=str, default="sample/")
    parser.add_argument('--save_latent', action='store_true')

    args = parser.parse_args()

    args.latent = 512
    args.n_mlp = 8

    g_ema = Generator(
        args.size,
        args.latent,
        args.n_mlp,
        channel_multiplier=args.channel_multiplier
    ).to(args.device)

    checkpoint = torch.load(args.ckpt)
    g_ema.load_state_dict(checkpoint['g_ema'])

    generate(args, g_ema)
