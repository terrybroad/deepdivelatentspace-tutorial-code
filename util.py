import math
import random
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