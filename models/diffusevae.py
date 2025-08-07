# diffusevae.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
import torch.optim as optim
import numpy as np

from models.vae import VAE, train as train_vae
from models.iwae import IWAE, train as train_iwae
from models.diffusion import DiffusionModel, train as train_diffusion

import pdb


class DiffuseVAE(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        latent_dim,
        K,
        timesteps,
        vae_type,
    ):
        super().__init__()
        self.diffusion = DiffusionModel(input_dim, hidden_dim, timesteps)
        if vae_type == "vae":
            self.vae = VAE(input_dim, hidden_dim, latent_dim)
        elif vae_type == "iwae":
            self.vae = IWAE(input_dim, hidden_dim, latent_dim, K)
        else:
            raise ValueError(f"Unknown VAE type: {vae_type}")

    def sample(self, n_samples, device):
        return self.diffusion.sample(n_samples, device)


def train(model, loader, device, batch_size, epochs, lr):
    # Stage 1: VAE/IWAE training
    optimizer_vae = optim.Adam(model.vae.parameters(), lr=lr)
    if isinstance(model.vae, VAE):
        train_vae(model.vae, loader, optimizer_vae, epochs, device)
    else:
        train_iwae(model.vae, loader, optimizer_vae, epochs, device)
    model.vae.eval()

    # Precompute reconstructions for diffusion
    # pdb.set_trace()
    # recon_list = []
    with torch.no_grad():
        recon_list = model.vae.sample(len(loader.dataset), device)
    recon_tensor = torch.from_numpy(recon_list).float().to(device)
    recon_dataset = TensorDataset(recon_tensor)

    # Stage 2: Diffusion training on reconstructions
    diff_loader = DataLoader(recon_dataset, batch_size=batch_size, shuffle=True)
    optimizer_diff = optim.Adam(model.diffusion.parameters(), lr=lr)
    train_diffusion(model.diffusion, diff_loader, optimizer_diff, epochs, device)
