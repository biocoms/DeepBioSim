# diffusevae.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np

from diffusion import DiffusionModel
from vae import VAE
from iwae import IWAE


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
            self.vae = IWAE(input_dim, latent_dim, hidden_dim, K)
        else:
            raise ValueError(f"Unknown VAE type: {vae_type}")

    def forward(self, x):
        if isinstance(self.vae, VAE):
            _, mu, lv, recon, kl = self.vae(x)
            std = torch.exp(0.5 * lv)
            z0 = mu + std * torch.randn_like(mu)
            vae_loss = recon + kl
            diffusion_loss = self.diffusion(z0)
            return diffusion_loss + vae_loss
        elif isinstance(self.vae, IWAE):
            iwae_loss = self.vae.forward(x)
            mu, logvar = self.vae.encode(x)
            std = torch.exp(0.5 * logvar)
            z0 = mu + std * torch.randn_like(mu)
            diffusion_loss = self.diffusion(z0)
            return diffusion_loss + iwae_loss

    def sample(self, n_samples, device):
        z0 = self.diffusion.sample(n_samples, device)
        if isinstance(self.vae, VAE):
            return self.vae.decode(z0).cpu().numpy()
        elif isinstance(self.vae, IWAE):
            return self.vae.sample(n_samples)


def train(model, dataloader, optimizer, num_epochs, device):
    model.train()
    for epoch in range(num_epochs):
        for x in dataloader:
            x = x.to(device)
            optimizer.zero_grad()
            loss = model(x)
            loss.backward()
            optimizer.step()
    return loss.item()
