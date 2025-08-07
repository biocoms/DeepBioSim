# ldm.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from models.vae import VAE, train as train_vae
from models.iwae import IWAE, train as train_iwae
from models.diffusion import DiffusionModel, train as train_diffusion
import numpy as np


class LDM(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, K, timesteps, ae_type):

        super().__init__()
        self.diffusion = DiffusionModel(latent_dim, hidden_dim, timesteps)
        if ae_type == "vae":
            self.ae = VAE(input_dim, hidden_dim, latent_dim)
        elif ae_type == "iwae":
            self.ae = IWAE(input_dim, hidden_dim, latent_dim, K)
        else:
            raise ValueError(f"Unknown AE type: {ae_type}")

    def sample(self, n_samples, device):
        z_np = self.diffusion.sample(n_samples, device)
        print(
            "latent z has NaNs?",
            np.isnan(z_np).any(),
            " | z mean/std:",
            z_np.mean().item(),
            "/",
            z_np.std().item(),
        )
        z = torch.from_numpy(z_np).float().to(device)
        out = self.ae.decode(z)
        print("decoded out has NaNs?", torch.isnan(out).any())
        return self.ae.decode(z).clamp(min=0).detach().cpu()


def train(model, loader, device, batch_size, epochs, lr):
    optimizer_ae = torch.optim.Adam(model.ae.parameters(), lr=lr)
    if isinstance(model.ae, VAE):
        train_vae(model.ae, loader, optimizer_ae, epochs, device)
    else:
        train_iwae(model.ae, loader, optimizer_ae, epochs, device)
    model.ae.eval()

    zs = []
    with torch.no_grad():
        for batch in loader:
            if isinstance(model.ae, VAE):
                latent_dists = model.ae.encode(batch[0].to(device))
                z = latent_dists.rsample()
            elif isinstance(model.ae, IWAE):
                mu, lv = model.ae.sample(batch[0].to(device))
                scale = torch.exp(0.5 * lv)
                z = mu + scale * torch.randn_like(mu)
            if torch.isnan(z).any():
                raise RuntimeError(f"NaNs in latent z on AE")
            zs.append(z)
    z_tensor = torch.cat(zs, dim=0)
    z_dataset = TensorDataset(z_tensor)
    diff_loader = DataLoader(z_dataset, batch_size=batch_size, shuffle=True)
    optimizer_diff = torch.optim.Adam(model.diffusion.parameters(), lr=lr)
    train_diffusion(model.diffusion, diff_loader, optimizer_diff, epochs, device)


    # ----- LDM-VAE -----
    # ldm_vae_start_time = time.perf_counter()
    # ldm_vae = (
    #     LDM(input_dim, hidden_dim, latent_dim, K, timesteps, "vae").float().to(device)
    # )
    # train_ldm(
    #     ldm_vae,
    #     loader,
    #     device,
    #     batch_size,
    #     num_epochs,
    #     learning_rate,
    # )
    # ldm_vae.eval()
    # with torch.no_grad():
    #     gen_ldm_vae = ldm_vae.sample(n_samples, device=device)
    # save_generated_samples(gen_ldm_vae, "ldmvae", dataset_name)
    # ldm_vae_end_time = time.perf_counter()
    # print(f"LDM-VAE running time: {ldm_vae_end_time - ldm_vae_start_time:.4f} seconds")

    # gen_ldm_vae_path = f"./output/{dataset_name}_ldmvae_samples.npy"
    # if os.path.exists(gen_ldm_vae_path):
    #     gen_ldm_vae = np.load(gen_ldm_vae_path)
    #     plot_pca(data, gen_ldm_vae, "ldmvae", dataset_name)
    #     plot_tsne(data, gen_ldm_vae, "ldmvae", dataset_name)