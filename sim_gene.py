# comp_ds.py
import os
import glob
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
import numpy as np
from utils import *
from KDEpy import FFTKDE
from kde_mcmc import mcmc_sampling
from vae import VAE, train as train_vae, k_fold_validation_vae
from iwae import IWAE, train as train_iwae, k_fold_validation_iwae
from diffusion import DiffusionModel, train as train_diffusion

import pdb

# --------------------
# Configuration
# --------------------
latent_dim = 16  # 8, 16
hidden_dim = 128  # 64, 128
num_epochs = 50
batch_size = 128
learning_rate = 1e-3
K = 20
random_seed = 42

torch.manual_seed(random_seed)
np.random.seed(random_seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def process_file(filepath: str):
    dataset_name = os.path.splitext(os.path.basename(filepath))[0]
    print(f"\n=== Processing {dataset_name} ===")

    # Load data (first column as index, first row as header)
    df = pd.read_csv(filepath, index_col=0)
    data = df.values.T  # NOTE: p_genes * n_samples
    data = np.log1p(data)
    n_samples, input_dim = data.shape
    print(f"Loaded data: {n_samples} samples, {input_dim} features")

    loader = DataLoader(
        TensorDataset(torch.from_numpy(data).float()),
        batch_size=batch_size,
        shuffle=True,
    )

    data = np.round(data)

    # ----- Vanilla VAE -----
    # vae = VAE(input_dim, hidden_dim, latent_dim).to(device)
    # opt_vae = optim.Adam(vae.parameters(), lr=learning_rate)
    # train_vae(vae, loader, opt_vae, num_epochs=num_epochs, device=device)
    # vae.eval()
    # with torch.no_grad():
    #     z = torch.randn(n_samples, latent_dim, device=device)
    #     gen_vae = vae.decode(z).cpu().numpy()
    # gen_vae = np.round(gen_vae)
    # save_generated_samples(gen_vae, "VAE", dataset_name)
    # gen_vae = np.load("./output/TCGA_HNSC_rawcount_data_t_VAE_samples.npy")
    # gen_vae = np.load("./output/momspi16s_VAE_samples.npy")
    gen_vae = np.load(f"./output/{dataset_name}_VAE_samples.npy")

    # plot_pca(data, gen_vae, "VAE", dataset_name)
    # plot_tsne(data, gen_vae, "VAE", dataset_name)
    # plot_umap(data, gen_vae, "VAE", dataset_name)

    # ----- IWAE -----
    # iwae = IWAE(input_dim, latent_dim, hidden_dim, K).to(device)
    # opt_iwae = optim.Adam(iwae.parameters(), lr=learning_rate)
    # train_iwae(iwae, loader, opt_iwae, num_epochs=num_epochs, device=device)
    # iwae.eval()
    # with torch.no_grad():
    #     z = torch.randn(n_samples, latent_dim, device=device)
    #     gen_iwae = iwae.sample(n_samples)
    # gen_iwae = np.round(gen_iwae)
    # save_generated_samples(gen_iwae, "IWAE", dataset_name)
    # gen_iwae = np.load("./output/TCGA_HNSC_rawcount_data_t_IWAE_samples.npy")
    # gen_iwae = np.load("./output/momspi16s_IWAE_samples.npy")
    gen_iwae = np.load(f"./output/{dataset_name}_IWAE_samples.npy")

    # plot_pca(data, gen_iwae, "IWAE", dataset_name)
    # plot_tsne(data, gen_iwae, "IWAE", dataset_name)
    # plot_umap(data, gen_iwae, "IWAE", dataset_name)

    # ----- Diffusion -----
    # diff = DiffusionModel(input_dim, hidden_dim, timesteps=3000).to(device)
    # opt_diff = torch.optim.Adam(diff.parameters(), lr=1e-3)
    # train_diffusion(diff, loader, opt_diff, num_epochs, device=device)
    # diff.eval()
    # with torch.no_grad():
    #     gen_diff = diff.sample(n_samples, device=device)
    # plot_pca(data, gen_diff, "diffusion", dataset_name)
    # plot_tsne(data, gen_diff, "diffusion", dataset_name)
    # plot_umap(data, gen_diff, "diffusion", dataset_name)

    # ----- KDE-MCMC -----
    # kde = FFTKDE(kernel="gaussian").fit(data)
    # bw = kde.bw
    # init = data[np.random.choice(n_samples)]
    # gen_mcmc = mcmc_sampling(
    #     initial_point=init,
    #     data=data,
    #     bandwidth=bw,
    #     num_samples=n_samples,
    #     step_size=0.5,
    #     leapfrog_steps=10,
    #     burn_in=100,
    # )
    # plot_pca(data, gen_mcmc, "KDE-MCMC", dataset_name)
    # plot_tsne(data, gen_mcmc, "KDE-MCMC", dataset_name)
    # plot_umap(data, gen_mcmc, "KDE-MCMC", dataset_name)

    # ----- MIDASim -----
    gen_ms = np.load(f"./output/{dataset_name}_MS_samples.npy")
    plot_pca(data, gen_ms, "MS", dataset_name)
    plot_tsne(data, gen_ms, "MS", dataset_name)
    plot_umap(data, gen_ms, "MS", dataset_name)

    # NOTE shannon diversity can't compute negative values so it's not included in the analysis
    H_orig = shannon(data)
    H_vae = shannon(gen_vae)
    H_iwae = shannon(gen_iwae)
    # H_diff = shannon(gen_diff)
    H_ms = shannon(gen_ms)

    plot_violin(
        [H_orig, H_vae, H_iwae, H_ms],
        ["Original", "VAE", "IWAE", "MS"],
        "shannon entropy",
        dataset_name,
    )

    # plot_violin(
    #     [H_orig, H_vae, H_iwae, H_diff],
    #     ["Original", "VAE", "IWAE", "Diffusion"],
    #     "shannon entropy",
    #     dataset_name,
    # )

    # plot_violin(
    #     [H_orig, H_diff],
    #     ["Original", "Diffusion"],
    #     "shannon entropy",
    #     dataset_name,
    # )

    rich_orig = richness(data)
    rich_vae = richness(gen_vae)
    rich_iwae = richness(gen_iwae)
    # rich_diff = richness(gen_diff)
    rich_ms = richness(gen_ms)

    plot_violin(
        [rich_orig, rich_vae, rich_iwae, rich_ms],
        ["Original", "VAE", "IWAE", "MS"],
        "richness",
        dataset_name,
    )

    # plot_violin(
    #     [rich_orig, rich_vae, rich_iwae, rich_diff],
    #     ["Original", "VAE", "IWAE", "Diffusion"],
    #     "richness",
    #     dataset_name,
    # )

    # plot_violin(
    #     [rich_orig, rich_diff],
    #     ["Original", "Diffusion"],
    #     "richness",
    #     dataset_name,
    # )


# NOTE TCGA will *not* run on MCMC because of the high dimensionality
if __name__ == "__main__":
    os.makedirs("./output", exist_ok=True)
    process_file("./input/ibd.csv")
    process_file("./input/vaginal.csv")
    # process_file("./input/momspi16s.csv")
    # process_file("./input/t2d16s.csv")
    # process_file("./input/TCGA_HNSC_rawcount_data_t.csv")
    # process_file("./input/gene_MTB_caries_cleaned_t.csv")
