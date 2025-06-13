# sim_gene.py
import os, time
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
import numpy as np
from utils import *
from KDEpy import FFTKDE
from kde_mcmc import mcmc_sampling, kde_sampling
from vae import VAE, train as train_vae, k_fold_validation_vae
from iwae import IWAE, train as train_iwae, k_fold_validation_iwae
from diffusion import DiffusionModel, train as train_diffusion

import pdb

import warnings

warnings.filterwarnings("ignore")  # Ignore all warnings

# --------------------
# Configuration
# --------------------
latent_dim = 16  # 8, 16
hidden_dim = 128  # 64, 128
num_epochs = 300
batch_size = 128
learning_rate = 1e-3
K = 20
random_seed = 42

torch.manual_seed(random_seed)
np.random.seed(random_seed)


def get_device():
    """
    Determines the available device (CUDA or MPS) for computation.
    Returns:
      torch.device: The device to use (CUDA, MPS, or CPU).
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    return device


device = get_device()


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
    # vae_start_time = time.perf_counter()
    # vae = VAE(input_dim, hidden_dim, latent_dim).to(device)
    # opt_vae = optim.Adam(vae.parameters(), lr=learning_rate)
    # train_vae(vae, loader, opt_vae, num_epochs=num_epochs, device=device)
    # vae.eval()
    # with torch.no_grad():
    #     z = torch.randn(n_samples, latent_dim, device=device)
    #     gen_vae = vae.decode(z).cpu().numpy()
    # gen_vae = np.round(gen_vae)
    # save_generated_samples(gen_vae, "VAE", dataset_name)
    # vae_end_time = time.perf_counter()
    # print(f"VAE running time: {vae_end_time - vae_start_time:.4f} seconds")
    # gen_vae_path = f"./output/{dataset_name}_VAE_samples.npy"
    # if os.path.exists(gen_vae_path):
    #     gen_vae = np.load(gen_vae_path)
    #     plot_pca(data, gen_vae, "VAE", dataset_name)
    #     plot_tsne(data, gen_vae, "VAE", dataset_name)
    #     plot_umap(data, gen_vae, "VAE", dataset_name)

    # ----- IWAE -----
    # iwae_start_time = time.perf_counter()
    # iwae = IWAE(input_dim, latent_dim, hidden_dim, K).to(device)
    # opt_iwae = optim.Adam(iwae.parameters(), lr=learning_rate)
    # train_iwae(iwae, loader, opt_iwae, num_epochs=num_epochs, device=device)
    # iwae.eval()
    # with torch.no_grad():
    #     z = torch.randn(n_samples, latent_dim, device=device)
    #     gen_iwae = iwae.sample(n_samples)
    # gen_iwae = np.round(gen_iwae)
    # save_generated_samples(gen_iwae, "IWAE", dataset_name)
    # iwae_end_time = time.perf_counter()
    # print(f"IWAE running time: {iwae_end_time - iwae_start_time:.4f} seconds")

    # gen_iwae_path = f"./output/{dataset_name}_IWAE_samples.npy"
    # if os.path.exists(gen_iwae_path):
    #     gen_iwae = np.load(gen_iwae_path)
    #     plot_pca(data, gen_iwae, "IWAE", dataset_name)
    #     plot_tsne(data, gen_iwae, "IWAE", dataset_name)
    #     plot_umap(data, gen_iwae, "IWAE", dataset_name)

    # ----- Diffusion -----
    # diff_start_time = time.perf_counter()
    # diff = DiffusionModel(input_dim, hidden_dim, timesteps=3000).to(device)
    # opt_diff = torch.optim.Adam(diff.parameters(), lr=1e-3)
    # train_diffusion(diff, loader, opt_diff, num_epochs, device=device)
    # diff.eval()
    # with torch.no_grad():
    #     gen_diff = diff.sample(n_samples, device=device)
    # save_generated_samples(gen_diff, "diffusion", dataset_name)
    # diff_end_time = time.perf_counter()
    # print(f"Diffusion running time: {diff_end_time - diff_start_time:.4f} seconds")

    gen_diff_path = f"./output/{dataset_name}_diffusion_samples.npy"
    if os.path.exists(gen_diff_path):
        gen_diff = np.load(gen_diff_path)
        plot_pca(data, gen_diff, "diffusion", dataset_name)
        plot_tsne(data, gen_diff, "diffusion", dataset_name)
        plot_umap(data, gen_diff, "diffusion", dataset_name)

    # ----- KDE -----
    # if input_dim <= 10:
    # kde_start_time = time.perf_counter()
    # kde = FFTKDE(kernel="gaussian").fit(data)
    # bw = kde.bw
    # # direct mixture sampling from KDE
    # gen_kde = kde_sampling(data, bw, num_samples=n_samples)
    # save_generated_samples(gen_kde, "KDE", dataset_name)
    # kde_end_time = time.perf_counter()
    # print(f"KDE running time: {kde_end_time - kde_start_time:.4f} seconds")

    gen_kde_path = f"./output/{dataset_name}_KDE_samples.npy"
    if os.path.exists(gen_kde_path):
        gen_kde = np.load(gen_kde_path)
        plot_pca(data, gen_kde, "KDE", dataset_name)
        plot_tsne(data, gen_kde, "KDE", dataset_name)
        plot_umap(data, gen_kde, "KDE", dataset_name)

    # ----- MIDASim -----
    # gen_ms = np.load(f"./output/{dataset_name}_MS_samples.npy")
    # plot_pca(data, gen_ms, "MS", dataset_name)
    # plot_tsne(data, gen_ms, "MS", dataset_name)
    # plot_umap(data, gen_ms, "MS", dataset_name)

    # NOTE shannon diversity can't compute negative values so it's not included in the analysis
    # H_orig = shannon(data)
    # H_vae = shannon(gen_vae)
    # H_iwae = shannon(gen_iwae)
    # H_diff = shannon(gen_diff)
    # H_kde = shannon(gen_kde)
    # H_ms = shannon(gen_ms)

    # plot_violin(
    #     [H_orig, H_vae, H_iwae, H_ms],
    #     ["Original", "VAE", "IWAE", "MS"],
    #     "shannon entropy",
    #     dataset_name,
    # )

    # plot_violin(
    #     [H_orig, H_vae, H_iwae, H_diff, H_ms],
    #     ["Original", "VAE", "IWAE", "Diffusion", "MS"],
    #     "shannon entropy",
    #     dataset_name,
    # )

    # plot_violin(
    #     [H_orig, H_vae, H_iwae, H_diff],
    #     ["Original", "VAE", "IWAE", "Diffusion"],
    #     "shannon entropy",
    #     dataset_name,
    # )

    # plot_violin(
    #     [H_orig, H_kde, H_diff],
    #     ["Original", "KDE", "Diffusion"],
    #     "shannon entropy",
    #     dataset_name,
    # )

    # rich_orig = richness(data)
    # rich_vae = richness(gen_vae)
    # rich_iwae = richness(gen_iwae)
    # rich_diff = richness(gen_diff)
    # rich_kde = richness(gen_kde)
    # rich_ms = richness(gen_ms)

    # plot_violin(
    #     [rich_orig, rich_vae, rich_iwae, rich_ms],
    #     ["Original", "VAE", "IWAE", "MS"],
    #     "richness",
    #     dataset_name,
    # )

    # plot_violin(
    #     [rich_orig, rich_vae, rich_iwae, rich_diff, rich_ms],
    #     ["Original", "VAE", "IWAE", "Diffusion", "MS"],
    #     "richness",
    #     dataset_name,
    # )

    # plot_violin(
    #     [rich_orig, rich_vae, rich_iwae, rich_diff],
    #     ["Original", "VAE", "IWAE", "Diffusion"],
    #     "richness",
    #     dataset_name,
    # )

    # plot_violin(
    #     [rich_orig, rich_kde, rich_diff],
    #     ["Original", "KDE", "Diffusion"],
    #     "richness",
    #     dataset_name,
    # )


# NOTE TCGA will *not* run on MCMC because of the high dimensionality
if __name__ == "__main__":
    os.makedirs("./output", exist_ok=True)

    # process_file("./input/ibd.csv")
    # process_file("./input/momspi16s.csv")
    # process_file("./input/TCGA_HNSC_rawcount_data_t.csv")
    process_file("./input/gene_MTB_healthy_cleaned_t.csv")
    # process_file("./input/gene_MTB_caries_cleaned_t.csv")
    # process_file("./input/gene_MTB_periodontitis_cleaned_t.csv")
    # process_file("./input/gene_MGB_periodontitis_transposed.csv")
    # process_file("./input/gene_MGB_caries_transposed.csv")
    # process_file("./input/gene_MGB_healthy_transposed.csv")
    # process_file("./input/GSE165512_CD.csv")
    # process_file("./input/GSE165512_Control.csv")
    # process_file("./input/GSE165512_UC.csv")
