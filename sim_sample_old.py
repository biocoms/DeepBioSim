# comp_ds.py
import os
import glob
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
import numpy as np
from scipy.stats import ttest_ind
from KDEpy import FFTKDE
from vae import VAE, train as train_vae, k_fold_validation_vae
from iwae import IWAE, train as train_iwae, k_fold_validation_iwae
from kde_mcmc import mcmc_sampling
from mks_test import mkstest
from utils import plot_pca, plot_tsne, svd_reduce, svd_reconstruct, shannon

import pdb

# --------------------
# Configuration
# --------------------
latent_dim = 8
hidden_dim = 64
num_epochs = 800
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
    # data = df.values.T  # shape: (n_samples, n_features)
    data = np.log1p(df.values)
    n_samples, input_dim = data.shape
    Z, V_red = svd_reduce(data, max_fraction=0.1)
    input_dim_red = Z.shape[1]
    print(f"Data: {n_samples} samples, {input_dim} features")

    loader = DataLoader(
        TensorDataset(torch.from_numpy(Z).float()), batch_size=batch_size, shuffle=True
    )
    print(f"Loaded reduced data: {n_samples} samples, {input_dim_red} features")

    # ----- Vanilla VAE -----
    vae = VAE(input_dim_red, hidden_dim, latent_dim).to(device)
    opt_vae = optim.Adam(vae.parameters(), lr=learning_rate)
    train_vae(vae, loader, opt_vae, num_epochs=num_epochs, device=device)
    vae.eval()
    with torch.no_grad():
        z = torch.randn(n_samples, latent_dim, device=device)
        gen_vae_red = vae.decode(z).cpu().numpy()
    gen_vae = svd_reconstruct(gen_vae_red, V_red)
    plot_pca(data.T, gen_vae.T, "VAE", dataset_name)
    # plot_tsne(data, gen_vae, "VAE", dataset_name)

    # ----- IWAE -----
    iwae = IWAE(input_dim_red, latent_dim, hidden_dim, K).to(device)
    opt_iwae = optim.Adam(iwae.parameters(), lr=learning_rate)
    # train and print avg loss per epoch
    train_iwae(iwae, loader, opt_iwae, num_epochs=num_epochs, device=device)
    iwae.eval()
    with torch.no_grad():
        z = torch.randn(n_samples, latent_dim, device=device)
        gen_iwae_red = iwae.sample(n_samples)
    gen_iwae = svd_reconstruct(gen_iwae_red, V_red)
    plot_pca(data.T, gen_iwae.T, "IWAE", dataset_name)
    # plot_tsne(data, gen_iwae, "IWAE", dataset_name)

    # ----- KDE-MCMC -----
    # kde = FFTKDE(kernel="gaussian").fit(Z)
    # bw = kde.bw
    # init = Z[np.random.choice(n_samples)]
    # gen_mcmc_red = mcmc_sampling(
    #     initial_point=init,
    #     data=Z,
    #     bandwidth=bw,
    #     num_samples=n_samples,
    #     step_size=0.5,
    #     leapfrog_steps=10,
    #     burn_in=100,
    # )
    # gen_mcmc = svd_reconstruct(gen_mcmc_red, V_red)
    # plot_pca(data, gen_mcmc, "KDE-MCMC", dataset_name)
    # plot_tsne(data, gen_mcmc, "KDE-MCMC", dataset_name)

    # NOTE mkstest takes a long time to run, and the library itself is a bit buggy

    # print("VAE mkstest:")
    # mkstest(gen_vae, data, alpha=0.05, verbose=True)
    # print("IWAE mkstest:")
    # mkstest(gen_iwae, data, alpha=0.05, verbose=True)
    # print("KDE-MCMC mkstest:")
    # mkstest(gen_mcmc, data, alpha=0.05, verbose=True)

    # NOTE shannon diversity can't compute negative values so it's not included in the analysis
    # H_orig = shannon(data)
    # H_vae = shannon(gen_vae)
    # H_iwae = shannon(gen_iwae)
    # H_mcmc = shannon(gen_mcmc)

    # t_vae, p_vae = ttest_ind(H_orig, H_vae)
    # t_iwae, p_iwae = ttest_ind(H_orig, H_iwae)
    # t_mcmc, p_mcmc = ttest_ind(H_orig, H_mcmc)

    # print("Shannon diversity t-test results:")
    # print(f" VAE:  t={t_vae:.3f}, p={p_vae:.3e}")
    # print(f" IWAE: t={t_iwae:.3f}, p={p_iwae:.3e}")
    # print(f" MCMC: t={t_mcmc:.3f}, p={p_mcmc:.3e}")

    # means (axis=1 mean for each gene)
    # mean_orig = np.mean(data, axis=0)
    # mean_vae = np.mean(gen_vae, axis=0)
    # mean_iwae = np.mean(gen_iwae, axis=0)
    # mean_mcmc = np.mean(gen_mcmc, axis=0)

    # t-tests on means
    # t_mean_vae, p_mean_vae = ttest_ind(mean_orig, mean_vae)
    # t_mean_iwae, p_mean_iwae = ttest_ind(mean_orig, mean_iwae)
    # t_mean_mcmc, p_mean_mcmc = ttest_ind(mean_orig, mean_mcmc)

    # print("Mean t-test results:")
    # print(f" VAE:  t={t_mean_vae:.3f}, p={p_mean_vae:.3e}")
    # print(f" IWAE: t={t_mean_iwae:.3f}, p={p_mean_iwae:.3e}")
    # print(f" MCMC: t={t_mean_mcmc:.3f}, p={p_mean_mcmc:.3e}")

    # medians
    # med_orig = np.median(data, axis=0)
    # med_vae = np.median(gen_vae, axis=0)
    # med_iwae = np.median(gen_iwae, axis=0)
    # med_mcmc = np.median(gen_mcmc, axis=0)

    # from scipy.stats import wilcoxon

    # paired Wilcoxon signed-rank tests on medians
    # w_med_vae, p_w_med_vae = wilcoxon(med_orig, med_vae)
    # w_med_iwae, p_w_med_iwae = wilcoxon(med_orig, med_iwae)
    # if you had paired KDE-MCMC medians too:
    # w_med_mcmc, p_w_med_mcmc = wilcoxon(med_orig, med_mcmc)

    # print("Wilcoxon signed-rank test on medians:")
    # print(f" VAE:  W={w_med_vae:.3f}, p={p_w_med_vae:.3e}")
    # print(f" IWAE: W={w_med_iwae:.3f}, p={p_w_med_iwae:.3e}")
    # print(f" MCMC: W={w_med_mcmc:.3f}, p={p_w_med_mcmc:.3e}")

    # from scipy.stats import mannwhitneyu

    # u_med_vae, p_u_med_vae = mannwhitneyu(med_orig, med_vae)
    # u_med_iwae, p_u_med_iwae = mannwhitneyu(med_orig, med_iwae)

    # print("Mann–Whitney U test on medians:")
    # print(f" VAE:  W={u_med_vae:.3f}, p={p_u_med_vae:.3e}")
    # print(f" IWAE: W={u_med_iwae:.3f}, p={p_u_med_iwae:.3e}")
    # print(f" MCMC: W={w_med_mcmc:.3f}, p={p_w_med_mcmc:.3e}")


# NOTE TCGA will *not* run on MCMC because of the high dimensionality
if __name__ == "__main__":
    os.makedirs("./output", exist_ok=True)
    # process_file("./input/momspi16s.csv")
    # process_file("./input/t2d16s.csv")
    process_file("./input/ibd.csv")
    process_file("./input/vaginal.csv")
    # process_file("./input/TCGA_HNSC_rawcount_data_t.csv")
