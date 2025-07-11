# sim_metrics.py

import os
import numpy as np
import pandas as pd
import torch
from utils import *
import pdb

# from torch.utils.data import DataLoader, TensorDataset


def process_file(filepath: str):

    dataset_name = os.path.splitext(os.path.basename(filepath))[0]
    print(f"\n=== Processing {dataset_name} ===")

    # Load data (first column as index, first row as header)
    df = pd.read_csv(filepath, index_col=0)
    data = df.values  # n*p
    # data = np.log1p(data)
    print(f"Loaded data: {data.shape}")

    data = np.round(data)
    H_orig = shannon(data)  # n
    rich_orig = richness(data)  # n
    orig_bc = bc_matrix(data.astype(float))
    orig_jac = jaccard_matrix(data)
    gen_vae_path = f"./output/{dataset_name}_VAE_samples.npy"
    if os.path.exists(gen_vae_path):
        gen_vae = np.load(gen_vae_path)
        gen_vae = np.expm1(gen_vae).T  # n*p
        # Mean Absolute Error (MAE) – L1 norm
        mae_vae = np.mean(np.abs(data - gen_vae))
        print(f"MAE (VAE): {mae_vae:.4f}")
        # Pearson Correlation Coefficient for shannon and richness
        H_vae = shannon(gen_vae)
        H_pearson = pearson_corr(H_orig, H_vae)
        print(f"Shannon Pearson (VAE): {H_pearson:.4f}")
        rich_vae = richness(gen_vae)
        rich_pearson = pearson_corr(rich_orig, rich_vae)
        print(f"Richness Pearson (VAE): {rich_pearson:.4f}")
        # Pearson Correlation Coefficient comparison for Bray-Curtis and Jaccard indices
        gen_vae_bc = bc_matrix(gen_vae.astype(float))
        triu = np.triu_indices(orig_bc.shape[0], k=1)
        bc_pearson = pearson_corr(orig_bc[triu], gen_vae_bc[triu])
        print(f"BC Pearson (VAE): {bc_pearson:.4f}")
        gen_vae_jac = jaccard_matrix(gen_vae)
        jac_pearson = pearson_corr(orig_jac[triu], gen_vae_jac[triu])
        print(f"Jaccard Pearson (VAE): {jac_pearson:.4f}")

    gen_iwae_path = f"./output/{dataset_name}_IWAE_samples.npy"
    if os.path.exists(gen_iwae_path):
        gen_iwae = np.load(gen_iwae_path)
        gen_iwae = np.expm1(gen_iwae).T  # n*p
        mae_iwae = np.mean(np.abs(data - gen_iwae))
        print(f"MAE (IWAE): {mae_iwae:.4f}")
        H_iwae = shannon(gen_iwae)
        H_pearson = pearson_corr(H_orig, H_iwae)
        print(f"Shannon Pearson (IWAE): {H_pearson:.4f}")
        rich_iwae = richness(gen_iwae)
        rich_pearson = pearson_corr(rich_orig, rich_iwae)
        print(f"Richness Pearson (IWAE): {rich_pearson:.4f}")
        gen_iwae_bc = bc_matrix(gen_iwae.astype(float))
        bc_pearson = pearson_corr(orig_bc[triu], gen_iwae_bc[triu])
        print(f"BC Pearson (IWAE): {bc_pearson:.4f}")
        gen_iwae_jac = jaccard_matrix(gen_iwae)
        jac_pearson = pearson_corr(orig_jac[triu], gen_iwae_jac[triu])
        print(f"Jaccard Pearson (IWAE): {jac_pearson:.4f}")

    # gen_diff_path = f"./output/{dataset_name}_diffusion_samples.npy"
    # if os.path.exists(gen_diff_path):
    #     gen_diff = np.load(gen_diff_path).T

    # gen_kde_path = f"./output/{dataset_name}_KDE_samples.npy"
    # if os.path.exists(gen_kde_path):
    #     gen_kde = np.load(gen_kde_path).T

    gen_ms_path = f"./output/{dataset_name}_MS_samples.npy"
    if os.path.exists(gen_ms_path):
        gen_ms = np.load(gen_ms_path)
        gen_ms = np.expm1(gen_ms).T  # n*p
        mae_ms = np.mean(np.abs(data - gen_ms))
        print(f"MAE (MIDASim): {mae_ms:.4f}")
        H_ms = shannon(gen_ms)
        H_pearson = pearson_corr(H_orig, H_ms)
        print(f"Shannon Pearson (MIDASim): {H_pearson:.4f}")
        rich_ms = richness(gen_ms)
        rich_pearson = pearson_corr(rich_orig, rich_ms)
        print(f"Richness Pearson (MIDASim): {rich_pearson:.4f}")
        gen_ms_bc = bc_matrix(gen_ms.astype(float))
        bc_pearson = pearson_corr(orig_bc[triu], gen_ms_bc[triu])
        print(f"BC Pearson (MIDASim): {bc_pearson:.4f}")
        gen_ms_jac = jaccard_matrix(gen_ms)
        jac_pearson = pearson_corr(orig_jac[triu], gen_ms_jac[triu])
        print(f"Jaccard Pearson (MIDASim): {jac_pearson:.4f}")

    # TODO: Classification AUROC/Accuracy using a simple classifier


if __name__ == "__main__":
    os.makedirs("./output", exist_ok=True)

    process_file("./input/ibd.csv")
    # process_file("./input/momspi16s.csv")
    # process_file("./input/TCGA_HNSC_rawcount_data_t.csv")
    # process_file("./input/gene_MTB_healthy_cleaned_t.csv")
    # process_file("./input/gene_MTB_caries_cleaned_t.csv")
    # process_file("./input/gene_MTB_periodontitis_cleaned_t.csv")
    # process_file("./input/gene_MGB_periodontitis_transposed.csv")
    # process_file("./input/gene_MGB_caries_transposed.csv")
    # process_file("./input/gene_MGB_healthy_transposed.csv")
    # process_file("./input/GSE165512_CD.csv")
    # process_file("./input/GSE165512_Control.csv")
    # process_file("./input/GSE165512_UC.csv")
