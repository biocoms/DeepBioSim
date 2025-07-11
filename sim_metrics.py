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
    data = df.values.T  # n*p
    # data = np.log1p(data)
    print(f"Loaded data: {data.shape}")

    data = np.round(data)
    H_orig = shannon(data)
    # Mean Absolute Error (MAE) – L1 norm
    # Pearson Correlation Coefficient between alpha diversity (PCC)
    gen_vae_path = f"./output/{dataset_name}_VAE_samples.npy"
    if os.path.exists(gen_vae_path):
        gen_vae = np.load(gen_vae_path)  # p*n
        gen_vae = np.expm1(gen_vae)
        mae_vae = np.mean(np.abs(data - gen_vae))
        print(f"MAE (VAE): {mae_vae:.4f}")
        H_vae = shannon(gen_vae)
        H_pearson = pearson_corr(H_orig, H_vae)
        print(f"Shannon Pearson (VAE): {H_pearson:.4f}")

    gen_iwae_path = f"./output/{dataset_name}_IWAE_samples.npy"
    if os.path.exists(gen_iwae_path):
        gen_iwae = np.load(gen_iwae_path)
        gen_iwae = np.expm1(gen_iwae)
        mae_iwae = np.mean(np.abs(data - gen_iwae))
        print(f"MAE (IWAE): {mae_iwae:.4f}")
        H_iwae = shannon(gen_iwae)
        H_pearson = pearson_corr(H_orig, H_iwae)
        print(f"Shannon Pearson (IWAE): {H_pearson:.4f}")

    # gen_diff_path = f"./output/{dataset_name}_diffusion_samples.npy"
    # if os.path.exists(gen_diff_path):
    #     gen_diff = np.load(gen_diff_path).T

    # gen_kde_path = f"./output/{dataset_name}_KDE_samples.npy"
    # if os.path.exists(gen_kde_path):
    #     gen_kde = np.load(gen_kde_path).T

    gen_ms_path = f"./output/{dataset_name}_MS_samples.npy"
    if os.path.exists(gen_ms_path):
        gen_ms = np.load(gen_ms_path)
        gen_ms = np.expm1(gen_ms)
        mae_ms = np.mean(np.abs(data - gen_ms))
        print(f"MAE (MIDASim): {mae_ms:.4f}")
        H_ms = shannon(gen_ms)
        H_pearson = pearson_corr(H_orig, H_ms)
        print(f"Shannon Pearson (MIDASim): {H_pearson:.4f}")

    # Classification AUROC/Accuracy using a simple classifier
    return


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
