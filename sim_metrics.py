import os
import numpy as np
import pandas as pd
import torch

# from torch.utils.data import DataLoader, TensorDataset


def process_file(filepath: str):

    dataset_name = os.path.splitext(os.path.basename(filepath))[0]
    print(f"\n=== Processing {dataset_name} ===")

    # Load data (first column as index, first row as header)
    df = pd.read_csv(filepath, index_col=0)
    data = df.values.T  # NOTE: p_genes * n_samples
    data = np.log1p(data)
    n_samples, input_dim = data.shape
    print(f"Loaded data: {n_samples} samples, {input_dim} features")

    data = np.round(data)
    # percentage of 0 RA across all samples
    gen_vae_path = f"./output/{dataset_name}_VAE_samples.npy"
    # if os.path.exists(gen_vae_path):

    gen_iwae_path = f"./output/{dataset_name}_IWAE_samples.npy"
    # if os.path.exists(gen_iwae_path):

    gen_diff_path = f"./output/{dataset_name}_diffusion_samples.npy"
    # if os.path.exists(gen_diff_path):

    gen_kde_path = f"./output/{dataset_name}_KDE_samples.npy"
    # if os.path.exists(gen_kde_path):

    gen_ms_path = f"./output/{dataset_name}_MS_samples.npy"
    # if os.path.exists(gen_ms_path):

    # Mean Absolute Error (MAE) – L1 norm
    # Mean Squared Error (MSE)
    # Pearson Correlation Coefficient between alpha diversity (PCC)
    # Classification AUROC/Accuracy using a simple classifier
    return


if __name__ == "__main__":
    os.makedirs("./output", exist_ok=True)

    process_file("./input/ibd.csv")
    process_file("./input/momspi16s.csv")
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
