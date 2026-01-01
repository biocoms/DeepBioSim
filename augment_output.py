#!/usr/bin/env python3

import os
import numpy as np
import pandas as pd

# Input files to augment
INPUT_FILES = [
    "output_sample/yachida_crc_abundance_iwae_mae_seed0.csv",
    "output_sample/yachida_crc_abundance_diffusion_mae_seed0.csv",
    "output_sample/yachida_crc_abundance_vae_mae_seed0.csv",
]
OUTPUT_PATH = "augmented_simulated_samples"

PERCENTS = [0.05, 0.10, 0.15, 0.20]
BETAS = [0.5, 1.0, 1.5, 2.0]


def load_table(path):
    return pd.read_csv(path, index_col=0)


def augment(df, percent, beta, rng):
    num_genes = df.shape[1]
    k = max(1, int(round(num_genes * percent)))
    selected_idx = rng.choice(num_genes, size=k, replace=False)

    counts = df.to_numpy(dtype=float)
    row_sums = counts.sum(axis=1, keepdims=True)
    # Avoid division by zero if any row is empty
    row_sums_safe = np.where(row_sums == 0, 1.0, row_sums)
    rel = counts / row_sums_safe

    rel[:, selected_idx] *= np.exp(beta)
    rel /= rel.sum(axis=1, keepdims=True)

    new_counts = rel * row_sums
    return pd.DataFrame(new_counts, index=df.index, columns=df.columns)


def output_name(input_path, percent, beta):
    base = os.path.basename(input_path)
    root, ext = os.path.splitext(base)
    percent_int = int(round(percent * 100))
    return f"{root}_select{percent_int}_beta{beta}{ext}"


def main():
    rng = np.random.default_rng(seed=0)
    for input_path in INPUT_FILES:
        df = load_table(input_path)
        for percent in PERCENTS:
            for beta in BETAS:
                augmented = augment(df, percent, beta, rng)
                out_name = output_name(input_path, percent, beta)
                out_path = os.path.join(OUTPUT_PATH, out_name)
                augmented.to_csv(out_path)


if __name__ == "__main__":
    main()
