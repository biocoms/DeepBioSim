#!/usr/bin/env python3
# train_mbgan_csv.py

import os
import argparse
import numpy as np
import pandas as pd
from scipy.stats import describe

from mbgan.model import MBGAN
from mbgan.utils import *


def main():
    # Configuration variables (replace argparse)
    csv = "input/vaginal.csv"
    output_dir = "gen_mbgan"
    name = "mbgan"
    iterations = 100000
    batch_size = 32
    save_interval = 1000
    latent_dim = 100
    seq_depth = 100
    n_critic = 5
    n_generator = 1

    # Load data and taxa names
    df = pd.read_csv(csv, index_col=0)
    data = df.values / seq_depth
    taxa_list = df.columns.tolist()
    print(f"Loaded data matrix {data.shape} (samples × taxa).")

    # Training
    model_config = {
        "ntaxa": data.shape[1],
        "latent_dim": latent_dim,
        "generator": {"n_channels": 512},
        "critic": {
            "n_channels": 256,
            "dropout_rate": 0.25,
        },
    }
    train_config = {
        "generator": {
            "optimizer": ("RMSprop", {}),
            "lr": 5e-5,
        },
        "critic": {
            "loss_weights": [1, 1, 10],
            "optimizer": ("RMSprop", {}),
            "lr": 5e-5,
        },
    }

    # 4. Initialize and train
    mbgan = MBGAN(name, model_config, train_config)
    mbgan.train(
        dataset=data,
        iteration=iterations,
        batch_size=batch_size,
        n_critic=n_critic,
        n_generator=n_generator,
        save_interval=save_interval,
        save_fn=None,
        experiment_dir=output_dir,
        verbose=1,
    )

    synth = mbgan.predict(n_samples=data.shape[0], seed=42)

    output_name = csv.split("/")[-1].replace(".csv", "_mbgan.csv")
    pd.DataFrame(synth, columns=taxa_list).to_csv(
        os.path.join(output_dir, output_name), index=False
    )


if __name__ == "__main__":
    main()
