import os
import time
import argparse
import warnings

import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
from KDEpy import FFTKDE

from models.kde_mcmc import kde_sampling
from models.vae import VAE, train as train_vae
from models.iwae import IWAE, train as train_iwae
from models.diffusion import DiffusionModel, train as train_diffusion

from utils import match_columns


# --------------------
# Utilities
# --------------------
def get_device():
    """
    Determines the available device (CUDA, MPS, or CPU).
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


# --------------------
# Main routine
# --------------------
def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic microbiome count data using KDE, VAE, IWAE, or Diffusion models"
    )
    # Required arguments
    parser.add_argument(
        "--simulation_method",
        required=True,
        choices=["kde", "vae", "iwae", "diffusion"],
        help="Generative method to use",
    )
    parser.add_argument(
        "--matching_method",
        default="rank",
        choices=["rank", "mae", "corr"],
        help="Method for matching taxa/gene in generated data",
    )
    parser.add_argument(
        "--input", required=True, help="Path to input CSV file of counts"
    )
    parser.add_argument(
        "--output_folder", required=True, help="Directory to save generated samples"
    )
    # Optional arguments
    parser.add_argument(
        "--latent_dim",
        type=int,
        default=16,
        help="Dimensionality of latent space (used in VAE/IWAE)",
    )
    parser.add_argument(
        "--hidden_dim", type=int, default=128, help="Size of hidden layers"
    )
    parser.add_argument(
        "--num_epochs", type=int, default=300, help="Number of training epochs"
    )
    parser.add_argument(
        "--batch_size", type=int, default=128, help="Batch size for training"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=1e-3, help="Learning rate for optimizer"
    )
    parser.add_argument(
        "--K", type=int, default=20, help="Number of importance samples for IWAE"
    )
    parser.add_argument(
        "--time_steps",
        type=int,
        default=3000,
        help="Number of diffusion timesteps (for diffusion model)",
    )
    parser.add_argument(
        "--random_seed", type=int, default=42, help="Random seed for reproducibility"
    )

    args = parser.parse_args()
    warnings.filterwarnings("ignore")  # Suppress warnings

    # Set seeds
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)

    # Prepare device
    device = get_device()

    # Prepare output directory
    os.makedirs(args.output_folder, exist_ok=True)

    # Load and preprocess data
    dataset_name = os.path.splitext(os.path.basename(args.input))[0]
    print(f"\n=== Processing {dataset_name}: method={args.simulation_method} ===")
    df = pd.read_csv(args.input, index_col=0)
    data = df.values.T  # shape: (p_taxa, n_samples)
    data_log = np.log1p(data)

    n_samples, input_dim = data_log.shape
    print(f"Loaded data: {input_dim} samples, {n_samples} taxa")

    loader = DataLoader(
        TensorDataset(torch.from_numpy(data_log).float()),
        batch_size=args.batch_size,
        shuffle=True,
    )

    # Generate synthetic samples
    if args.simulation_method == "vae":
        # Train VAE
        start = time.perf_counter()
        vae = VAE(input_dim, args.hidden_dim, args.latent_dim).to(device)
        optimizer = optim.Adam(vae.parameters(), lr=args.learning_rate)
        train_vae(vae, loader, optimizer, num_epochs=args.num_epochs, device=device)
        vae.eval()
        with torch.no_grad():
            z = torch.randn(n_samples, args.latent_dim, device=device)
            gen = vae.decode(z).cpu().numpy()
        elapsed = time.perf_counter() - start
        print(f"VAE done in {elapsed:.2f}s")

    elif args.simulation_method == "iwae":
        # Train IWAE
        start = time.perf_counter()
        iwae = IWAE(input_dim, args.hidden_dim, args.latent_dim, args.K).to(device)
        optimizer = optim.Adam(iwae.parameters(), lr=args.learning_rate)
        train_iwae(iwae, loader, optimizer, num_epochs=args.num_epochs, device=device)
        iwae.eval()
        with torch.no_grad():
            gen = iwae.sample(n_samples)
        elapsed = time.perf_counter() - start
        print(f"IWAE done in {elapsed:.2f}s")

    elif args.simulation_method == "diffusion":
        # Train Diffusion model
        start = time.perf_counter()
        diff = DiffusionModel(input_dim, args.hidden_dim, timesteps=args.time_steps).to(
            device
        )
        optimizer = optim.Adam(diff.parameters(), lr=args.learning_rate)
        train_diffusion(
            diff, loader, optimizer, num_epochs=args.num_epochs, device=device
        )
        diff.eval()
        with torch.no_grad():
            gen = diff.sample(n_samples, device=device)
        gen[gen < 0] = 0  # Ensure no negative counts
        elapsed = time.perf_counter() - start
        print(f"Diffusion done in {elapsed:.2f}s")

    elif args.simulation_method == "kde":
        # KDE sampling
        start = time.perf_counter()
        kde = FFTKDE(kernel="gaussian").fit(data_log)
        bw = kde.bw
        gen = kde_sampling(data_log, bw, num_samples=n_samples)
        gen[gen < 0] = 0  # Ensure no negative counts
        elapsed = time.perf_counter() - start
        print(f"KDE done in {elapsed:.2f}s")

    else:
        raise ValueError(f"Unknown method: {args.simulation_method}")

    # Transform back from log1p domain and round
    gen_exp = np.expm1(gen)
    gen_counts = np.round(gen_exp).astype(int)
    gen_np = gen_counts.T  # shape: (n_samples, p_taxa)
    out_np = match_columns(df.values, gen_np, mode=args.matching_method)

    # Build DataFrame (features × samples)
    base_out = os.path.join(
        args.output_folder, f"{dataset_name}_{args.simulation_method}"
    )
    # Save matched CSV
    match_file = base_out + ".csv"
    out_df = pd.DataFrame(out_np, index=df.index, columns=df.columns)
    out_df.to_csv(match_file)
    print(f"Saved matched synthetic data to {match_file}")


if __name__ == "__main__":
    main()
