# vae.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Subset
import numpy as np
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import pdb


class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        """
        Initialize the VAE model.

        Parameters:
            input_dim (int): Dimensionality of the input data.
            hidden_dim (int): Number of hidden units.
            latent_dim (int): Dimensionality of the latent space.
        """
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),  # Swish activation function
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),  # Swish activation function
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.SiLU(),  # Swish activation function
            nn.Linear(hidden_dim // 4, hidden_dim // 8),
            nn.SiLU(),  # Swish activation function
            nn.Linear(hidden_dim // 8, 2 * latent_dim),  # 2 for mean and variance.
        )
        self.softplus = nn.Softplus()

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 8),
            nn.SiLU(),  # Swish activation function
            nn.Linear(hidden_dim // 8, hidden_dim // 4),
            nn.SiLU(),  # Swish activation function
            nn.Linear(hidden_dim // 4, hidden_dim // 2),
            nn.SiLU(),  # Swish activation function
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.SiLU(),  # Swish activation function
            nn.Linear(hidden_dim, input_dim),
            nn.Softplus(),  # ensure nonnegativity
        )

    def encode(self, x, eps: float = 1e-8):
        """
        Encodes the input data into the latent space.

        Args:
            x (torch.Tensor): Input data.
            eps (float): Small value to avoid numerical instability.

        Returns:
            torch.distributions.MultivariateNormal: Normal distribution of the encoded data.
        """
        x = self.encoder(x)
        mu, logvar = torch.chunk(x, 2, dim=-1)
        # Use standard parameterization for numerical stability:
        scale = torch.exp(0.5 * logvar)
        scale_tril = torch.diag_embed(scale + eps)

        return torch.distributions.MultivariateNormal(mu, scale_tril=scale_tril)

    def reparameterize(self, dist):
        """
        Reparameterizes the encoded data to sample from the latent space.

        Args:
            dist (torch.distributions.MultivariateNormal): Normal distribution of the encoded data.
        Returns:
            torch.Tensor: Sampled data from the latent space.
        """
        return dist.rsample()

    def decode(self, z):
        """
        Decodes the data from the latent space to the original input space.

        Args:
            z (torch.Tensor): Data in the latent space.

        Returns:
            torch.Tensor: Reconstructed data in the original input space.
        """
        return self.decoder(z)

    def forward(self, x):
        # encode → (mu, logvar)
        stats = self.encoder(x)
        mu, logvar = torch.chunk(stats, 2, dim=-1)

        # clamp and softplus on logvar
        logvar = torch.clamp(logvar, -10.0, 10.0)
        # scale = F.softplus(logvar) + 1e-6
        scale = torch.exp(0.5 * logvar)

        # reparameterize
        z = mu + scale * torch.randn_like(mu)

        # decode
        recon_x = self.decoder(z)  # if your data ∈[0,1], add nn.Sigmoid() here!

        # reconstruction loss: sum over dims, mean over batch
        recon_loss = F.mse_loss(recon_x, x, reduction="sum") / x.size(0)

        # KL(q||p): closed form for diagonal Gaussian
        kl = (
            0.5
            * torch.sum(
                mu * mu + scale * scale - 1 - torch.log(scale * scale), dim=-1
            ).mean()
        )

        return recon_x, mu, logvar, recon_loss, kl

    def sample(self, n_samples, device):
        z = torch.randn(n_samples, self.latent_dim, device=device)
        return self.decode(z).cpu().numpy()


# ---------------------------
# Training Routine for VAE
# ---------------------------
def train(model, data_loader, optimizer, num_epochs=50, device="cpu"):
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0
        for batch_idx, (data,) in enumerate(data_loader):
            data = data.to(device).float()
            optimizer.zero_grad()
            # Unpack all outputs from the forward pass.
            recon_x, mu, logvar, recon_loss, kl = model(data)
            loss = recon_loss + kl
            loss.backward()
            optimizer.step()
            # Multiply by batch size to sum over samples
            epoch_loss += loss.item() * data.size(0)
        avg_loss = epoch_loss / len(data_loader.dataset)
        # print(f"VAE Epoch {epoch+1}/{num_epochs} - Average Loss: {avg_loss:.4f}")
    return avg_loss


def k_fold_validation_vae(
    dataset,
    input_dim,
    k=5,
    batch_size=128,
    num_epochs=50,
    hidden_dim=128,
    latent_dim=4,
    lr=1e-3,
    device="cpu",
):
    """
    Perform k-fold cross-validation for the VAE model.

    Returns a dict mapping fold index -> (val_loss, recon_loss, kl_loss).
    """
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    fold_results = {}

    for fold, (train_idx, val_idx) in enumerate(kf.split(dataset), start=1):
        # Split dataset
        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

        # Initialize model
        model = VAE(input_dim, hidden_dim, latent_dim).to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr)

        # Training
        for epoch in range(num_epochs):
            model.train()
            for batch in train_loader:
                x = batch[0].to(device)
                optimizer.zero_grad()
                recon_x, mu, logvar, recon_loss, kl = model(x)
                loss = recon_loss + kl
                loss.backward()
                optimizer.step()

        # Validation
        model.eval()
        total_loss = total_recon = total_kl = 0.0
        with torch.no_grad():
            for batch in val_loader:
                x = batch[0].to(device)
                recon_x, mu, logvar, recon_loss, kl = model(x)
                total_loss += (recon_loss + kl).item() * x.size(0)
                total_recon += recon_loss.item() * x.size(0)
                total_kl += kl.item() * x.size(0)
        n_val = len(val_subset)
        fold_results[fold] = (
            total_loss / n_val,
            total_recon / n_val,
            total_kl / n_val,
        )
    return fold_results
