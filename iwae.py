import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Subset
import numpy as np
from sklearn.model_selection import KFold


class IWAE(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dim, K=5):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.K = K

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.SiLU(),
            nn.Linear(hidden_dim // 4, hidden_dim // 8),
            nn.SiLU(),
            nn.Linear(hidden_dim // 8, 2 * latent_dim),  # output mu and logvar
        )
        self.softplus = nn.Softplus()

        # Decoder: symmetric to encoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 8),
            nn.SiLU(),
            nn.Linear(hidden_dim // 8, hidden_dim // 4),
            nn.SiLU(),
            nn.Linear(hidden_dim // 4, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Softplus(),  # ensure nonnegativity
        )

    def encode(self, x):
        stats = self.encoder(x)
        mu, logvar = torch.chunk(stats, 2, dim=-1)
        return mu, logvar

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        batch_size = x.size(0)
        mu, logvar = self.encode(x)
        # print("DEBUG: pre-clamp logvar:", logvar.min().item(), logvar.max().item())

        logvar = torch.clamp(logvar, -5.0, 5.0)
        scale = F.softplus(logvar) + 1e-6

        # print("scale (std) range:", scale.min().item(), scale.max().item())

        std = scale
        eps = torch.randn(self.K, batch_size, self.latent_dim, device=x.device)
        z = mu.unsqueeze(0) + eps * std.unsqueeze(0)
        # print("DEBUG z:", torch.isnan(z).any(), z.mean().item(), z.std().item())

        recon = self.decode(z.view(-1, self.latent_dim))
        recon = recon.view(self.K, batch_size, self.input_dim)

        log_p_x_given_z = -F.mse_loss(
            recon, x.unsqueeze(0).expand(self.K, -1, -1), reduction="none"
        ).sum(-1)

        log_p_z = -0.5 * (z**2).sum(-1)
        log_q = -0.5 * (
            ((z - mu.unsqueeze(0)) / std.unsqueeze(0)) ** 2 + logvar.unsqueeze(0)
        ).sum(-1)

        log_w = log_p_x_given_z + log_p_z - log_q

        iwae_loss = -(torch.logsumexp(log_w, dim=0) - np.log(self.K))
        return iwae_loss.mean()

    def sample(self, n_samples):
        z = torch.randn(
            n_samples, self.latent_dim, device=next(self.parameters()).device
        )
        return self.decode(z).detach().cpu().numpy()


def train(model, data_loader, optimizer, num_epochs=50, device="cpu"):
    """
    Train an IWAE model, printing average loss per epoch, analogous to the VAE train function.

    Args:
        model (torch.nn.Module): IWAE instance.
        data_loader (DataLoader): DataLoader yielding batches of (x,) tensors.
        optimizer (torch.optim.Optimizer): Optimizer for model parameters.
        num_epochs (int): Number of training epochs.
        device (str or torch.device): Compute device.

    Returns:
        float: Final epoch's average loss.
    """
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for batch in data_loader:
            x = batch[0].to(device).float()
            optimizer.zero_grad()
            loss = model(x)
            loss.backward()
            # Clip gradients to stabilize training
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            epoch_loss += loss.item() * x.size(0)
        avg_loss = epoch_loss / len(data_loader.dataset)
        # print(f"IWAE Epoch {epoch+1}/{num_epochs} - Average Loss: {avg_loss:.4f}")
    return avg_loss


def k_fold_validation_iwae(
    dataset_or_array,
    input_dim,
    hidden_dim,
    latent_dim,
    K=5,
    k=5,
    batch_size=128,
    num_epochs=50,
    lr=1e-3,
    device=None,
):
    """
    Perform k-fold cross-validation for IWAE.

    Args:
        dataset_or_array: TensorDataset or numpy array of shape (n_samples, input_dim).
        input_dim: Dimensionality of input.
        hidden_dim: Hidden layer size.
        latent_dim: Latent space dimensionality.
        K: Importance samples for IWAE.
        k: Number of CV folds (default 5).
        batch_size: Training batch size.
        num_epochs: Epochs per fold.
        lr: Learning rate.
        device: torch device.
    Returns:
        dict: fold → average validation IWAE loss.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build dataset
    if isinstance(dataset_or_array, TensorDataset):
        dataset = dataset_or_array
    elif isinstance(dataset_or_array, np.ndarray):
        X_t = torch.from_numpy(dataset_or_array).float()
        dataset = TensorDataset(X_t)
    else:
        raise TypeError(
            f"Expected TensorDataset or np.ndarray, got {type(dataset_or_array)}"
        )

    n_samples = len(dataset)
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    results = {}

    for fold_idx, (train_idx, val_idx) in enumerate(
        kf.split(range(n_samples)), start=1
    ):
        train_loader = DataLoader(
            Subset(dataset, train_idx), batch_size=batch_size, shuffle=True
        )
        val_loader = DataLoader(
            Subset(dataset, val_idx), batch_size=batch_size, shuffle=False
        )

        model = IWAE(input_dim, latent_dim, hidden_dim, K).to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr)

        # Train
        model.train()
        for epoch in range(num_epochs):
            for batch in train_loader:
                x = batch[0].to(device)
                optimizer.zero_grad()
                loss = model(x)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()

        # Validate
        model.eval()
        total_loss = 0.0
        total_samples = 0
        with torch.no_grad():
            for batch in val_loader:
                x = batch[0].to(device)
                l = model(x).item() * x.size(0)
                total_loss += l
                total_samples += x.size(0)
        results[fold_idx] = total_loss / total_samples

    return results
