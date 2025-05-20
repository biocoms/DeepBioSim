# diffusion.py
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm


def linear_beta_schedule(T, beta_start=1e-4, beta_end=0.02):
    """
    Linear schedule from beta_start to beta_end over T timesteps.
    """
    return torch.linspace(beta_start, beta_end, T)


class DiffusionModel(nn.Module):
    """
    A simple Gaussian diffusion model with an MLP score network.
    Takes input_dim features, maps (x_t, t) → predicted noise.
    """

    def __init__(
        self, input_dim, hidden_dim=256, timesteps=1000, beta_start=1e-4, beta_end=0.02
    ):
        super().__init__()
        self.timesteps = timesteps
        # Score network: predict noise given noisy x_t and time index
        self.net = nn.Sequential(
            nn.Linear(input_dim + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, input_dim),
        )
        # Precompute noise schedule
        betas = linear_beta_schedule(timesteps, beta_start, beta_end)
        alphas = 1 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bar", alpha_bar)
        self.register_buffer("sqrt_ab", torch.sqrt(alpha_bar))
        self.register_buffer("sqrt_omb", torch.sqrt(1 - alpha_bar))

    def forward(self, x0):
        """
        Training step: sample random t, noise x0 → x_t, predict noise, MSE loss.
        """
        b, d = x0.shape
        t = torch.randint(0, self.timesteps, (b,), device=x0.device)
        noise = torch.randn_like(x0)
        # gather schedules at t
        ab_t = self.sqrt_ab[t].unsqueeze(-1)  # sqrt(alpha_bar_t)
        omb_t = self.sqrt_omb[t].unsqueeze(-1)  # sqrt(1 - alpha_bar_t)
        x_t = ab_t * x0 + omb_t * noise
        # normalize t ∈ [0,1]
        t_norm = t.float() / (self.timesteps - 1)
        inp = torch.cat([x_t, t_norm.unsqueeze(-1)], dim=1)
        pred_noise = self.net(inp)
        return nn.functional.mse_loss(pred_noise, noise)

    def sample(self, n_samples, device):
        x = torch.randn(n_samples, self.net[-1].out_features, device=device)
        print(
            f"[sample] init x: min={x.min():.4f}, max={x.max():.4f}, nan={torch.isnan(x).any()}"
        )
        for t in reversed(range(self.timesteps)):
            beta_t = self.betas[t]
            alpha_t = self.alphas[t]
            alpha_bar_t = self.alpha_bar[t]
            # Network input
            t_batch = torch.full((n_samples,), t, device=device, dtype=torch.long)
            t_norm = t_batch.float() / (self.timesteps - 1)
            inp = torch.cat([x, t_norm.unsqueeze(-1)], dim=1)
            eps_pred = self.net(inp)

            # Compute posterior mean via DDPM formula
            sqrt_alpha_t = torch.sqrt(alpha_t)
            sqrt_one_minus_ab = torch.sqrt(1 - alpha_bar_t)
            # (x_t - beta_t/√(1-ᾱ_t) * ε) / √α_t
            mean = (x - (beta_t / sqrt_one_minus_ab) * eps_pred) / sqrt_alpha_t

            if torch.isnan(mean).any():
                print(f"[sample] NaNs in mean at t={t}, 1-ᾱ_t={1-alpha_bar_t:.2e}")

            if t > 0:
                noise = torch.randn_like(x)
                sigma = torch.sqrt(beta_t)
                x = mean + sigma * noise
            else:
                x = mean
        return x.cpu().numpy()


def train(model, dataloader, optimizer, num_epochs=50, device="cpu"):
    """
    Train the diffusion model, printing average loss each epoch.
    """
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0.0
        for batch in dataloader:
            x = batch[0].to(device)
            optimizer.zero_grad()
            loss = model(x)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x.size(0)
        avg_loss = total_loss / len(dataloader.dataset)
        print(f"Diffusion Epoch {epoch+1}/{num_epochs} - Average Loss: {avg_loss:.4f}")
    return avg_loss
