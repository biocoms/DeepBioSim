# diffusion.py
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm


# https://dzdata.medium.com/intro-to-diffusion-model-part-4-62bd94bd93fd
def cosine_beta_schedule(T, s=0.008):
    """
    Cosine schedule as proposed in https://arxiv.org/abs/2102.09672
    """

    def f(t):
        return torch.cos(((t / T) + s) / (1 + s) * torch.pi * 0.5) ** 2

    x = torch.linspace(0, T, T + 1)
    alphas_cumprod = f(x) / f(torch.tensor([0]))
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


class DiffusionModel(nn.Module):
    """
    A simple Gaussian diffusion model with an MLP score network.
    Takes input_dim features, maps (x_t, t) → predicted noise.
    """

    def __init__(self, input_dim, hidden_dim=256, timesteps=1000):
        super().__init__()
        self.timesteps = timesteps
        # Score network: predict noise given noisy x_t and time index
        self.net = nn.Sequential(
            nn.Linear(input_dim + 1, hidden_dim),
            # nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, input_dim),
        )
        # Precompute noise schedule
        betas = cosine_beta_schedule(timesteps)
        alphas = 1 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)
        sqrt_ab = torch.sqrt(alpha_bar)  # √(ᾱ_t)
        sqrt_1mab = torch.sqrt(1 - alpha_bar)  # √(1 - ᾱ_t)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bar", alpha_bar)
        self.register_buffer("sqrt_ab", sqrt_ab)
        self.register_buffer("sqrt_1mab", sqrt_1mab)

    def forward(self, x0):
        """
        Training step: sample random t, noise x0 → x_t, predict noise, MSE loss.
        """
        b, d = x0.shape
        t = torch.randint(0, self.timesteps, (b,), device=x0.device)
        noise = torch.randn_like(x0)
        # gather schedules at t
        ab_t = self.sqrt_ab[t].unsqueeze(-1)
        omb_t = self.sqrt_1mab[t].unsqueeze(-1)
        x_t = ab_t * x0 + omb_t * noise
        # normalize t ∈ [0,1]
        t_norm = t.float() / (self.timesteps - 1)
        inp = torch.cat([x_t, t_norm.unsqueeze(-1)], dim=1)
        pred_noise = self.net(inp)
        return nn.functional.mse_loss(pred_noise, noise)

    def sample(self, n_samples, device):
        input_dim = self.net[0].in_features - 1  # subtract 1 for time embedding
        x = torch.randn(n_samples, input_dim, device=device)
        for t in reversed(range(self.timesteps)):
            beta_t = self.betas[t]
            alpha_t = self.alphas[t]
            alpha_bar_t = self.alpha_bar[t]
            alpha_bar_t_m1 = (
                self.alpha_bar[t - 1] if t > 0 else torch.tensor(1.0, device=device)
            )
            # Network input
            t_emb = t / (self.timesteps - 1)
            inp = torch.cat([x, t_emb * torch.ones(n_samples, 1, device=device)], dim=1)
            eps_pred = self.net(inp)
            # Compute posterior mean via DDPM formula
            mean = (x - (beta_t / torch.sqrt(1 - alpha_bar_t)) * eps_pred) / torch.sqrt(
                alpha_t
            )
            post_beta_t = beta_t * (1 - alpha_bar_t_m1) / (1 - alpha_bar_t)
            post_sigma_t = torch.sqrt(post_beta_t)

            if t > 0:
                noise = torch.randn_like(x)
                x = mean + post_sigma_t * noise
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
        # print(f"Diffusion Epoch {epoch+1}/{num_epochs} - Average Loss: {avg_loss:.4f}")
    return avg_loss
