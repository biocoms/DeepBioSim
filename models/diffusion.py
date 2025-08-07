# diffusion.py
import torch
import torch.nn as nn
import math


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


class SinusoidalTimeEmbedding(nn.Module):
    """
    Sinusoidal time embedding layer.
    Takes a tensor of timesteps [B,] and returns a tensor of embeddings [B, dim].
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        # The 'div_term' calculates the 1 / 10000^(2i/dim) part of the formula
        half_dim = self.dim // 2
        div_term = torch.exp(
            torch.arange(half_dim, device=device) * -(math.log(10000.0) / half_dim)
        )

        # Unsqueeze 't' to broadcast it with 'div_term'
        # t: [B,] -> [B, 1]
        # div_term: [D/2,]
        # t * div_term: [B, D/2]
        pe = t.unsqueeze(1) * div_term

        # The final embedding uses sin for even indices and cos for odd indices
        embedding = torch.cat((torch.sin(pe), torch.cos(pe)), dim=1)
        return embedding


class DiffusionModel(nn.Module):
    """
    A simple Gaussian diffusion model with an MLP score network.
    Takes input_dim features, maps (x_t, t) → predicted noise.
    """

    def __init__(self, input_dim, hidden_dim=128, timesteps=1000):
        super().__init__()
        self.timesteps = timesteps
        self.input_dim = input_dim

        # 1. Time embedding network
        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )

        # 2. Projection layers
        self.input_proj = nn.Linear(self.input_dim, hidden_dim)
        self.time_proj = nn.Linear(hidden_dim, hidden_dim)

        # 3. Corrected main network (self.net)
        # It now receives the sum of projections, which has size hidden_dim.
        self.net = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, self.input_dim),  # Predict noise of size input_dim
        )

        # Precompute noise schedule (this part was correct)
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

        # Create noisy input x_t
        ab_t = self.sqrt_ab[t].unsqueeze(-1)
        omb_t = self.sqrt_1mab[t].unsqueeze(-1)
        x_t = ab_t * x0 + omb_t * noise
        # print(f"[t={t[0].item()}] x_t mean={x_t.mean():.4f}, std={x_t.std():.4f}")

        # ✨ Corrected training logic
        t_emb = self.time_embed(t)
        x_proj = self.input_proj(x_t)
        t_proj = self.time_proj(t_emb)

        pred_noise = self.net(x_proj + t_proj)  # Pass the sum to the network
        # print(f"pred_noise mean/std = {pred_noise.mean():.4f}/{pred_noise.std():.4f}")

        return nn.functional.mse_loss(pred_noise, noise)

    def sample(self, n_samples, device):
        # Use self.input_dim directly instead of the brittle calculation
        x = torch.randn(n_samples, self.input_dim, device=device)
        # print("betas:", self.betas.min(), self.betas.max())
        # print("alpha_bar:", self.alpha_bar[:5], "...", self.alpha_bar[-5:])
        if torch.isnan(x).any():
            raise RuntimeError(f"NaN encountered in diffusion sample at timestep {t}")

        for t in reversed(range(self.timesteps)):
            beta_t = self.betas[t]
            alpha_t = self.alphas[t]
            alpha_bar_t = self.alpha_bar[t]
            alpha_bar_t_m1 = (
                self.alpha_bar[t - 1] if t > 0 else torch.tensor(1.0, device=device)
            )

            # Generate and process time embedding for the current step 't'
            t_batch = torch.full((n_samples,), t, device=device, dtype=torch.long)
            t_emb = self.time_embed(t_batch)

            # Project inputs and add time embedding
            x_proj = self.input_proj(x)
            t_proj = self.time_proj(t_emb)

            eps_pred = self.net(x_proj + t_proj)  # This now works correctly

            # Compute posterior mean via DDPM formula
            mean = (x - (beta_t / torch.sqrt(1 - alpha_bar_t)) * eps_pred) / torch.sqrt(
                alpha_t
            )
            post_beta_t = beta_t * (1 - alpha_bar_t_m1) / (1 - alpha_bar_t)
            post_sigma_t = torch.sqrt(post_beta_t).clamp(min=1e-20)
            # FIXME:
            if torch.isnan(post_sigma_t).any():
                raise RuntimeError(f"post_sigma_t has NaNs at t={t}")
            if torch.isnan(x).any():
                raise RuntimeError(f"input x has NaNs before denoising at t={t}")
            if t > 0:
                noise = torch.randn_like(x)
                x = mean + post_sigma_t * noise
            else:
                x = mean
            if torch.isnan(x).any():
                raise RuntimeError(f"x turned NaN after sampling at t={t}")
        return x.clamp(min=0).cpu().numpy()


def train(model, dataloader, optimizer, num_epochs=50, device="cpu"):
    """
    Train the diffusion model, printing average loss each epoch.
    """
    model.train()
    with torch.autograd.detect_anomaly():
        for epoch in range(num_epochs):
            total_loss = 0.0
            for batch in dataloader:
                x = batch[0].to(device)
                optimizer.zero_grad()
                loss = model(x)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                total_loss += loss.item() * x.size(0)
            avg_loss = total_loss / len(dataloader.dataset)
            # print(f"Diffusion Epoch {epoch+1}/{num_epochs} - Average Loss: {avg_loss:.4f}")
    return avg_loss
