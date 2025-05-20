# kde_mcmc.py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from KDEpy import FFTKDE


def evaluate_kde(point, data, bandwidth):
    """
    Evaluate the Gaussian KDE density at a point (normalized).
    """
    _, d = data.shape
    diff = data - point  # (n_samples, d)
    norm2 = np.sum(diff**2, axis=1)
    kernel_vals = np.exp(-norm2 / (2 * bandwidth**2))
    norm_const = (2 * np.pi) ** (d / 2) * (bandwidth**d)
    return np.mean(kernel_vals) / norm_const


def log_density(q, data, bandwidth):
    """
    U(x): Unnormalized log-density (up to additive constant) via KDE.
    """
    diff = data - q
    norm2 = np.sum(diff**2, axis=1)
    # log-kernel values
    log_k = -norm2 / (2 * bandwidth**2)
    # log-sum-exp for stability, ignore -log(n) constant
    m = np.max(log_k)
    return m + np.log(np.sum(np.exp(log_k - m)))


def grad_log_density(q, data, bandwidth):
    """
    Gradient of log-density wrt q.
    """
    diff = data - q  # (n_samples, d)
    norm2 = np.sum(diff**2, axis=1)
    w = np.exp(-norm2 / (2 * bandwidth**2))
    sum_w = np.sum(w) + 1e-12
    # gradient = sum_i [w_i * (x_i - q)] / (h^2 * sum_w)
    return (w[:, None] * diff).sum(axis=0) / (bandwidth**2 * sum_w)


def mcmc_sampling(
    initial_point,
    data,
    bandwidth,
    num_samples=1000,
    step_size=0.1,
    leapfrog_steps=10,
    burn_in=100,
):
    """
    Hamiltonian Monte Carlo sampling from the KDE-based density.

    H(q,p) = U(q) + K(p)

    Args:
        initial_point: (d,) start position.
        data: (n, d) training data.
        bandwidth: scalar bandwidth h.
        num_samples: number of samples to return after burn-in.
        step_size: leapfrog integrator step size.
        leapfrog_steps: number of leapfrog steps per proposal.
        burn_in: number of initial samples to discard.
    Returns:
        samples: (num_samples, d) array of HMC-drawn points.
    """
    dim = initial_point.shape[0]
    current = initial_point.copy()
    # U(q)
    current_logp = log_density(current, data, bandwidth)
    samples = []

    for i in range(num_samples + burn_in):
        # draw initial momentum K(p)
        p0 = np.random.randn(dim)
        p = p0.copy()
        q = current.copy()

        # half-step momentum update
        grad = grad_log_density(q, data, bandwidth)
        p += 0.5 * step_size * grad

        # full leapfrog steps
        for _ in range(leapfrog_steps):
            q += step_size * p
            grad = grad_log_density(q, data, bandwidth)
            if _ != leapfrog_steps - 1:
                p += step_size * grad

        # final half-step momentum
        p += 0.5 * step_size * grad
        # negate momentum for detailed balance
        p = -p

        # evaluate new state
        proposal_logp = log_density(q, data, bandwidth)
        # kinetic energies
        current_K = 0.5 * np.sum(p0**2)
        proposal_K = 0.5 * np.sum(p**2)
        # Metropolis acceptance
        log_accept = (proposal_logp - current_logp) + (current_K - proposal_K)
        if np.log(np.random.rand()) < log_accept:
            current = q
            current_logp = proposal_logp

        if i >= burn_in:
            samples.append(current.copy())

    return np.array(samples)
