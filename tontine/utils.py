import numpy as np
from scipy.special import binom

from .data import MortalityData


def kappa(n: int, p: np.ndarray, γ: float):
    k = np.arange(n)
    κ = np.fromiter(
        map(
            lambda pp: np.sum(
                binom(n - 1, k)
                * pp ** (k + 1)
                * (1 - pp) ** (n - 1 - k)
                / (k + 1) ** (1 - γ)
            ),
            p,
        ),
        np.float64,
    )

    return κ


def pi(μ, σ, mort_data: MortalityData, γ: float):
    return (μ - mort_data.r) / (γ * σ**2) if μ - mort_data.r <= γ * σ**2 else 1


def log_normal_pdf(x: float, mu: float, sigma: float) -> float:
    return float(
        np.exp(-((np.log(x) - mu) ** 2) / (2 * sigma**2))
        / (x * sigma * np.sqrt(2 * np.pi))
    )


def log_normal_mean(mu: float, sigma: float) -> float:
    return float(np.exp(mu + sigma**2 / 2))


def log_normal_std(mu: float, sigma: float) -> float:
    return float(np.sqrt((np.exp(sigma**2) - 1) * np.exp(2 * mu + sigma**2)))
