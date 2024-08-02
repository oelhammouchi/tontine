import numpy as np
from scipy.special import binom

from .context import Context


def kappa(ctx: Context):
    k = np.arange(ctx.n)
    κ = map(
        lambda pp: np.sum(
            binom(ctx.n - 1, k)
            * pp ** (k + 1)
            * (1 - pp) ** (ctx.n - 1 - k)
            / (k + 1) ** (1 - ctx.γ)
        ),
        ctx.p,
    )

    return κ


def annuity_payoff(w: float, ctx: Context):
    c = (w * ctx.capital / (1 + ctx.risk_loading)) * (
        1 / (np.sum(np.exp(-ctx.r * ctx.t) * ctx.p))
    )

    return c ** (1 - ctx.γ) * np.sum(np.exp(-ctx.r * ctx.t) * ctx.p) / (1 - ctx.γ)


def ul_annuity_payoff(w: float, t: float, ctx: Context):
    v0 = w * ctx.capital / ((1 + ctx.risk_loading) * np.sum(ctx.p))
    return (
        v0 ** (1 - ctx.γ)
        * np.sum(
            np.exp(-ctx.γ * ctx.r * ctx.t)
            * np.exp(((ctx.μ - ctx.r) * ctx.π - 0.5 * ctx.γ * ctx.σ**2 * ctx.π**2) * t)
        )
        / (1 - ctx.γ)
    )


def tontine_payoff(w: float, ctx: Context):
    d = (w * ctx.capital / (1 + ctx.risk_loading)) * (
        1 / (np.sum(np.exp(-ctx.r * ctx.t) * (1 - (1 - ctx.p) ** ctx.n)))
    )

    κ = kappa(ctx)
    return ctx.n ** (1 - ctx.γ) * d ** (1 - ctx.γ) * np.sum(np.exp(-ctx.r * ctx.t) * κ)


def ul_tontine_payoff(w: float, t: float, ctx: Context):
    V0 = w * ctx.capital / ((1 + ctx.risk_loading) * np.sum(1 - (1 - ctx.p) ** ctx.n))
    κ = kappa(ctx)
    return (
        ctx.n ** (1 - ctx.γ)
        * np.sum(
            np.exp(-ctx.r * ctx.t)
            * κ
            * V0
            * np.exp(
                (ctx.r + (ctx.μ - ctx.r) * ctx.π - ctx.γ * ctx.σ**2 * ctx.π) * ctx.t
            )
        )
        ** (1 - ctx.γ)
        / (1 - ctx.γ)
    )
