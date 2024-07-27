import numpy as np

from .context import Context


def fia_payout(w: float, ctx: Context):
    return (w * ctx.capital / (1 + ctx.risk_loading)) * (
        1 / (np.sum(np.exp(-ctx.r * ctx.t) * ctx.p))
    )


def uia_payout(w: float, t: float, ctx: Context):
    v0 = w * ctx.capital / ((1 + ctx.risk_loading) * np.sum(ctx.p))
    return v0 * np.exp(
        (ctx.r + (ctx.μ - ctx.r) * ctx.π - 0.5 * ctx.γ * ctx.σ**2 * ctx.π**2) * t
    )


def tto_payout(w: float, ctx: Context):
    return (w * ctx.capital / (1 + ctx.risk_loading)) * (
        1 / (np.sum(np.exp(-ctx.r * ctx.t) * (1 - (1 - ctx.p) ** ctx.n)))
    )


def uto_payout(w: float, t: float, ctx: Context):
    v0 = w * ctx.capital / ((1 + ctx.risk_loading) * np.sum(1 - (1 - ctx.p) ** ctx.n))
    return v0 * np.exp(
        (ctx.r + (ctx.μ - ctx.r) * ctx.π - 0.5 * ctx.γ * ctx.σ**2 * ctx.π**2) * t
    )
