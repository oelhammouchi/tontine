import numpy as np
from scipy.special import binom

from .context import Context
from .payoffs import (
    annuity_payoff,
    tontine_payoff,
    ul_annuity_payoff,
    ul_tontine_payoff,
)


def inner(p: float, t: float, k: np.ndarray, w: np.ndarray, ctx: Context):
    return np.sum(
        binom(ctx.n - 1, k)
        * p ** (k + 1)
        * (1 - p) ** (ctx.n - 1 - k)
        * (1 / (1 - ctx.γ))
        * (
            ul_annuity_payoff(w[0], t, ctx)
            + ul_tontine_payoff(w[1], t, ctx)
            + annuity_payoff(w[2], ctx)
            + tontine_payoff(w[3], ctx)
        )
        ** (1 - ctx.γ)
    )


def objective(x, ctx: Context) -> float:
    print("Performing objective evaluation")
    res = np.sum(
        np.exp(-ctx.r * ctx.t)
        * np.array(
            [
                inner(ctx.p[i], ctx.t[i], np.arange(ctx.n), x, ctx)
                for i in range(len(ctx.p))
            ]
        )
    )

    return float(res)
