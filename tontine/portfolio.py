import numpy as np
from scipy.special import binom

from .payoffs import uia_payout, uto_payout, fia_payout, tto_payout
from .context import Context


def inner(p: float, t: float, k: np.ndarray, w: np.ndarray, ctx: Context):
    return np.sum(
        binom(ctx.n - 1, k)
        * p ** (k + 1)
        * (1 - p) ** (ctx.n - 1 - k)
        * (1 / (1 - ctx.γ))
        * (
            uia_payout(w[0], t, ctx)
            + uto_payout(w[1], t, ctx)
            + fia_payout(w[2], ctx)
            + tto_payout(w[3], ctx)
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
