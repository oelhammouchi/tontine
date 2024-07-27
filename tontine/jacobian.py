from sympy import *
from sympy.utilities.autowrap import autowrap, binary_function
import numpy as np

from .payoffs import Context

n, k, r, t, v, C, T = symbols("n k r t v C T")
w1, w2, w3, w4 = symbols("w1:5")
gamma, lamda, mu, sigma, xi = symbols("gamma lambda mu sigma xi")
# p_x = Function("p_x")
p_x = IndexedBase("p_x")
w = Matrix([w1, w2, w3, w4])


def fia(w1):
    return (w1 * v / (1 + C)) * (1 / (Sum(exp(-r * t) * p_x[t], (t, 0, T))))


def uia(w2):
    v0 = w2 * v / ((1 + C) * Sum(p_x[t], (t, 0, T)))
    return v0 * exp((r + (mu - r) * xi - 0.5 * gamma * sigma**2 * xi**2) * t)


def tto(w3):
    return (w3 * v / (1 + C)) * (
        1 / (Sum(exp(-r * t) * (1 - (1 - p_x[t]) ** n), (t, 0, T)))
    )


def uto(w4):
    v0 = w4 * v / ((1 + C) * Sum(1 - (1 - p_x[t]) ** n, (t, 0, T)))
    return v0 * exp((r + (mu - r) * xi - 0.5 * gamma * sigma**2 * xi**2) * t)


summand = (
    binomial(n - 1, k)
    * p_x[t] ** (k + 1)
    * (1 - p_x[t]) ** (n - 1 - k)
    * (1 / (1 - gamma))
    * (fia(w1) + uia(w2) + tto(w3) + uto(w4)) ** (1 - gamma)
)

f = Sum(exp(-r * t) * Sum(summand, (k, 0, n - 1)), (t, 0, T))  # objective function


class DerivativeHelper:
    def __init__(self, ctx):
        self.ctx = ctx
        derivs = []
        for i, var in enumerate([w1, w2, w3, w4]):
            print(f"Computing derivative {i}")
            derivs.append(
                lambdify(
                    w,
                    diff(f, var).subs(
                        [
                            (v, ctx.capital),
                            (n, ctx.n),
                            (gamma, ctx.γ),
                            (C, ctx.risk_loading),
                            (r, ctx.r),
                            (T, ctx.T),
                            (mu, ctx.μ),
                            (sigma, ctx.σ),
                            (xi, ctx.π),
                            (p_x, Array(ctx.p)),
                        ]
                    ),
                )
            )

        self._dfdw1 = derivs[0]
        self._dfdw2 = derivs[1]
        self._dfdw3 = derivs[2]
        self._dfdw4 = derivs[3]

    def dfdw1(self, ww: np.ndarray) -> float:
        return float(self._dfdw1(*ww))

    def dfdw2(self, ww: np.ndarray) -> float:
        return float(self._dfdw2(*ww))

    def dfdw3(self, ww: np.ndarray) -> float:
        return float(self._dfdw3(*ww))

    def dfdw4(self, ww: np.ndarray) -> float:
        return float(self._dfdw4(*ww))


def jac_factory(ctx: Context, helper: DerivativeHelper) -> callable:
    def jac(w: np.ndarray):
        return np.array(
            [
                helper.dfdw1(w),
                helper.dfdw2(w),
                helper.dfdw3(w),
                helper.dfdw4(w),
            ]
        )

    return jac
