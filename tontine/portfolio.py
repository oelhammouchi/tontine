import numpy as np
from scipy.special import binom
from scipy.integrate import dblquad
from scipy.optimize import minimize, LinearConstraint
from sympy import binomial
import multiprocessing
from tqdm import tqdm

from .utils import kappa
from .data import MarketData, MortalityData
from .instruments import (
    MarketMixin,
    MortalityMixin,
    Annuity,
    Tontine,
    UnitLinkedAnnuity,
    UnitLinkedTontine,
)

LARGE_PENALTY = -1e5


def u(c, γ):
    return c ** (1 - γ) / (1 - γ)


def u_prime(c, γ):
    return c ** (-γ)


class ProgressCallBack:
    def __init__(self, n_iter: int):
        self.tqdm = tqdm(total=n_iter)
        self.n_iter = n_iter
        self.ctr = 0

    def __call__(self, xk):
        self.tqdm.update()
        self.ctr += 1
        if self.ctr >= self.n_iter:
            raise StopIteration


class LagrangianIntegrand:
    def __init__(
        self, n: int, p: np.ndarray, γ: float, annuity, tontine, ul_annuity, ul_tontine
    ):
        self.n = n
        self.p = p
        self.γ = γ

        self.annuity = annuity
        self.tontine = tontine
        self.ul_annuity = ul_annuity
        self.ul_tontine = ul_tontine

    def __call__(self, x: float, y: float, t: float, w: np.ndarray = None):
        k = np.arange(0, self.n)
        p = self.p[self.annuity.t == t]

        return (
            np.sum(
                binom(self.n - 1, k)
                * u(
                    (k + 1) * w[0] * self.annuity.c
                    + w[1] * self.tontine.d * self.n
                    + w[2] * x * (k + 1)
                    + w[3] * y * self.n,
                    self.γ,
                )
                * p**k
                * (1 - p) ** (self.n - 1 - k)
            )
            * self.ul_annuity.psi_dist(t, x)
            * self.ul_tontine.psi_dist(t, y)
        )


class dLdw1Integrand:
    def __init__(
        self, n: int, p: np.ndarray, γ: float, annuity, tontine, ul_annuity, ul_tontine
    ):
        self.n = n
        self.p = p
        self.γ = γ

        self.annuity = annuity
        self.tontine = tontine
        self.ul_annuity = ul_annuity
        self.ul_tontine = ul_tontine

    def __call__(self, x: float, y: float, t: float, w: np.ndarray):
        k = np.arange(0, self.n)
        p = self.p[self.annuity.t == t]

        return (
            self.annuity.c
            * np.sum(
                binom(self.n - 1, k)
                * (k + 1)
                * u_prime(
                    (k + 1) * w[0] * self.annuity.c
                    + w[1] * self.tontine.d * self.n
                    + w[2] * x * (k + 1)
                    + w[3] * y * self.n,
                    self.γ,
                )
                * p**k
                * (1 - p) ** (self.n - 1 - k)
            )
            * self.ul_annuity.psi_dist(t, x)
            * self.ul_tontine.psi_dist(t, y)
        )


class dLdw2Integrand:
    def __init__(
        self, n: int, p: np.ndarray, γ: float, annuity, tontine, ul_annuity, ul_tontine
    ):
        self.n = n
        self.p = p
        self.γ = γ

        self.annuity = annuity
        self.tontine = tontine
        self.ul_annuity = ul_annuity
        self.ul_tontine = ul_tontine

    def __call__(self, x: float, y: float, t: float, w: np.ndarray):
        k = np.arange(0, self.n)
        p = self.p[self.annuity.t == t]

        return (
            self.tontine.d
            * self.n
            * np.sum(
                binom(self.n - 1, k)
                * u_prime(
                    (k + 1) * w[0] * self.annuity.c
                    + w[1] * self.tontine.d * self.n
                    + w[2] * x * (k + 1)
                    + w[3] * y * self.n,
                    self.γ,
                )
                * p**k
                * (1 - p) ** (self.n - 1 - k)
            )
            * self.ul_annuity.psi_dist(t, x)
            * self.ul_tontine.psi_dist(t, y)
        )


class dLdw3Integrand:
    def __init__(
        self, n: int, p: np.ndarray, γ: float, annuity, tontine, ul_annuity, ul_tontine
    ):
        self.n = n
        self.p = p
        self.γ = γ

        self.annuity = annuity
        self.tontine = tontine
        self.ul_annuity = ul_annuity
        self.ul_tontine = ul_tontine

    def __call__(self, x: float, y: float, t: float, w: np.ndarray):
        k = np.arange(0, self.n)
        p = self.p[self.annuity.t == t]

        return (
            np.sum(
                binom(self.n - 1, k)
                * (k + 1)
                * x
                * u_prime(
                    (k + 1) * w[0] * self.annuity.c
                    + w[1] * self.tontine.d * self.n
                    + w[2] * x * (k + 1)
                    + w[3] * y * self.n,
                    self.γ,
                )
                * p**k
                * (1 - p) ** (self.n - 1 - k)
            )
            * self.ul_annuity.psi_dist(t, x)
            * self.ul_tontine.psi_dist(t, y)
        )


class dLdw4Integrand:
    def __init__(
        self, n: int, p: np.ndarray, γ: float, annuity, tontine, ul_annuity, ul_tontine
    ):
        self.n = n
        self.p = p
        self.γ = γ

        self.annuity = annuity
        self.tontine = tontine
        self.ul_annuity = ul_annuity
        self.ul_tontine = ul_tontine

    def __call__(self, x: float, y: float, t: float, w: np.ndarray):
        k = np.arange(0, self.n)
        p = self.p[self.annuity.t == t]

        return (
            self.n
            * np.sum(
                binom(self.n - 1, k)
                * x
                * u_prime(
                    (k + 1) * w[0] * self.annuity.c
                    + w[1] * self.tontine.d * self.n
                    + w[2] * x * (k + 1)
                    + w[3] * y * self.n,
                    self.γ,
                )
                * p**k
                * (1 - p) ** (self.n - 1 - k)
            )
            * self.ul_annuity.psi_dist(t, x)
            * self.ul_tontine.psi_dist(t, y)
        )


class DoubleQuadWrapper:
    def __init__(self, integrand, w, ul_annuity, ul_tontine):
        self.integrand = integrand
        self.w = w
        self.ul_annuity = ul_annuity
        self.ul_tontine = ul_tontine

    def __call__(self, t: float):
        mu1, sigma1 = self.ul_annuity.psi_dist_params(t)
        lb1, ub1 = (
            mu1 - sigma1,
            mu1 + sigma1,
        )

        mu2, sigma2 = self.ul_annuity.psi_dist_params(t)
        lb2, ub2 = (
            mu2 - sigma2,
            mu2 + sigma2,
        )

        integral, _ = dblquad(
            self.integrand,
            np.max([0, lb1]),
            np.max([0, ub1]),
            np.max([0, lb2]),
            np.max([0, ub2]),
            args=(t, self.w),
        )

        return integral


class Portfolio(MortalityMixin, MarketMixin):
    def __init__(
        self,
        v: float,
        n: int,
        γ: float,
        risk_loadings: dict,
        mkt_data: MarketData,
        mort_data: MortalityData,
    ):
        self.v = v
        self.n = n
        self.γ = γ

        self.mort_data = mort_data
        self.mkt_data = mkt_data
        self.risk_loadings = risk_loadings

        self.pool = multiprocessing.Pool(3 * multiprocessing.cpu_count() // 4)

        self.annuity = Annuity(
            prem=0,
            risk_loading=self.risk_loadings["annuity"],
            mort_data=self.mort_data,
        )

        self.tontine = Tontine(
            prem=0,
            n=self.n,
            risk_loading=self.risk_loadings["tontine"],
            mort_data=self.mort_data,
        )

        self.ul_annuity = UnitLinkedAnnuity(
            prem=0,
            risk_loading=self.risk_loadings["ul_annuity"],
            mort_data=self.mort_data,
            mkt_data=self.mkt_data,
        )

        self.ul_tontine = UnitLinkedTontine(
            n=self.n,
            prem=0,
            risk_loading=self.risk_loadings["ul_tontine"],
            mort_data=self.mort_data,
            mkt_data=self.mkt_data,
        )

    def __del__(self) -> None:
        self.pool.close()

    def _guard(f: callable) -> callable:
        def guarded(self, w: np.ndarray):
            if np.any(w < 0):
                return LARGE_PENALTY

            self.annuity.prem = self.v * w[0]
            self.tontine.prem = self.v * w[1]
            self.ul_annuity.prem = self.v * w[2]
            self.ul_tontine.prem = self.v * w[3]

            return f(self, w)

        return guarded

    @_guard
    def L(self, w: np.ndarray):
        inner_factors = np.array(
            self.pool.map(
                DoubleQuadWrapper(
                    LagrangianIntegrand(
                        self.n,
                        self.p,
                        self.γ,
                        self.annuity,
                        self.tontine,
                        self.ul_annuity,
                        self.ul_tontine,
                    ),
                    w,
                    self.ul_annuity,
                    self.ul_tontine,
                ),
                self.t,
            )
        )
        κ = kappa(self.n, self.p, self.γ)
        res = np.sum(np.exp(-self.annuity.r * self.annuity.t) * κ * inner_factors) / (
            1 - self.γ
        )

        return res

    @_guard
    def _dLdw1(self, w: np.ndarray):
        if np.any(w < 0):
            return LARGE_PENALTY

        inner_factors = self.pool.map(
            DoubleQuadWrapper(
                dLdw1Integrand(
                    self.n,
                    self.p,
                    self.γ,
                    self.annuity,
                    self.tontine,
                    self.ul_annuity,
                    self.ul_tontine,
                ),
                w,
                self.ul_annuity,
                self.ul_tontine,
            ),
            self.t,
        )
        κ = kappa(self.n, self.p, self.γ)
        res = np.sum(np.exp(-self.annuity.r * self.annuity.t) * κ * inner_factors) / (
            1 - self.γ
        )

        return res

    @_guard
    def _dLdw2(self, w: np.ndarray) -> float:
        if np.any(w < 0):
            return LARGE_PENALTY

        inner_factors = self.pool.map(
            DoubleQuadWrapper(
                dLdw2Integrand(
                    self.n,
                    self.p,
                    self.γ,
                    self.annuity,
                    self.tontine,
                    self.ul_annuity,
                    self.ul_tontine,
                ),
                w,
                self.ul_annuity,
                self.ul_tontine,
            ),
            self.t,
        )
        κ = kappa(self.n, self.p, self.γ)
        res = (
            self.n
            * self.tontine.d
            * np.sum(np.exp(-self.annuity.r * self.annuity.t) * κ * inner_factors)
            / (1 - self.γ)
        )

        return res

    @_guard
    def _dLdw3(self, w: np.ndarray) -> float:
        if np.any(w < 0):
            return LARGE_PENALTY

        inner_factors = self.pool.map(
            DoubleQuadWrapper(
                dLdw3Integrand(
                    self.n,
                    self.p,
                    self.γ,
                    self.annuity,
                    self.tontine,
                    self.ul_annuity,
                    self.ul_tontine,
                ),
                w,
                self.ul_annuity,
                self.ul_tontine,
            ),
            self.t,
        )
        κ = kappa(self.n, self.p, self.γ)
        res = np.sum(np.exp(-self.annuity.r * self.annuity.t) * κ * inner_factors) / (
            1 - self.γ
        )

        return res

    @_guard
    def _dLdw4(self, w: np.ndarray) -> float:
        if np.any(w < 0):
            return LARGE_PENALTY

        inner_factors = self.pool.map(
            DoubleQuadWrapper(
                dLdw4Integrand(
                    self.n,
                    self.p,
                    self.γ,
                    self.annuity,
                    self.tontine,
                    self.ul_annuity,
                    self.ul_tontine,
                ),
                w,
                self.ul_annuity,
                self.ul_tontine,
            ),
            self.t,
        )
        κ = kappa(self.n, self.p, self.γ)
        res = np.sum(np.exp(-self.annuity.r * self.annuity.t) * κ * inner_factors) / (
            1 - self.γ
        )

        return res

    def dLdw(self, w: np.ndarray) -> np.ndarray:
        return np.array(
            [self._dLdw1(w), self._dLdw2(w), self._dLdw3(w), self._dLdw4(w)]
        )

    def optimise(self):
        max_iter = int(1e3)

        res = minimize(
            lambda w: -self.L(w),
            np.array([0.1, 0.1, 0.4, 0.4]),
            method="cobyqa",
            constraints=[
                LinearConstraint(np.eye(4), np.zeros(4)),
                LinearConstraint(np.ones((4,)), 1, 1),
            ],
            jac=self.dLdw,
            tol=None,
            callback=ProgressCallBack(max_iter),
            options={"maxiter": max_iter},
        )

        return res
