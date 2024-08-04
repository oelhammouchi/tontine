import numpy as np
from scipy.special import binom
from scipy.integrate import dblquad
from scipy.optimize import minimize, LinearConstraint, OptimizeResult
from sympy import binomial
import multiprocessing
from tqdm import tqdm
import seaborn as sns
import pandas as pd
from copy import copy

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

    def __call__(self, intermediate_result: OptimizeResult):
        self.tqdm.update()
        self.ctr += 1
        if self.ctr >= self.n_iter:
            raise StopIteration


class LagrangianIntegrand:
    def __init__(self, ptfl: "Portfolio"):
        self.ptfl = ptfl

    def __call__(self, x: float, y: float, t: float, w: np.ndarray = None):
        k = np.arange(0, self.ptfl.n)
        p = self.ptfl.p[self.ptfl.annuity.t == t]

        return (
            np.sum(
                binom(self.ptfl.n - 1, k)
                * u(
                    (k + 1) * self.ptfl.annuity.c
                    + self.ptfl.tontine.d * self.ptfl.n
                    + x * (k + 1)
                    + y * self.ptfl.n,
                    self.ptfl.γ,
                )
                * p**k
                * (1 - p) ** (self.ptfl.n - 1 - k)
            )
            * self.ptfl.ul_annuity.psi_dist(t, x)
            * self.ptfl.ul_tontine.psi_dist(t, y)
        )


class dLdw1Integrand:
    def __init__(self, ptfl: "Portfolio"):
        self.ptfl = ptfl

    def __call__(self, x: float, y: float, t: float, w: np.ndarray):
        k = np.arange(0, self.ptfl.n)
        p = self.ptfl.p[self.ptfl.annuity.t == t]

        return (
            (
                self.ptfl.v
                / (
                    (1 + self.ptfl.annuity.risk_loading)
                    * np.sum(np.exp(-self.ptfl.annuity.r * t) * p)
                )
            )
            * np.sum(
                binom(self.ptfl.n - 1, k)
                * (k + 1)
                * u_prime(
                    (k + 1) * self.ptfl.annuity.c
                    + p * self.ptfl.tontine.d * self.ptfl.n
                    + self.ptfl.ul_annuity.V0 * x * (k + 1)
                    + self.ptfl.ul_tontine.V0 * p * y * self.ptfl.n,
                    self.ptfl.γ,
                )
                * p**k
                * (1 - p) ** (self.ptfl.n - 1 - k)
            )
            * self.ptfl.ul_annuity.psi_dist(t, x)
            * self.ptfl.ul_tontine.psi_dist(t, y)
        )


class dLdw2Integrand:
    def __init__(self, ptfl: "Portfolio"):
        self.ptfl = ptfl

    def __call__(self, x: float, y: float, t: float, w: np.ndarray):
        k = np.arange(0, self.ptfl.n)
        p = self.ptfl.p[self.ptfl.tontine.t == t]

        return (
            self.ptfl.n
            * p
            * (self.ptfl.v / (1 + self.ptfl.tontine.risk_loading))
            * (
                1
                / (
                    np.sum(
                        np.exp(-self.ptfl.r * self.ptfl.t)
                        * (1 - (1 - self.ptfl.p) ** self.ptfl.n)
                    )
                )
            )
            * np.sum(
                binom(self.ptfl.n - 1, k)
                * u_prime(
                    (k + 1) * self.ptfl.annuity.c
                    + p * self.ptfl.tontine.d * self.ptfl.n
                    + self.ptfl.ul_annuity.V0 * x * (k + 1)
                    + self.ptfl.ul_tontine.V0 * p * y * self.ptfl.n,
                    self.ptfl.γ,
                )
                * p**k
                * (1 - p) ** (self.ptfl.n - 1 - k)
            )
            * self.ptfl.ul_annuity.psi_dist(t, x)
            * self.ptfl.ul_tontine.psi_dist(t, y)
        )


class dLdw3Integrand:
    def __init__(self, ptfl: "Portfolio"):
        self.ptfl = ptfl

    def __call__(self, x: float, y: float, t: float, w: np.ndarray):
        k = np.arange(0, self.ptfl.n)
        p = self.ptfl.p[self.ptfl.ul_annuity.t == t]

        factor = self.ptfl.v / (
            (1 + self.ptfl.ul_annuity.risk_loading) * np.sum(self.ptfl.p)
        )

        return (
            factor
            * np.sum(
                binom(self.ptfl.n - 1, k)
                * (k + 1)
                * x
                * u_prime(
                    (k + 1) * self.ptfl.annuity.c
                    + p * self.ptfl.tontine.d * self.ptfl.n
                    + self.ptfl.ul_annuity.V0 * x * (k + 1)
                    + p * y * self.ptfl.ul_tontine.V0 * self.ptfl.n,
                    self.ptfl.γ,
                )
                * p**k
                * (1 - p) ** (self.ptfl.n - 1 - k)
            )
            * self.ptfl.ul_annuity.psi_dist(t, x)
            * self.ptfl.ul_tontine.psi_dist(t, y)
        )


class dLdw4Integrand:
    def __init__(self, ptfl: "Portfolio"):
        self.ptfl = ptfl

    def __call__(self, x: float, y: float, t: float, w: np.ndarray):
        k = np.arange(0, self.ptfl.n)
        p = self.ptfl.p[self.ptfl.ul_tontine.t == t]

        factor = (
            p
            * self.ptfl.v
            / (
                (1 + self.ptfl.ul_tontine.risk_loading)
                * np.sum(1 - (1 - self.ptfl.p) ** self.ptfl.n)
            )
        )

        return (
            self.ptfl.n
            * factor
            * np.sum(
                binom(self.ptfl.n - 1, k)
                * y
                * u_prime(
                    (k + 1) * self.ptfl.annuity.c
                    + p * self.ptfl.tontine.d * self.ptfl.n
                    + self.ptfl.ul_annuity.V0 * x * (k + 1)
                    + p * self.ptfl.ul_tontine.V0 * y * self.ptfl.n,
                    self.ptfl.γ,
                )
                * p**k
                * (1 - p) ** (self.ptfl.n - 1 - k)
            )
            * self.ptfl.ul_annuity.psi_dist(t, x)
            * self.ptfl.ul_tontine.psi_dist(t, y)
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
        if self.pool is not None:
            self.pool.close()

    def __copy__(self):
        cls = self.__class__
        new_obj = cls.__new__(cls)

        for attr, value in self.__dict__.items():
            if attr != "pool":
                setattr(new_obj, attr, copy(value))

        new_obj.pool = None

        return new_obj

    def payout(self) -> float:
        return (
            self.annuity.expected_payoff()
            + self.tontine.expected_payoff()
            + self.ul_annuity.expected_payoff()
            + self.ul_tontine.expected_payoff()
        )

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
        context = copy(self)

        inner_factors = np.array(
            self.pool.map(
                DoubleQuadWrapper(
                    LagrangianIntegrand(context),
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

        context = copy(self)

        inner_factors = self.pool.map(
            DoubleQuadWrapper(
                dLdw1Integrand(context),
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

        context = copy(self)

        inner_factors = self.pool.map(
            DoubleQuadWrapper(
                dLdw2Integrand(context),
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

        context = copy(self)

        inner_factors = self.pool.map(
            DoubleQuadWrapper(
                dLdw3Integrand(context),
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

        context = copy(self)

        inner_factors = self.pool.map(
            DoubleQuadWrapper(
                dLdw4Integrand(context),
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
        max_iter = int(100)

        res = minimize(
            lambda w: -self.L(w),
            np.array([0.4, 0.4, 0.1, 0.1]),
            method="trust-constr",
            constraints=[
                LinearConstraint(np.eye(4), np.zeros(4)),
                LinearConstraint(np.ones((4,)), 1, 1),
            ],
            jac=self.dLdw,
            tol=None,
            callback=ProgressCallBack(max_iter),
            options={
                "maxiter": max_iter,
                "xtol": np.finfo(float).eps,
                "gtol": np.finfo(float).eps,
            },
        )

        self.w = res.x
        self.annuity.prem = self.w[0] * self.v
        self.tontine.prem = self.w[1] * self.v
        self.ul_annuity.prem = self.w[2] * self.v
        self.ul_tontine.prem = self.w[3] * self.v

        return res

    def plot(self):
        plt_df = pd.DataFrame(
            {
                "Time": self.t,
                "Portfolio": self.payout(),
                "FIA": self.annuity.expected_payoff(),
                "TTO": self.tontine.expected_payoff(),
                "UIA": self.ul_annuity.expected_payoff(),
                "UTO": self.ul_tontine.expected_payoff(),
            }
        )

        plt_df = plt_df.melt(
            id_vars=["Time"],
            value_vars=["FIA", "TTO", "UIA", "UTO", "Portfolio"],
            var_name="Instrument",
            value_name="Payout",
        )

        plot = sns.lineplot(plt_df, x="Time", y="Payout", hue="Instrument")
        return plot.get_figure()
