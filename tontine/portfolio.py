import numpy as np
from scipy.special import binom
from scipy.integrate import quad
from scipy.optimize import minimize, LinearConstraint, OptimizeResult
from sympy import binomial
import multiprocessing
from tqdm import tqdm
import seaborn as sns
import pandas as pd
from copy import copy

from .data import MarketData, MortalityData
from .instruments import (
    MarketMixin,
    MortalityMixin,
    Annuity,
    Tontine,
    UnitLinkedAnnuity,
    UnitLinkedTontine,
)

OBJ_PENALTY = -1e5
JAC_PENALTY = 1e5


def u(c, γ):
    return c ** (1 - γ) / (1 - γ) if γ != 1 else np.log(c)


def u_prime(c, γ):
    return c ** (-γ) if γ != 1 else 1 / c


class ProgressCallBack:
    def __init__(self, n_iter: int, show: bool):
        self.tqdm = tqdm(total=n_iter, disable=not show)
        self.n_iter = n_iter
        self.ctr = 0

    def __call__(self, intermediate_result: OptimizeResult):
        self.tqdm.update()
        self.ctr += 1
        # if self.ctr >= self.n_iter:
        #     raise StopIteration


class MinimiseHelper:
    def __init__(self, ptfl: "Portfolio"):
        self.ptfl = ptfl

    def __call__(self, candidate: np.ndarray, max_iter: int, progress: bool):
        res = minimize(
            lambda w: -self.ptfl.L(w),
            candidate,
            method="trust-constr",
            constraints=[
                LinearConstraint(np.ones((4,)), 1, 1),
                LinearConstraint(np.eye(4), np.zeros(4)),
            ],
            jac=self.ptfl.dLdw,
            tol=1e-8,
            callback=ProgressCallBack(max_iter, show=progress),
            options={"maxiter": max_iter},
        )

        return res.x, res.fun


class LagrangianIntegrand:
    def __init__(self, ptfl: "Portfolio"):
        self.ptfl = ptfl

    def __call__(self, x: float, t: float):
        k = np.arange(0, self.ptfl.n)
        p = self.ptfl.p[self.ptfl.annuity.t == t]

        return np.sum(
            binom(self.ptfl.n - 1, k)
            * p ** (k + 1)
            * (1 - p) ** (self.ptfl.n - 1 - k)
            * u(
                self.ptfl.annuity.c
                + self.ptfl.tontine.d * self.ptfl.n / (k + 1)
                + self.ptfl.ul_annuity.V0 * x
                + self.ptfl.ul_tontine.V0 * x * self.ptfl.n / (k + 1),
                self.ptfl.γ,
            )
        ) * self.ptfl.ul_annuity.psi_pdf(t, x)


class dLdw1Integrand:
    def __init__(self, ptfl: "Portfolio"):
        self.ptfl = ptfl

    def __call__(self, x: float, t: float):
        k = np.arange(0, self.ptfl.n)
        p = self.ptfl.p[self.ptfl.annuity.t == t]

        return (
            (
                self.ptfl.v
                / (
                    (1 + self.ptfl.annuity.risk_loading)
                    * np.sum(np.exp(-self.ptfl.annuity.r * self.ptfl.t) * self.ptfl.p)
                )
            )
            * np.sum(
                binom(self.ptfl.n - 1, k)
                * p ** (k + 1)
                * (1 - p) ** (self.ptfl.n - 1 - k)
                * u_prime(
                    self.ptfl.annuity.c
                    + self.ptfl.tontine.d * self.ptfl.n / (k + 1)
                    + self.ptfl.ul_annuity.V0 * x
                    + self.ptfl.ul_tontine.V0 * x * self.ptfl.n / (k + 1),
                    self.ptfl.γ,
                )
            )
            * self.ptfl.ul_annuity.psi_pdf(t, x)
        )


class dLdw2Integrand:
    def __init__(self, ptfl: "Portfolio"):
        self.ptfl = ptfl

    def __call__(self, x: float, t: float):
        k = np.arange(0, self.ptfl.n)
        p = self.ptfl.p[self.ptfl.tontine.t == t]

        return (
            self.ptfl.v
            / (
                (1 + self.ptfl.tontine.risk_loading)
                * (
                    np.sum(
                        np.exp(-self.ptfl.r * self.ptfl.t)
                        * (1 - (1 - self.ptfl.p) ** self.ptfl.n)
                    )
                )
            )
            * np.sum(
                binom(self.ptfl.n - 1, k)
                * p ** (k + 1)
                * (1 - p) ** (self.ptfl.n - 1 - k)
                * u_prime(
                    self.ptfl.annuity.c
                    + self.ptfl.tontine.d * self.ptfl.n / (k + 1)
                    + self.ptfl.ul_annuity.V0 * x
                    + self.ptfl.ul_tontine.V0 * x * self.ptfl.n / (k + 1),
                    self.ptfl.γ,
                )
            )
            * self.ptfl.ul_annuity.psi_pdf(t, x)
        )


class dLdw3Integrand:
    def __init__(self, ptfl: "Portfolio"):
        self.ptfl = ptfl

    def __call__(self, x: float, t: float):
        k = np.arange(0, self.ptfl.n)
        p = self.ptfl.p[self.ptfl.ul_annuity.t == t]

        res = (
            (
                self.ptfl.v
                / ((1 + self.ptfl.ul_annuity.risk_loading) * np.sum(self.ptfl.p))
            )
            * np.sum(
                binom(self.ptfl.n - 1, k)
                * p ** (k + 1)
                * (1 - p) ** (self.ptfl.n - 1 - k)
                * x
                * u_prime(
                    self.ptfl.annuity.c
                    + self.ptfl.tontine.d * self.ptfl.n / (k + 1)
                    + self.ptfl.ul_annuity.V0 * x
                    + self.ptfl.ul_tontine.V0 * x * self.ptfl.n / (k + 1),
                    self.ptfl.γ,
                )
            )
            * self.ptfl.ul_annuity.psi_pdf(t, x)
        )

        return res


class dLdw4Integrand:
    def __init__(self, ptfl: "Portfolio"):
        self.ptfl = ptfl

    def __call__(self, x: float, t: float):
        k = np.arange(0, self.ptfl.n)
        p = self.ptfl.p[self.ptfl.ul_tontine.t == t]

        return (
            self.ptfl.v
            / (
                (1 + self.ptfl.ul_tontine.risk_loading)
                * np.sum(1 - (1 - self.ptfl.p) ** self.ptfl.n)
            )
            * np.sum(
                binom(self.ptfl.n - 1, k)
                * p ** (k + 1)
                * (1 - p) ** (self.ptfl.n - 1 - k)
                * (x * self.ptfl.n / (k + 1))
                * u_prime(
                    self.ptfl.annuity.c
                    + self.ptfl.tontine.d * self.ptfl.n / (k + 1)
                    + self.ptfl.ul_annuity.V0 * x
                    + self.ptfl.ul_tontine.V0 * x * self.ptfl.n / (k + 1),
                    self.ptfl.γ,
                )
            )
            * self.ptfl.ul_annuity.psi_pdf(t, x)
        )


class QuadWrapper:
    def __init__(self, integrand, ul_annuity, ul_tontine):
        self.integrand = integrand
        self.ul_annuity = ul_annuity
        self.ul_tontine = ul_tontine

    def __call__(self, t: float):
        mu1, sigma1 = self.ul_annuity.psi_params(t)
        lb1, ub1 = (
            mu1 - 2 * sigma1,
            mu1 + 2 * sigma1,
        )

        integral, _ = quad(
            self.integrand,
            0,
            np.inf,
            args=(t,),
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

    # def __copy__(self):
    #     cls = self.__class__
    #     new_obj = cls.__new__(cls)

    #     for attr, value in self.__dict__.items():
    #         if attr != "pool":
    #             setattr(new_obj, attr, copy(value))

    #     new_obj.pool = None

    #     return new_obj

    def payout(self) -> float:
        return (
            self.annuity.expected_payoff()
            + self.tontine.expected_payoff()
            + self.ul_annuity.expected_payoff()
            + self.ul_tontine.expected_payoff()
        )

    def L(self, w: np.ndarray):
        if np.any(w < 0):
            return OBJ_PENALTY

        self.annuity.prem = self.v * w[0]
        self.tontine.prem = self.v * w[1]
        self.ul_annuity.prem = self.v * w[2]
        self.ul_tontine.prem = self.v * w[3]

        inner_factors = np.array(
            list(
                map(
                    QuadWrapper(
                        LagrangianIntegrand(self),
                        self.ul_annuity,
                        self.ul_tontine,
                    ),
                    self.t,
                )
            )
        )

        return np.sum(np.exp(-self.annuity.r * self.annuity.t) * inner_factors)

    def _dLdw1(self):
        inner_factors = np.array(
            list(
                map(
                    QuadWrapper(
                        dLdw1Integrand(self),
                        self.ul_annuity,
                        self.ul_tontine,
                    ),
                    self.t,
                )
            )
        )

        return np.sum(np.exp(-self.annuity.r * self.annuity.t) * inner_factors)

    def _dLdw2(self) -> float:
        inner_factors = np.array(
            list(
                map(
                    QuadWrapper(
                        dLdw2Integrand(self),
                        self.ul_annuity,
                        self.ul_tontine,
                    ),
                    self.t,
                )
            )
        )

        return np.sum(np.exp(-self.annuity.r * self.annuity.t) * inner_factors)

    def _dLdw3(self) -> float:
        inner_factors = np.array(
            list(
                map(
                    QuadWrapper(
                        dLdw3Integrand(self),
                        self.ul_annuity,
                        self.ul_tontine,
                    ),
                    self.t,
                )
            )
        )

        return np.sum(np.exp(-self.annuity.r * self.annuity.t) * inner_factors)

    def _dLdw4(self) -> float:
        inner_factors = np.array(
            list(
                map(
                    QuadWrapper(
                        dLdw4Integrand(self),
                        self.ul_annuity,
                        self.ul_tontine,
                    ),
                    self.t,
                )
            )
        )

        return np.sum(np.exp(-self.annuity.r * self.annuity.t) * inner_factors)

    def dLdw(self, w: np.ndarray) -> np.ndarray:
        if np.any(w < 0):
            return np.ones(4) * JAC_PENALTY

        self.annuity.prem = self.v * w[0]
        self.tontine.prem = self.v * w[1]
        self.ul_annuity.prem = self.v * w[2]
        self.ul_tontine.prem = self.v * w[3]

        jac = np.array([self._dLdw1(), self._dLdw2(), self._dLdw3(), self._dLdw4()])

        print(f"Weights: {w}")
        print(f"Jacobian: {jac}")

        return jac

    def optimise(self, progress=True):
        pool = multiprocessing.Pool(3 * multiprocessing.cpu_count() // 4)
        max_iter = int(100)

        candidates = np.random.dirichlet(np.ones(4), 10)
        minimiser = MinimiseHelper(self)

        results = list(
            pool.starmap(
                minimiser, [(candidate, max_iter, progress) for candidate in candidates]
            )
        )

        scores = [res[1] for res in results]
        weights = [res[0] for res in results]
        w = weights[np.array(scores).argmin()]

        self.w = w
        self.annuity.prem = self.w[0] * self.v
        self.tontine.prem = self.w[1] * self.v
        self.ul_annuity.prem = self.w[2] * self.v
        self.ul_tontine.prem = self.w[3] * self.v

        pool.close()
        pool.join()

        return w

    def plot(self, ax=None):
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

        sns.lineplot(plt_df, x="Time", y="Payout", hue="Instrument", ax=ax)
