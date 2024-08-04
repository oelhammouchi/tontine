import numpy as np
from abc import ABC, abstractmethod
from scipy.stats import lognorm

from .data import MortalityData, MarketData
from .utils import kappa, log_normal_mean, log_normal_std, log_normal_pdf


class MarketMixin:
    def __init__(self, **kwargs):
        self.mkt_data = kwargs.pop("mkt_data")
        super().__init__(**kwargs)

    @property
    def μ(self) -> float:
        return self.mkt_data.μ

    @property
    def σ(self) -> float:
        return self.mkt_data.σ

    @property
    def π(self) -> float:
        return self.mkt_data.π

    @property
    @abstractmethod
    def V0(self) -> float:
        pass


class MortalityMixin:
    def __init__(self, **kwargs):
        self.mort_data = kwargs.pop("mort_data")
        super().__init__(**kwargs)

    @property
    def p(self) -> np.ndarray:
        return self.mort_data.p

    @property
    def t(self) -> np.ndarray:
        return self.mort_data.t

    @property
    def r(self):
        return self.mort_data.r


class Instrument(ABC):
    def __init__(self, **kwargs):
        self.prem = kwargs.pop("prem")
        self.risk_loading = kwargs.pop("risk_loading")

    @abstractmethod
    def expected_payoff(self) -> np.ndarray:
        """
        Compute the expected payoff of the instrument at all times contained
        in :attr:`t`
        """
        pass

    @abstractmethod
    def expected_utility(self, γ: float) -> float:
        """
        Compute the total expected utility of the instrument according to
        the CRRA utility function with parameter :math:`\gamma`
        """
        pass


class Annuity(MortalityMixin, Instrument):
    def __init__(self, *, prem: float, risk_loading: float, mort_data: MortalityData):
        super().__init__(mort_data=mort_data, prem=prem, risk_loading=risk_loading)

    @property
    def c(self) -> float:
        return (self.prem / (1 + self.risk_loading)) * (
            1 / (np.sum(np.exp(-self.r * self.t) * self.p))
        )

    def expected_payoff(self) -> np.ndarray:
        return self.c * self.p

    def expected_utility(self, γ: float) -> float:
        return self.c ** (1 - γ) * np.sum(np.exp(-self.r * self.t) * self.p) / (1 - γ)


class Tontine(MortalityMixin, Instrument):
    def __init__(
        self, *, n: int, prem: float, risk_loading: float, mort_data: MortalityData
    ):
        self.n = n
        super().__init__(mort_data=mort_data, prem=prem, risk_loading=risk_loading)

    @property
    def d(self) -> float:
        return (self.prem / (1 + self.risk_loading)) * (
            1 / (np.sum(np.exp(-self.r * self.t) * (1 - (1 - self.p) ** self.n)))
        )

    def expected_payoff(self) -> np.ndarray:
        return self.d * (1 - (1 - self.p) ** self.n)

    def expected_utility(self, γ):
        κ = kappa(self.n, self.p, γ)
        return (
            self.n ** (1 - γ) * self.d ** (1 - γ) * np.sum(np.exp(-self.r * self.t) * κ)
        )


class UnitLinkedAnnuity(MarketMixin, MortalityMixin, Instrument):
    def __init__(
        self,
        *,
        prem: float,
        risk_loading: float,
        mort_data: MortalityData,
        mkt_data: MarketData,
    ):
        super().__init__(
            mort_data=mort_data, mkt_data=mkt_data, prem=prem, risk_loading=risk_loading
        )

    @property
    def V0(self) -> float:
        res = self.prem / ((1 + self.risk_loading) * np.sum(self.p))
        if res < 0:
            raise RuntimeError(f"Negative initial portfolio value {res}")
        return res

    def expected_payoff(self) -> np.ndarray:
        return self.p * self.V0 * np.exp(self.t * (self.r + (self.μ - self.r) * self.π))

    def expected_utility(self, γ: float):
        return (
            self.V0 ** (1 - γ)
            * np.sum(
                np.exp(-γ * self.r * self.t)
                * np.exp(
                    ((self.μ - self.r) * self.π - 0.5 * γ * self.σ**2 * self.π**2)
                    * self.t
                )
            )
            / (1 - γ)
        )

    def psi_dist(self, t: float, x: float):
        mu = (self.r + (self.μ - self.r) * self.π - 0.5 * self.σ**2 * self.π**2) * t

        sigma = self.σ * self.π * np.sqrt(t)

        return log_normal_pdf(x, mu, sigma)

    def psi_dist_params(self, t: float):
        mu = (self.r + (self.μ - self.r) * self.π - 0.5 * self.σ**2 * self.π**2) * t

        sigma = self.σ * self.π * np.sqrt(t)

        return log_normal_mean(mu, sigma), log_normal_std(mu, sigma)


class UnitLinkedTontine(MarketMixin, MortalityMixin, Instrument):
    def __init__(
        self,
        *,
        n: int,
        prem: float,
        risk_loading: float,
        mort_data: MortalityData,
        mkt_data: MarketData,
    ):
        self.n = n
        super().__init__(
            mort_data=mort_data, mkt_data=mkt_data, prem=prem, risk_loading=risk_loading
        )

    @property
    def V0(self) -> float:
        res = self.prem / ((1 + self.risk_loading) * np.sum(1 - (1 - self.p) ** self.n))

        if res < 0:
            raise RuntimeError(f"Negative initial portfolio value {res}")

        return res

    def expected_payoff(self) -> np.ndarray:
        return (
            (1 - (1 - self.p) ** self.n)
            * self.V0
            * np.exp(self.t * (self.r + (self.μ - self.r) * self.π))
        )

    def expected_utility(self, γ: float):
        κ = kappa(self)
        return (
            self.n ** (1 - γ)
            * np.sum(
                np.exp(-self.r * self.t)
                * κ
                * self.V0
                * np.exp(
                    (self.r + (self.μ - self.r) * self.π - γ * self.σ**2 * self.π**2)
                    * self.t
                )
            )
            ** (1 - γ)
            / (1 - γ)
        )

    def psi_dist(self, t: float, x: float):
        mu = (self.r + (self.μ - self.r) * self.π - 0.5 * self.σ**2 * self.π**2) * t

        sigma = self.σ * self.π * np.sqrt(t)

        return log_normal_pdf(x, mu, sigma)

    def psi_dist_params(self, t: float):
        mu = (self.r + (self.μ - self.r) * self.π - 0.5 * self.σ**2 * self.π**2) * t

        sigma = self.σ * self.π * np.sqrt(t)

        return log_normal_mean(mu, sigma), log_normal_std(mu, sigma)
