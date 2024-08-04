from scipy.optimize import minimize, LinearConstraint
import numpy as np
import pandas as pd
import os
from multiprocessing import freeze_support

from tontine.portfolio import Portfolio
from tontine.data import MarketData, MortalityData
from tontine.utils import pi

if __name__ == "__main__":
    freeze_support()
    survival_curve = pd.read_excel(os.path.join("data", "AG2022prob.xlsx"))
    # TODO: internalise the fact that the time parameter is not allowed to be zero
    # because it breaks the log-normal distribution
    survival_curve = survival_curve[survival_curve["t"] > 0]

    mort_data = MortalityData(0.04, survival_curve)
    mkt_data = MarketData(0.08, 0.10, 0.5)

    risk_loadings = {
        "annuity": 0.05,
        "tontine": 0.011,
        "ul_tontine": 0,
        "ul_annuity": 0.038,
    }

    μ = 0.08
    σ = 0.1
    π = pi(μ, σ, mort_data, 9.0)

    mkt_data = MarketData(μ, σ, π)
    ptfl = Portfolio(1e5, 10, 9.0, risk_loadings, mkt_data, mort_data)

    res = ptfl.optimise()

    ptfl.plot().show()
