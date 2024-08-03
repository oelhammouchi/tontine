from scipy.optimize import minimize, LinearConstraint
import numpy as np
import pandas as pd
import os

from tontine.portfolio import Portfolio
from tontine.data import MarketData, MortalityData


survival_curve = pd.read_excel(os.path.join("data", "AG2022prob.xlsx"))
# TODO: internalise the fact that the time parameter is not allowed to be zero
# because it breaks the log-normal distribution
survival_curve = survival_curve[survival_curve["t"] > 0]

mort_data = MortalityData(0.04, survival_curve)
mkt_data = MarketData(0.08, 0.10, 0.5)

risk_loadings = {
    "annuity": 0.02,
    "tontine": 0.02,
    "ul_tontine": 0.02,
    "ul_annuity": 0.02,
}

ptfl = Portfolio(1e5, 10, 0.85, risk_loadings, mkt_data, mort_data)

res = ptfl.optimise()

print(res)
