import numpy as np
import pandas as pd
import os
from multiprocessing import freeze_support
import matplotlib.pyplot as plt

from tontine.portfolio import Portfolio
from tontine.data import MarketData, MortalityData
from tontine.utils import pi

if __name__ == "__main__":
    freeze_support()
    survival_curve = pd.read_excel(os.path.join("data", "AG2022prob.xlsx"))
    survival_curve = survival_curve[survival_curve["t"] > 0]

    mort_data = MortalityData(0.04, survival_curve)
    mkt_data = MarketData(0.08, 0.10, 0.5)

    risk_loadings = {
        "annuity": 0,
        "tontine": 0,
        "ul_tontine": 0,
        "ul_annuity": 0,
    }

    μ = 0.08
    σ = 0.1

    risk_aversions = [-1.5, 0, 1.5]  # risk-seeking, risk-neutral, risk-averse

    fig, axs = plt.subplots(2, 2, sharey=True)

    for i, γ in enumerate(risk_aversions):
        π = pi(μ, σ, mort_data, γ)
        mkt_data = MarketData(μ, σ, π)
        ptfl = Portfolio(1e5, 100, γ, risk_loadings, mkt_data, mort_data)

        res = ptfl.optimise(progress=False)

        ptfl.plot(ax=axs.reshape(-1)[i])

    plt.show()
