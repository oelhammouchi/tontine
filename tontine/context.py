import numpy as np
import pandas as pd
from dataclasses import dataclass, field


@dataclass
class Context:
    # investor characteristics
    γ: float
    capital: float
    risk_loading: float
    T: int = field(init=False)

    # mortality data
    survival_curve: pd.DataFrame
    t: np.ndarray = field(init=False)
    p: np.ndarray = field(init=False)

    # contract properties
    n: int

    # market data
    r: float
    μ: float
    σ: float
    π: float

    def __post_init__(self):
        self.t = self.survival_curve["t"].to_numpy()
        self.p = np.cumprod(1 - self.survival_curve["prob"].to_numpy())
        self.T = int(self.t.max() - self.t.min())
