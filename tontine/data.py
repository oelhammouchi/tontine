import numpy as np
import pandas as pd
from dataclasses import dataclass, field


@dataclass
class MarketData:
    μ: float
    σ: float
    π: float


@dataclass
class MortalityData:
    r: float
    survival_curve: pd.DataFrame
    t: np.ndarray = field(init=False)
    p: np.ndarray = field(init=False)
    T: int = field(init=False)

    def __post_init__(self):
        self.t = self.survival_curve["t"].to_numpy()
        self.p = np.cumprod(1 - self.survival_curve["prob"].to_numpy())
        self.T = int(self.t.max() - self.t.min())
