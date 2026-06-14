from dataclasses import dataclass
from typing import Optional
import numpy as np

@dataclass
class PortfolioState:
    cash: float
    holdings: np.ndarray
    cost_basis: np.ndarray
    accumulated_tax: float = 0.0
    prev_month: Optional[int] = None
    prev_year: Optional[int] = None
