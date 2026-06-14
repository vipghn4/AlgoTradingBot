from abc import ABC, abstractmethod
import pandas as pd
from typing import List

class DataProvider(ABC):
    @abstractmethod
    def get_data(self, tickers: List[str]) -> pd.DataFrame:
        """Retrieve historical market data for multiple tickers.

        Args:
            tickers: A list of ticker symbols.

        Returns:
            pd.DataFrame: A DataFrame with a DatetimeIndex. Columns should ideally 
                       be a MultiIndex (Ticker, Metric) or structured to clearly 
                       distinguish between assets.
        """
        raise NotImplementedError("Subclasses must implement get_data method.")

class Strategy(ABC):
    @abstractmethod
    def generate_trades(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate trade quantities for multiple tickers at each timestamp.

        Args:
            data: Historical market data (typically a MultiIndex DataFrame).

        Returns:
            pd.DataFrame: A DataFrame indexed by time where each column corresponds 
                          to a ticker, and the values represent the quantity to trade:
                          - Positive (> 0): Buy quantity
                          - Negative (< 0): Sell quantity
                          - Zero (0): No action
        """
        raise NotImplementedError("Subclasses must implement generate_trades method.")