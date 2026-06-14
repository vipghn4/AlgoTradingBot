import yfinance as yf
import pandas as pd
from typing import List, Optional
from data_structures import DataProvider

class YFinanceDataProvider(DataProvider):
    def __init__(self, start_date: str, end_date: Optional[str] = None):
        """
        Data provider using yfinance to download historical market data.
        """
        if not start_date:
            raise ValueError("Start date must be provided.")
        self.start_date = start_date
        self.end_date = end_date

    def get_data(self, tickers: List[str]) -> pd.DataFrame:
        """Fetch historical data and strictly enforce (Ticker, Metric) MultiIndex structure."""
        if not tickers:
            raise ValueError("Ticker list cannot be empty.")
            
        downloaded_data = yf.download(tickers, start=self.start_date, end=self.end_date, auto_adjust=True)
        
        if downloaded_data.empty:
            raise ValueError(f"No historical data returned for tickers {tickers} between {self.start_date} and {self.end_date}.")
            
        if len(tickers) > 1:
            return self._format_multi_ticker_data(downloaded_data)
        else:
            return self._format_single_ticker_data(downloaded_data, tickers[0])

    def _format_multi_ticker_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Formats columns from (Metric, Ticker) to (Ticker, Metric) MultiIndex."""
        return data.swaplevel(axis=1).sort_index(axis=1)

    def _format_single_ticker_data(self, data: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """Enforces a 2-level (Ticker, Metric) MultiIndex on single ticker DataFrames."""
        if not isinstance(data.columns, pd.MultiIndex):
            data.columns = pd.MultiIndex.from_product([[ticker], data.columns])
        else:
            if data.columns.levels[0][0] != ticker:
                data = data.swaplevel(axis=1)
        return data.sort_index(axis=1)