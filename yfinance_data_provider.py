import yfinance as yf
import pandas as pd
from typing import List
from data_structures import DataProvider

class YFinanceDataProvider(DataProvider):
    def __init__(self, start: str, end: str = None):
        self.start = start
        self.end = end

    def get_data(self, tickers: List[str]) -> pd.DataFrame:
        """Fetch historical data and strictly enforce (Ticker, Metric) MultiIndex structure."""
        df = yf.download(tickers, start=self.start, end=self.end, auto_adjust=True)

        if len(tickers) > 1:
            # yfinance returns (Metric, Ticker) -> swap to (Ticker, Metric)
            df = df.swaplevel(axis=1).sort_index(axis=1)
        else:
            # For single ticker, ensure it's a MultiIndex (Ticker, Metric)
            ticker = tickers[0]
            if not isinstance(df.columns, pd.MultiIndex):
                df.columns = pd.MultiIndex.from_product([[ticker], df.columns])
            else:
                # If it's already a MultiIndex (sometimes yfinance does this for [ticker]),
                # ensure the levels are (Ticker, Metric)
                if df.columns.levels[0][0] != ticker:
                    df = df.swaplevel(axis=1)
            df = df.sort_index(axis=1)

        return df