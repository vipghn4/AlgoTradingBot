import pandas as pd
from typing import Optional
from data_structures import Strategy

class MovingAverageCrossover(Strategy):
    def __init__(self, short_window: int = 50, long_window: int = 200, trade_quantity: float = 1.0):
        """
        Moving average crossover strategy.
        """
        self.short_window = short_window
        self.long_window = long_window
        self.trade_quantity = trade_quantity

    def generate_trades(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculates trade quantities based on short/long moving average crossovers."""
        trade_signals = pd.DataFrame(index=data.index)
        tickers = data.columns.get_level_values(0).unique()
        
        for ticker in tickers:
            trade_signals[ticker] = self._calculate_ticker_trades(data[ticker])
            
        return trade_signals

    def _calculate_ticker_trades(self, ticker_data: pd.DataFrame) -> pd.Series:
        """Calculates trade signals for a single ticker."""
        close_prices = ticker_data['Close']
        short_moving_avg = close_prices.rolling(window=self.short_window, min_periods=1).mean()
        long_moving_avg = close_prices.rolling(window=self.long_window, min_periods=1).mean()
        
        target_position = (short_moving_avg > long_moving_avg).astype(float) * self.trade_quantity
        return target_position.diff().fillna(0.0)


class DollarCostAverage(Strategy):
    def __init__(self, monthly_share_quantity: Optional[float] = None, monthly_investment_amount: Optional[float] = None):
        """
        Dollar Cost Averaging strategy buying a fixed quantity of shares or a fixed cash amount monthly.
        """
        if monthly_share_quantity is None and monthly_investment_amount is None:
            raise ValueError("Must specify either monthly_share_quantity or monthly_investment_amount")
        self.monthly_share_quantity = monthly_share_quantity
        self.monthly_investment_amount = monthly_investment_amount

    def generate_trades(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generates DCA buy orders at the beginning of each calendar month."""
        tickers = data.columns.get_level_values(0).unique()
        trade_signals = pd.DataFrame(0.0, index=data.index, columns=tickers)
        
        is_first_day_of_month = trade_signals.index.to_series().dt.to_period('M') != \
                                trade_signals.index.to_series().shift(1).dt.to_period('M')
        
        for ticker in tickers:
            if self.monthly_investment_amount is not None:
                prices = data[ticker]['Close']
                # To prevent division by zero or NaN prices
                valid_prices = prices.where(prices > 0.0, np.nan)
                trade_signals.loc[is_first_day_of_month, ticker] = self.monthly_investment_amount / valid_prices.loc[is_first_day_of_month]
                trade_signals[ticker] = trade_signals[ticker].fillna(0.0)
            else:
                trade_signals.loc[is_first_day_of_month, ticker] = self.monthly_share_quantity
                
        return trade_signals


class ValueAveraging(Strategy):
    def __init__(self, monthly_target_increment: float = 2000.0, allow_fractional: bool = True):
        """
        Value Averaging strategy targeting a fixed monthly portfolio value increase.
        """
        self.monthly_target_increment = monthly_target_increment
        self.allow_fractional = allow_fractional

    def generate_trades(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generates value averaging trades to hit monthly target increments."""
        tickers = data.columns.get_level_values(0).unique()
        trade_signals = pd.DataFrame(0.0, index=data.index, columns=tickers)
        
        is_first_day_of_month = data.index.to_series().dt.to_period('M') != \
                                data.index.to_series().shift(1).dt.to_period('M')
        monthly_dates = data.index[is_first_day_of_month]
        
        shares_owned = {ticker: 0.0 for ticker in tickers}
        
        for month_idx, date in enumerate(monthly_dates):
            target_value = (month_idx + 1) * self.monthly_target_increment
            self._process_monthly_trade(date, target_value, tickers, data, shares_owned, trade_signals)
                
        return trade_signals

    def _process_monthly_trade(self, date: pd.Timestamp, target_value: float, tickers: list,
                               data: pd.DataFrame, shares_owned: dict, trade_signals: pd.DataFrame):
        """Calculates and records the monthly trade required to meet target value."""
        for ticker in tickers:
            price = data.loc[date, (ticker, 'Close')]
            current_holding_value = shares_owned[ticker] * price
            
            shortfall = target_value - current_holding_value
            if self.allow_fractional:
                shares_to_purchase = shortfall / price if price > 0.0 else 0.0
            else:
                shares_to_purchase = shortfall // price if price > 0.0 else 0.0
            
            trade_signals.loc[date, ticker] = shares_to_purchase
            shares_owned[ticker] += shares_to_purchase