import pandas as pd
from data_structures import Strategy

class MovingAverageCrossover(Strategy):
    def __init__(self, short_window: int = 50, long_window: int = 200, trade_quantity: float = 1.0):
        self.short_window = short_window
        self.long_window = long_window
        self.trade_quantity = trade_quantity

    def generate_trades(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculates trade quantities for each ticker in the multi-ticker DataFrame."""
        all_trades = pd.DataFrame(index=data.index)
        
        # Get unique tickers from the MultiIndex columns
        tickers = data.columns.get_level_values(0).unique()
        
        for ticker in tickers:
            ticker_close = data[ticker]['Close']
            short_mavg = ticker_close.rolling(window=self.short_window, min_periods=1).mean()
            long_mavg = ticker_close.rolling(window=self.long_window, min_periods=1).mean()
            
            target_position = (short_mavg > long_mavg).astype(float) * self.trade_quantity
            all_trades[ticker] = target_position.diff().fillna(0.0)
            
        return all_trades

class DollarCostAverage(Strategy):
    def __init__(self, monthly_quantity: float = 10.0):
        """
        Implements a DCA strategy buying a fixed quantity at the start of each month.
        
        Args:
            monthly_quantity (float): Number of shares to purchase each interval.
        """
        self.monthly_quantity = monthly_quantity

    def generate_trades(self, data: pd.DataFrame) -> pd.DataFrame:
        all_trades = pd.DataFrame(0.0, index=data.index, columns=data.columns.get_level_values(0).unique())
        
        # Identify the first trading day of each month
        monthly_mask = all_trades.index.to_series().dt.to_period('M') != all_trades.index.to_series().shift(1).dt.to_period('M')
        
        for ticker in all_trades.columns:
            # Apply the buy quantity only on the first day of the month
            all_trades.loc[monthly_mask, ticker] = self.monthly_quantity
            
        return all_trades

class ValueAveraging(Strategy):
    def __init__(self, monthly_value_increase: float = 2000.0):
        """
        Targeting a fixed increase in total portfolio value every month.
        """
        self.monthly_value_increase = monthly_value_increase

    def generate_trades(self, data: pd.DataFrame) -> pd.DataFrame:
        tickers = data.columns.get_level_values(0).unique()
        all_trades = pd.DataFrame(0.0, index=data.index, columns=tickers)
        
        # Track target value over time
        monthly_mask = data.index.to_series().dt.to_period('M') != data.index.to_series().shift(1).dt.to_period('M')
        monthly_dates = data.index[monthly_mask]
        
        shares_owned = {t: 0.0 for t in tickers}
        
        for i, date in enumerate(monthly_dates):
            target_total_value = (i + 1) * self.monthly_value_increase
            
            for ticker in tickers:
                price = data.loc[date, (ticker, 'Close')]
                current_val = shares_owned[ticker] * price
                
                # Calculate how much we need to buy to hit the target value
                shortfall = target_total_value - current_val
                shares_to_buy = shortfall // price
                
                all_trades.loc[date, ticker] = shares_to_buy
                shares_owned[ticker] += shares_to_buy
                
        return all_trades