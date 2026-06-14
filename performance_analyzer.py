import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from backtester import Backtester
from data_structures import Strategy

class PerformanceAnalyzer:
    def __init__(self, portfolio_results: pd.DataFrame, backtester: Backtester, strategy: Strategy, data: pd.DataFrame, risk_free_rate: float = 0.0):
        self.portfolio_results = portfolio_results
        self.backtester = backtester
        self.strategy = strategy
        self.data = data
        self.risk_free_rate = risk_free_rate
        self.metrics = {}

    def analyze(self):
        """Public entry point for full analysis."""
        self._calculate_all_metrics()
        self.plot_performance()

    def _calculate_all_metrics(self):
        """Orchestrates the calculation of all metric groups."""
        df = self.portfolio_results
        if len(df) < 2:
            print("Insufficient data for analysis.")
            return

        self.metrics.update(self._get_return_metrics(df))
        self.metrics.update(self._get_risk_metrics(df))
        self.metrics.update(self._get_drawdown_metrics(df))
        self.metrics.update(self._get_activity_metrics())

        print("\n--- Verified Performance Metrics ---")
        print(pd.Series(self.metrics))

    def _get_return_metrics(self, df: pd.DataFrame) -> dict:
        total_ret = (df['total_equity'].iloc[-1] / df['total_equity'].iloc[0]) - 1
        days = (df.index[-1] - df.index[0]).days
        cagr = ((1 + total_ret) ** (365.25 / days)) - 1 if days > 0 else 0
        return {"Total Return": f"{total_ret:.2%}", "CAGR": f"{cagr:.2%}"}

    def _get_risk_metrics(self, df: pd.DataFrame) -> dict:
        daily_rf = (1 + self.risk_free_rate) ** (1/252) - 1
        excess_returns = df['returns'] - daily_rf
        ann_vol = df['returns'].std() * np.sqrt(252)
        sharpe = (excess_returns.mean() * 252) / ann_vol if ann_vol > 1e-9 else 0
        return {"Annualized Vol": f"{ann_vol:.2%}", "Sharpe Ratio": f"{sharpe:.2f}"}

    def _get_drawdown_metrics(self, df: pd.DataFrame) -> dict:
        rolling_max = df['total_equity'].cummax()
        drawdowns = (df['total_equity'] - rolling_max) / rolling_max
        return {"Max Drawdown": f"{drawdowns.min():.2%}"}

    def _get_activity_metrics(self) -> dict:
        trades_df = self.strategy.generate_trades(self.data)
        count = (trades_df != 0).sum().sum()
        return {"Total Trades": int(count)}

    def plot_performance(self):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
        self._plot_benchmarks(ax1)
        self._plot_equity_curve(ax2)
        plt.tight_layout()
        plt.show()

    def _plot_benchmarks(self, ax):
        tickers = self.data.columns.get_level_values(0).unique()
        for t in tickers:
            bench = self.data[t]['Close'] / self.data[t]['Close'].iloc[0]
            ax.plot(bench, label=f'{t}', alpha=0.5)
        ax.set_title('Normalized Benchmark Growth')
        ax.legend()
        ax.grid(True, alpha=0.3)

    def _plot_equity_curve(self, ax):
        ax.plot(self.portfolio_results['total_equity'], color='green', label='Strategy Equity')
        ax.set_title('Portfolio Equity Curve (Net of Fees & Taxes)')
        ax.fill_between(self.portfolio_results.index, self.portfolio_results['total_equity'],
                         self.backtester.initial_capital, alpha=0.1, color='green')
        ax.legend()
        ax.grid(True, alpha=0.3)