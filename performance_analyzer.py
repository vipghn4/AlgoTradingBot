import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Optional
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

    def analyze(self, save_path: Optional[str] = None) -> plt.Figure:
        """Public entry point for full analysis."""
        self._calculate_all_metrics()
        fig = self.plot_performance()
        if save_path:
            fig.savefig(save_path)
            plt.close(fig)
        return fig

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
        total_ret = (df['returns'] + 1.0).prod() - 1.0
        days = (df.index[-1] - df.index[0]).days
        cagr = ((1.0 + total_ret) ** (365.25 / days)) - 1.0 if days > 0 else 0.0
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
        count = self.backtester.state.executed_trades_count if hasattr(self.backtester, 'state') else 0
        return {"Total Trades": int(count)}

    def plot_performance(self) -> plt.Figure:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
        self._plot_benchmarks(ax1)
        self._plot_equity_curve(ax2)
        plt.tight_layout()
        return fig

    @classmethod
    def plot_comparison(cls, analyzers: dict) -> plt.Figure:
        """
        Generates a consolidated comparison plot of equity curves and drawdowns
        across multiple PerformanceAnalyzer instances.
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
        
        # Subplot 1: Absolute Equity Curve Comparison
        for name, analyzer in analyzers.items():
            ax1.plot(analyzer.portfolio_results['total_equity'], label=name, alpha=0.8)
            
        # Plot Cumulative Invested Capital as a baseline using parameters from the first analyzer
        if analyzers:
            first_analyzer = next(iter(analyzers.values()))
            market_data = first_analyzer.data
            is_new_month = market_data.index.to_series().dt.to_period('M') != market_data.index.to_series().shift(1).dt.to_period('M')
            config = first_analyzer.backtester.config
            initial_usd = config.initial_capital
            monthly_usd = config.monthly_deposit * config.deposit_fx_rate
            invested_capital = (is_new_month.astype(float) * monthly_usd).cumsum() + initial_usd
            ax1.plot(invested_capital, label='Cumulative Invested Capital', color='black', linestyle='--', alpha=0.7)
            
        ax1.set_title('Portfolio Equity Comparison ($ USD)')
        ax1.set_ylabel('Total Value ($)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Subplot 2: Under-water Drawdown Comparison
        for name, analyzer in analyzers.items():
            rolling_max = analyzer.portfolio_results['total_equity'].cummax()
            drawdowns = (analyzer.portfolio_results['total_equity'] - rolling_max) / rolling_max
            ax2.plot(drawdowns * 100.0, label=name, alpha=0.8)
            
        ax2.set_title('Portfolio Drawdown Comparison (%)')
        ax2.set_ylabel('Drawdown %')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig

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
                         self.backtester.config.initial_capital, alpha=0.1, color='green')
        ax.legend()
        ax.grid(True, alpha=0.3)