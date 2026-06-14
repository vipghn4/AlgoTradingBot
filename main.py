import os
import sys

# Automatically run within the local virtual environment (.venv) if it exists
_current_dir = os.path.dirname(os.path.abspath(__file__))
_venv_python = os.path.join(_current_dir, ".venv", "bin", "python3")
if os.path.exists(_venv_python) and not sys.executable.startswith(os.path.join(_current_dir, ".venv")):
    os.execv(_venv_python, [_venv_python] + sys.argv)

from yfinance_data_provider import YFinanceDataProvider
from utils import get_market_risk_free_rate
from strategies import DollarCostAverage, ValueAveraging, MovingAverageCrossover
from backtester import Backtester
from performance_analyzer import PerformanceAnalyzer

# 1. Centralized Data Fetching
data_provider = YFinanceDataProvider(start='2015-01-01')
qqq_data = data_provider.get_data(['QQQ'])
rf_rate = get_market_risk_free_rate()

# 2. Shared Backtester Configuration (UK Trader trading US Stocks: GIA with 0.35% FX fee & 20% CGT)
common_params = {
    'data': qqq_data,
    'initial_capital': 5000.0,
    'monthly_deposit': 1500.0,
    'commission_pct': 0.0035,  # 0.35% FX conversion fee on USD assets
    'tax_rate': 0.20,           # 20% UK Capital Gains Tax (Higher-rate band)
    'tax_deferred': True,       # Taxes settled annually at tax year-end
    'allow_margin': False,      # Cash account (no leverage/borrowing)
    'allow_short': False,       # Short selling disabled
    'allow_fractional': True,   # Fractional shares enabled
    'slippage_pct': 0.0005,     # 0.05% bid-ask execution spread
    'execution_delay': 1,       # 1-day execution delay
    'execution_price_type': 'Open' # Execute at next US market open
}

# 3. Define Strategies to compare
comparison_configs = [
    ("Dollar Cost Averaging", DollarCostAverage(monthly_quantity=10.0)),
    ("Value Averaging", ValueAveraging(monthly_value_increase=2500.0)),
    ("MA Crossover (50/200)", MovingAverageCrossover(short_window=50, long_window=200, trade_quantity=10.0))
]

# 4. Run and Analyze All Strategies
for name, strategy in comparison_configs:
    print(f"\n{'='*20} Running: {name} {'='*20}")
    tester = Backtester(strategy=strategy, **common_params)
    results = tester.run()
    analyzer = PerformanceAnalyzer(results, tester, strategy, qqq_data, risk_free_rate=rf_rate)
    analyzer.analyze()