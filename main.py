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

# 2. Shared Backtester Configuration
common_params = {
    'data': qqq_data,
    'initial_capital': 5000.0,
    'monthly_deposit': 1500.0,
    'annual_margin_rate': 0.08,
    'tax_rate': 0.20
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