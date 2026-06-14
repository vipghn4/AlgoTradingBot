import math
from typing import Optional, List
import numpy as np
import pandas as pd
from data_structures import Strategy
from portfolio_state import PortfolioState
from trading_models import (
    CommissionModel, DefaultCommissionModel,
    SlippageModel, DefaultSlippageModel,
    TaxModel, DefaultTaxModel
)

# =====================================================================
# CORE BACKTESTER ENGINE
# =====================================================================

class Backtester:
    def __init__(self, data: pd.DataFrame, strategy: Strategy,
                 initial_capital: float = 1000.0,
                 monthly_deposit: float = 500.0,
                 commission_per_share: float = 0.0,
                 commission_pct: float = 0.0,
                 commission_flat: float = 0.0,
                 commission_min: float = 0.0,
                 tax_rate: float = 0.0,
                 tax_deferred: bool = True,
                 annual_margin_rate: float = 0.08,
                 annual_borrow_rate: float = 0.02,
                 allow_margin: bool = False,
                 max_leverage: float = 2.0,
                 allow_short: bool = False,
                 allow_fractional: bool = True,
                 slippage_pct: float = 0.0005,
                 execution_delay: int = 1,
                 execution_price_type: str = 'Open',
                 commission_model: Optional[CommissionModel] = None,
                 slippage_model: Optional[SlippageModel] = None,
                 tax_model: Optional[TaxModel] = None):
        """
        Initializes the Backtester with chronological market data and a trading strategy.

        Args:
            data: MultiIndex DataFrame containing historical prices level-indexed by (Ticker, Price).
            strategy: The Strategy object generating trade signals.
            initial_capital: Starting cash balance of the portfolio.
            monthly_deposit: Periodic savings capital added to the cash balance monthly.
            commission_per_share: Per-share fee charged by the broker on transactions.
            commission_pct: Percentage-based fee of total transaction value.
            commission_flat: Fixed flat fee per trade execution.
            commission_min: Minimum transaction fee limit.
            tax_rate: Capital gains tax rate applied on realized returns.
            tax_deferred: If True, tax is compounded and deducted yearly; else paid on trade execution.
            annual_margin_rate: Annual borrowing interest rate for negative cash balances.
            annual_borrow_rate: Annual fee rate for carrying short positions.
            allow_margin: If True, margin debt is allowed up to the max_leverage constraint.
            max_leverage: Maximum allowed gross leverage (Gross Exposure / Net Equity).
            allow_short: If True, short-selling positions can be opened.
            allow_fractional: If True, fractional shares can be traded; else rounded to integers.
            slippage_pct: Slippage rate representing average execution bid-ask spread friction.
            execution_delay: Number of bars to delay the strategy signal (default 1).
            execution_price_type: Pricing type to execute trades at ('Open' or 'Close').
            commission_model: Optional custom CommissionModel policy implementation.
            slippage_model: Optional custom SlippageModel policy implementation.
            tax_model: Optional custom TaxModel policy implementation.
        """
        if not isinstance(data.columns, pd.MultiIndex):
            raise ValueError("Data columns must be a MultiIndex DataFrame with levels (Ticker, Price)")
            
        self.data = data.sort_index()
        self.strategy = strategy
        self.initial_capital = initial_capital
        self.monthly_deposit = monthly_deposit
        self.margin_rate_daily = annual_margin_rate / 360.0  # Actual/360 rate
        self.borrow_rate_daily = annual_borrow_rate / 360.0  # Actual/360 rate
        self.allow_margin = allow_margin
        self.max_leverage = max_leverage
        self.allow_short = allow_short
        self.allow_fractional = allow_fractional
        self.execution_delay = execution_delay
        self.execution_price_type = execution_price_type
        
        self.commission_model = commission_model or DefaultCommissionModel(
            per_share=commission_per_share, pct=commission_pct, flat=commission_flat, minimum=commission_min
        )
        self.slippage_model = slippage_model or DefaultSlippageModel(pct=slippage_pct)
        self.tax_model = tax_model or DefaultTaxModel(rate=tax_rate, deferred=tax_deferred)

    def run(self) -> pd.DataFrame:
        """Executes the backtest simulation over the entire historical dataset."""
        trades = self.strategy.generate_trades(self.data)
        
        # Ensure chronological index alignment and fill empty signals
        trades = trades.reindex(self.data.index, fill_value=0.0)

        # Apply execution signal delay
        if self.execution_delay > 0:
            trades = trades.shift(self.execution_delay).fillna(0.0)
            
        tickers = trades.columns.tolist()
        num_tickers = len(tickers)
        
        # Pre-extract matrices as 2D NumPy arrays for O(1) loop execution
        try:
            exec_prices = self.data.xs(self.execution_price_type, level=1, axis=1).reindex(columns=tickers).to_numpy()
            close_prices = self.data.xs('Close', level=1, axis=1).reindex(columns=tickers).to_numpy()
        except KeyError as e:
            raise KeyError(f"Pricing metric level error in input: {e}")

        state = PortfolioState(
            cash=float(self.initial_capital),
            holdings=np.zeros(num_tickers, dtype=float),
            cost_basis=np.zeros(num_tickers, dtype=float)
        )
        
        return self._simulate(trades.index, trades.to_numpy(), exec_prices, close_prices, tickers, state)

    # -----------------------------------------------------------------
    # Simulation Logic Loop
    # -----------------------------------------------------------------
    def _simulate(self, dates: pd.Index, trades_arr: np.ndarray, exec_prices_arr: np.ndarray,
                  close_prices_arr: np.ndarray, tickers: List[str], state: PortfolioState) -> pd.DataFrame:
        """Runs the day-by-day simulation loop tracking equity, cash flows, and fees."""
                  
        cash_hist = np.zeros(len(dates), dtype=float)
        val_hist = np.zeros(len(dates), dtype=float)
        deposit_hist = np.zeros(len(dates), dtype=float)
        
        prev_date = None
        for i, date in enumerate(dates):
            days_elapsed = (date - prev_date).days if prev_date is not None else 1
            
            # Apply calendar frictions (deposit, margin interest, tax deferral)
            deposit_made = self._apply_daily_frictions(date, state, days_elapsed)
            deposit_hist[i] = deposit_made
            
            # Execute transactions
            daily_val = self._process_row(i, date, trades_arr[i], state, exec_prices_arr, close_prices_arr, tickers, days_elapsed)
            
            cash_hist[i] = state.cash
            val_hist[i] = daily_val
            prev_date = date
            
        # Deduct final accumulated deferred tax at backtest end
        if isinstance(self.tax_model, DefaultTaxModel) and self.tax_model.deferred and state.accumulated_tax > 0.0:
            state.cash -= state.accumulated_tax
            state.accumulated_tax = 0.0
            if len(cash_hist) > 0:
                cash_hist[-1] = state.cash
                
        return self._finalize_results(cash_hist, val_hist, deposit_hist)

    def _apply_daily_frictions(self, date: pd.Timestamp, state: PortfolioState, days_elapsed: int) -> float:
        """Applies deposits, compounding margin interest, and year-end deferred tax deductions."""
        deposit_made = 0.0
        
        # Monthly deposit
        if state.prev_month is not None and date.month != state.prev_month:
            state.cash += self.monthly_deposit
            deposit_made = self.monthly_deposit
        state.prev_month = date.month

        # Margin interest compounding over elapsed days
        if state.cash < 0.0:
            state.cash *= (1.0 + self.margin_rate_daily) ** days_elapsed

        # Year-end tax deductions
        if isinstance(self.tax_model, DefaultTaxModel) and self.tax_model.deferred:
            if state.prev_year is not None and date.year != state.prev_year:
                state.cash -= state.accumulated_tax
                state.accumulated_tax = 0.0
            state.prev_year = date.year
            
        return deposit_made

    def _process_row(self, date_idx: int, date: pd.Timestamp, trade_row_arr: np.ndarray,
                     state: PortfolioState, exec_prices_arr: np.ndarray, close_prices_arr: np.ndarray,
                     tickers: List[str], days_elapsed: int) -> float:
        """Processes daily transactions (sells first, then buys) and returns portfolio valuation."""
                     
        # 1. Apply daily short borrow fees on all existing short positions
        for ticker_idx in range(len(tickers)):
            if state.holdings[ticker_idx] < 0.0:
                close_price = close_prices_arr[date_idx, ticker_idx]
                if not pd.isna(close_price) and close_price > 0.0:
                    self._apply_borrow_fee(ticker_idx, close_price, state, days_elapsed)

        # 2. Execution order: process sells (qty < 0) first to release cash/buying power
        execution_queue = sorted(
            zip(range(len(tickers)), trade_row_arr),
            key=lambda x: x[1]
        )
        
        for ticker_idx, qty in execution_queue:
            if qty == 0.0:
                continue
                
            price = exec_prices_arr[date_idx, ticker_idx]
            if pd.isna(price) or price <= 0.0:
                continue
                
            self._execute_trade(date_idx, ticker_idx, qty, price, state, close_prices_arr)
            
        # 3. Vectorized portfolio valuation using np.nansum to handle missing close prices
        return np.nansum(state.holdings * close_prices_arr[date_idx])

    def _apply_borrow_fee(self, ticker_idx: int, price: float, state: PortfolioState, days_elapsed: int):
        """Deducts the calendar short borrow fee based on position value and elapsed days."""
        short_val = abs(state.holdings[ticker_idx] * price)
        state.cash -= short_val * self.borrow_rate_daily * days_elapsed

    # -----------------------------------------------------------------
    # Order Execution & Accounting Detail
    # -----------------------------------------------------------------
    def _execute_trade(self, date_idx: int, ticker_idx: int, qty: float, price: float,
                       state: PortfolioState, close_prices_arr: np.ndarray):
        """Checks constraints, calculates costs/taxes, and executes a single asset transaction."""
                       
        price_eff = self.slippage_model.apply(price, qty)
        
        # Enforce short restrictions
        if qty < 0.0 and not self.allow_short:
            current_holding = state.holdings[ticker_idx]
            if current_holding <= 0.0:
                qty = 0.0
            else:
                qty = max(qty, -current_holding)
                
        if qty == 0.0:
            return
            
        # Enforce margin/cash limits for buys
        if qty > 0.0:
            qty = self._clamp_buy_quantity(date_idx, ticker_idx, qty, price_eff, state, close_prices_arr)
            
        # Enforce margin/cash limits for short entries
        elif qty < 0.0 and state.holdings[ticker_idx] <= 0.0:
            qty = self._clamp_short_quantity(date_idx, qty, price_eff, state, close_prices_arr)
            
        # Round to whole shares if fractional trading is disabled
        if not self.allow_fractional:
            qty = float(math.floor(qty)) if qty > 0.0 else float(math.ceil(qty))
            
        if qty == 0.0:
            return
            
        comm = self.commission_model.calculate(qty, price_eff)
        tax = self.tax_model.calculate_tax(qty, price_eff, state.cost_basis[ticker_idx], state.holdings[ticker_idx], comm)
        
        # Process tax deductions
        if isinstance(self.tax_model, DefaultTaxModel) and self.tax_model.deferred:
            state.accumulated_tax += tax
            tax_to_deduct = 0.0
        else:
            tax_to_deduct = tax
            
        # Adjust cash
        trade_cost = (qty * price_eff) + comm + tax_to_deduct
        state.cash -= trade_cost
        
        self._update_accounting(ticker_idx, qty, price_eff, comm, state)

    def _clamp_buy_quantity(self, date_idx: int, ticker_idx: int, qty: float, price_eff: float,
                            state: PortfolioState, close_prices_arr: np.ndarray) -> float:
        """Clamps requested purchase quantity based on available cash and margin limits."""
                            
        max_cash_qty = self.commission_model.get_max_qty(state.cash, price_eff)
        
        if self.allow_margin:
            # Leverage limits computed using gross holdings value
            gross_holdings_val = np.nansum(np.abs(state.holdings) * close_prices_arr[date_idx])
            net_holdings_val = np.nansum(state.holdings * close_prices_arr[date_idx])
            equity = state.cash + net_holdings_val
            
            # C_eff = MaxLeverage * Equity - GrossHoldings
            C_eff = self.max_leverage * equity - gross_holdings_val
            max_margin_qty = self.commission_model.get_max_qty(C_eff, price_eff)
            q_limit = max_margin_qty
        else:
            q_limit = max_cash_qty
            
        return min(qty, q_limit)

    def _clamp_short_quantity(self, date_idx: int, qty: float, price_eff: float,
                              state: PortfolioState, close_prices_arr: np.ndarray) -> float:
        """Clamps requested short sell quantity to prevent margin leverage limit violations."""
        
        net_holdings_val = np.nansum(state.holdings * close_prices_arr[date_idx])
        equity = state.cash + net_holdings_val
        
        if equity <= 0.0 or not self.allow_margin:
            return 0.0
            
        gross_holdings_val = np.nansum(np.abs(state.holdings) * close_prices_arr[date_idx])
        C_eff = self.max_leverage * equity - gross_holdings_val
        
        max_short_qty = self.commission_model.get_max_qty(C_eff, price_eff)
        return max(qty, -max_short_qty)  # qty and max_short_qty are negative/bounds

    def _update_accounting(self, ticker_idx: int, qty: float, price: float, comm: float, state: PortfolioState):
        """Updates portfolio holdings count and recalculates the average cost basis per share."""
        old_h = state.holdings[ticker_idx]
        new_h = old_h + qty
        
        is_flip = (old_h > 0.0 and new_h < 0.0) or (old_h < 0.0 and new_h > 0.0)
        is_closed = (new_h == 0.0)

        if is_flip:
            pro_rated_comm = comm * (abs(new_h) / abs(qty))
            if new_h > 0.0:
                state.cost_basis[ticker_idx] = price + (pro_rated_comm / abs(new_h))
            else:
                state.cost_basis[ticker_idx] = price - (pro_rated_comm / abs(new_h))
                
        elif (old_h >= 0.0 and qty > 0.0) or (old_h <= 0.0 and qty < 0.0):
            cur_total_cost = abs(old_h * state.cost_basis[ticker_idx])
            # For long positions, commission increases entry cost.
            # For short positions, commission reduces net short entry proceeds (price).
            comm_effect = comm if qty > 0.0 else -comm
            new_trade_cost = abs(qty * price) + comm_effect
            state.cost_basis[ticker_idx] = (cur_total_cost + new_trade_cost) / abs(new_h)
            
        elif is_closed:
            state.cost_basis[ticker_idx] = 0.0
            
        state.holdings[ticker_idx] = new_h

    def _finalize_results(self, cash_hist: np.ndarray, val_hist: np.ndarray, deposit_hist: np.ndarray) -> pd.DataFrame:
        """Formats the simulated historical timelines into a final output DataFrame with net returns."""
        res = pd.DataFrame(index=self.data.index)
        res['cash'] = cash_hist
        res['holdings_value'] = val_hist
        res['total_equity'] = res['cash'] + res['holdings_value']
        
        # Calculate daily returns net of capital deposits (deposits are inflows at the start of the day)
        # return_t = (Equity_t - Deposit_t - Equity_t-1) / Equity_t-1
        prev_equity = res['total_equity'].shift(1).fillna(self.initial_capital)
        res['returns'] = (res['total_equity'] - deposit_hist - prev_equity) / prev_equity
        res['returns'] = res['returns'].fillna(0.0)
        
        return res