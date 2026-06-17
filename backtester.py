import math
from dataclasses import dataclass
from typing import Optional, List
import numpy as np
import pandas as pd
from data_structures import Strategy
from portfolio_state import PortfolioState
from trading_models import (
    CommissionModel, DefaultCommissionModel,
    SlippageModel, DefaultSlippageModel,
    TaxModel, DefaultTaxModel, CurrencyConverter
)

# =====================================================================
# CONFIGURATION CONTAINER
# =====================================================================

@dataclass(frozen=True)
class BacktesterConfig:
    initial_capital: float
    monthly_deposit: float
    margin_rate_daily: float
    borrow_rate_daily: float
    allow_margin: bool
    max_leverage: float
    maintenance_margin_pct: float
    allow_short: bool
    allow_fractional: bool
    execution_delay: int
    execution_price_type: str
    deposit_fx_rate: float
    account_currency: str
    deposit_currency: str


@dataclass(frozen=True)
class SimulationContext:
    tickers: List[str]
    exec_prices: np.ndarray
    close_prices: np.ndarray
    low_prices: Optional[np.ndarray]
    high_prices: Optional[np.ndarray]


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
                 fx_pct: float = 0.0,
                 tax_rate: float = 0.0,
                 tax_deferred: bool = True,
                 annual_margin_rate: float = 0.08,
                 annual_borrow_rate: float = 0.02,
                 allow_margin: bool = False,
                 max_leverage: float = 2.0,
                 maintenance_margin_pct: float = 0.25,
                 allow_short: bool = False,
                 allow_fractional: bool = True,
                 slippage_pct: float = 0.0005,
                 execution_delay: int = 1,
                 execution_price_type: str = 'Open',
                 deposit_fx_rate: float = 1.0,
                 account_currency: str = 'USD',
                 deposit_currency: str = 'USD',
                 fx_rates: Optional[pd.Series] = None,
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
            fx_pct: FX conversion fee markup percentage (e.g. 0.0035 for 0.35%).
                - Reference: https://wise.com/in/blog/what-is-a-forex-markup-fee
            tax_rate: Capital gains tax rate applied on realized returns.
            tax_deferred: If True, tax is compounded and deducted yearly; else paid on trade execution.
            annual_margin_rate: Annual borrowing interest rate for negative cash balances.
            annual_borrow_rate: Annual fee rate for carrying short positions.
            allow_margin: If True, margin debt is allowed up to the max_leverage constraint.
            max_leverage: Maximum allowed gross leverage (Gross Exposure / Net Equity).
                - Reference: https://corporatefinanceinstitute.com/resources/accounting/leverage-ratios/
            maintenance_margin_pct: Minimum margin equity ratio below which liquidation occurs.
                - Reference: https://www.benzinga.com/money/what-margin-equity
            allow_short: If True, short-selling positions can be opened.
            allow_fractional: If True, fractional shares can be traded; else rounded to integers.
            slippage_pct: Slippage rate representing average execution bid-ask spread friction.
            execution_delay: Number of bars to delay the strategy signal (default 1).
            execution_price_type: Pricing type to execute trades at ('Open' or 'Close').
            deposit_fx_rate: Exchange rate to convert monthly deposits to portfolio currency.
            account_currency: Account base currency for cash and equity tracking.
            deposit_currency: Currency of the monthly deposit.
            fx_rates: Optional pandas Series containing daily exchange rates.
            commission_model: Optional custom CommissionModel policy implementation.
            slippage_model: Optional custom SlippageModel policy implementation.
            tax_model: Optional custom TaxModel policy implementation.
        """
        self._validate_inputs(data, max_leverage, maintenance_margin_pct, execution_delay, execution_price_type)
        self.data = data.sort_index()
        self.strategy = strategy
        self.fx_pct = fx_pct
        self.fx_rates = fx_rates if fx_rates is not None else pd.Series(deposit_fx_rate, index=data.index)
        self.currency_converter = CurrencyConverter(fx_rates=self.fx_rates, fx_pct=fx_pct)
        
        self.config = self._create_config(
            initial_capital, monthly_deposit, annual_margin_rate, annual_borrow_rate,
            allow_margin, max_leverage, maintenance_margin_pct, allow_short,
            allow_fractional, execution_delay, execution_price_type, deposit_fx_rate,
            account_currency, deposit_currency
        )
        self._init_models(
            commission_model, commission_per_share, commission_pct, commission_flat, commission_min,
            slippage_model, slippage_pct, tax_model, tax_rate, tax_deferred
        )

    def _validate_inputs(self, data: pd.DataFrame, max_leverage: float, maintenance_margin_pct: float,
                         execution_delay: int, execution_price_type: str):
        if not isinstance(data.columns, pd.MultiIndex):
            raise ValueError("Data columns must be a MultiIndex DataFrame with levels (Ticker, Price)")
        if max_leverage < 1.0:
            raise ValueError("max_leverage must be at least 1.0")
        if maintenance_margin_pct < 0.0 or maintenance_margin_pct >= 1.0:
            raise ValueError("maintenance_margin_pct must be between 0.0 and 1.0")
        if execution_delay == 0 and execution_price_type == 'Open':
            raise ValueError("execution_delay cannot be 0 when execution_price_type is 'Open' due to lookahead bias risk.")

    def _create_config(self, initial_capital: float, monthly_deposit: float, annual_margin_rate: float,
                       annual_borrow_rate: float, allow_margin: bool, max_leverage: float,
                       maintenance_margin_pct: float, allow_short: bool, allow_fractional: bool,
                       execution_delay: int, execution_price_type: str, deposit_fx_rate: float,
                       account_currency: str, deposit_currency: str) -> BacktesterConfig:
        return BacktesterConfig(
            initial_capital=initial_capital,
            monthly_deposit=monthly_deposit,
            margin_rate_daily=annual_margin_rate / 360.0,
            borrow_rate_daily=annual_borrow_rate / 360.0,
            allow_margin=allow_margin,
            max_leverage=max_leverage,
            maintenance_margin_pct=maintenance_margin_pct,
            allow_short=allow_short,
            allow_fractional=allow_fractional,
            execution_delay=execution_delay,
            execution_price_type=execution_price_type,
            deposit_fx_rate=deposit_fx_rate,
            account_currency=account_currency,
            deposit_currency=deposit_currency
        )

    def _init_models(self, commission_model: Optional[CommissionModel], commission_per_share: float,
                     commission_pct: float, commission_flat: float, commission_min: float,
                     slippage_model: Optional[SlippageModel], slippage_pct: float,
                     tax_model: Optional[TaxModel], tax_rate: float, tax_deferred: bool):
        if commission_model is None:
            self.commission_model = DefaultCommissionModel(
                per_share=commission_per_share, pct=commission_pct, flat=commission_flat, minimum=commission_min
            )
        else:
            self.commission_model = commission_model
        self.slippage_model = slippage_model or DefaultSlippageModel(pct=slippage_pct)
        self.tax_model = tax_model or DefaultTaxModel(rate=tax_rate, deferred=tax_deferred)

    def _calculate_commission(self, qty: float, price_eff: float) -> float:
        """Calculates transaction commission including FX fee if the account is in USD."""
        comm = self.commission_model.calculate(qty, price_eff)
        if self.config.account_currency == 'USD':
            comm += abs(qty) * price_eff * self.fx_pct
        return comm

    def _get_max_qty(self, cash: float, price: float) -> float:
        """Determines the maximum purchaseable quantity for a given cash limit, incorporating FX markup if USD account."""
        if not isinstance(self.commission_model, DefaultCommissionModel):
            return self.commission_model.get_max_qty(cash, price)
            
        if cash <= 0.0:
            return 0.0
            
        pct = self.commission_model.pct
        per_share = self.commission_model.per_share
        flat = self.commission_model.flat
        minimum = self.commission_model.minimum
        
        fx_pct = self.fx_pct if self.config.account_currency == 'USD' else 0.0
        
        price_factor = price * (1.0 + pct + fx_pct) + per_share
        if price_factor > 0.0:
            q_candidate = (cash - flat) / price_factor
            broker_comm_candidate = q_candidate * (price * pct + per_share) + flat
            if broker_comm_candidate >= minimum:
                return q_candidate
            else:
                denominator = price * (1.0 + fx_pct)
                return (cash - minimum) / denominator if denominator > 0.0 else 0.0
        return 0.0

    def run(self) -> pd.DataFrame:
        """Executes the backtest simulation over the entire historical dataset."""
        trades = self.strategy.generate_trades(self.data)
        assert all(trades.index == self.data.index), "Trade signals index must match data index."

        # Apply execution signal delay
        if self.config.execution_delay > 0:
            trades = trades.shift(self.config.execution_delay).fillna(0.0)
            
        tickers = trades.columns.tolist()
        num_tickers = len(tickers)
        
        # Pre-extract matrices and forward-fill to prevent NaNs in daily valuations/halts
        try:
            df_open = self.data.xs(self.config.execution_price_type, level=1, axis=1).reindex(columns=tickers).ffill().bfill()
            df_close = self.data.xs('Close', level=1, axis=1).reindex(columns=tickers).ffill().bfill()
            
            exec_prices = df_open.to_numpy()
            close_prices = df_close.to_numpy()
            
            # Extract Low/High for intraday margin checking if present
            try:
                df_low = self.data.xs('Low', level=1, axis=1).reindex(columns=tickers).ffill().bfill().to_numpy()
                df_high = self.data.xs('High', level=1, axis=1).reindex(columns=tickers).ffill().bfill().to_numpy()
            except KeyError:
                df_low = None
                df_high = None
        except KeyError as e:
            raise KeyError(f"Pricing metric level error in input: {e}")

        self.tax_model.reset()
        state = PortfolioState(
            cash=float(self.config.initial_capital),
            num_tickers=num_tickers
        )
        self.state = state
        
        ctx = SimulationContext(
            tickers=tickers,
            exec_prices=exec_prices,
            close_prices=close_prices,
            low_prices=df_low,
            high_prices=df_high
        )
        
        return self._simulate(trades.index, trades.to_numpy(), state, ctx)

    # -----------------------------------------------------------------
    # Simulation Logic Loop
    # -----------------------------------------------------------------
    def _simulate(self, dates: pd.Index, trades_arr: np.ndarray, state: PortfolioState, ctx: SimulationContext) -> pd.DataFrame:
        """Runs the day-by-day simulation loop tracking equity, cash flows, and fees."""
                  
        cash_hist = np.zeros(len(dates), dtype=float)
        val_hist = np.zeros(len(dates), dtype=float)
        deposit_hist = np.zeros(len(dates), dtype=float)
        
        prev_date = None
        for i, date in enumerate(dates):
            days_elapsed = (date - prev_date).days if prev_date is not None else 1
            rate = float(self.fx_rates.iloc[i])
            
            # Apply calendar frictions (deposit, margin interest, tax deferral)
            deposit_made = self._apply_daily_frictions(date, state, days_elapsed, rate)
            deposit_hist[i] = deposit_made
            
            # Reset liquidation flag for the new day
            state.is_liquidated = False
            
            if self.config.execution_price_type == 'Open':
                daily_val_usd = self._simulate_open_day(i, date, trades_arr[i], state, ctx, days_elapsed, rate)
            else:
                daily_val_usd = self._simulate_close_day(i, date, trades_arr[i], state, ctx, days_elapsed, rate)
                    
            if state.is_liquidated:
                daily_val_usd = 0.0
            
            cash_hist[i] = state.cash
            val_hist[i] = daily_val_usd / rate
            prev_date = date
            
        # Deduct final accumulated deferred tax at backtest end
        if self.tax_model.deferred:
            tax_due, new_carry = self.tax_model.calculate_annual_tax(state.realized_gain_loss_ytd, state.tax_loss_carry_forward)
            state.cash -= tax_due
            state.tax_loss_carry_forward = new_carry
            state.realized_gain_loss_ytd = 0.0
            if len(cash_hist) > 0:
                cash_hist[-1] = state.cash
                
        return self._finalize_results(cash_hist, val_hist, deposit_hist)

    def _simulate_open_day(self, idx: int, date: pd.Timestamp, trade_row: np.ndarray, state: PortfolioState,
                           ctx: SimulationContext, days_elapsed: int, rate: float) -> float:
        """Process Open-price trade execution and post-trade margin checking for a single day."""
        daily_val_usd = self._process_row(idx, date, trade_row, state, ctx.exec_prices, ctx.close_prices, ctx.tickers, days_elapsed, rate)
        self._check_maintenance_margin(idx, state, ctx.close_prices, ctx.low_prices, ctx.high_prices, ctx.tickers, rate)
        return daily_val_usd

    def _simulate_close_day(self, idx: int, date: pd.Timestamp, trade_row: np.ndarray, state: PortfolioState,
                            ctx: SimulationContext, days_elapsed: int, rate: float) -> float:
        """Process pre-trade margin check, Close-price trade execution, and Close margin verification for a single day."""
        self._check_maintenance_margin(idx, state, ctx.close_prices, ctx.low_prices, ctx.high_prices, ctx.tickers, rate)
        if not state.is_liquidated:
            daily_val_usd = self._process_row(idx, date, trade_row, state, ctx.exec_prices, ctx.close_prices, ctx.tickers, days_elapsed, rate)
            self._check_close_margin(idx, state, ctx.close_prices, ctx.tickers, rate)
            return daily_val_usd
        return 0.0

    def _apply_daily_frictions(self, date: pd.Timestamp, state: PortfolioState, days_elapsed: int, rate: float) -> float:
        """Applies deposits, compounding margin interest, and year-end deferred tax deductions."""
        deposit_made = 0.0
        
        # Monthly deposit with exchange rate conversion
        if state.prev_month is not None and date.month != state.prev_month:
            is_new_month = True
        elif state.prev_month is None:
            is_new_month = True
        else:
            is_new_month = False
            
        if is_new_month:
            converted_deposit = self.currency_converter.convert_deposit(
                self.config.monthly_deposit,
                self.config.deposit_currency,
                self.config.account_currency,
                rate
            )
            state.cash += converted_deposit
            deposit_made = converted_deposit
            
        state.prev_month = date.month

        # Margin interest compounding over elapsed days
        state.apply_interest(self.config.margin_rate_daily, days_elapsed)

        # Year-end tax deductions with offsetting & carry-forwards
        if self.tax_model.deferred:
            if state.prev_year_date is not None and self.tax_model.is_tax_year_end(date, state.prev_year_date):
                tax_due, new_carry = self.tax_model.calculate_annual_tax(state.realized_gain_loss_ytd, state.tax_loss_carry_forward)
                state.cash -= tax_due
                state.tax_loss_carry_forward = new_carry
                state.realized_gain_loss_ytd = 0.0
            state.prev_year_date = date
            
        return deposit_made

    def _process_row(self, date_idx: int, date: pd.Timestamp, trade_row_arr: np.ndarray,
                      state: PortfolioState, exec_prices_arr: np.ndarray, close_prices_arr: np.ndarray,
                      tickers: List[str], days_elapsed: int, rate: float) -> float:
        """Processes daily transactions (sells first, then buys) and returns portfolio valuation in USD."""
                      
        # Apply overnight borrowing fees on short positions using previous Close to prevent lookahead bias
        for ticker_idx in range(len(tickers)):
            if state.holdings[ticker_idx] < 0.0:
                price_idx = date_idx - 1 if date_idx > 0 else 0
                close_price = close_prices_arr[price_idx, ticker_idx]
                state.apply_short_borrow_fee(ticker_idx, close_price, self.config.borrow_rate_daily, days_elapsed, rate)

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
                
            self._execute_trade(date_idx, ticker_idx, qty, price, state, close_prices_arr, rate)
            
        # 3. Encapsulated portfolio valuation from PortfolioState (returns USD value)
        return state.get_holdings_value(close_prices_arr[date_idx])

    # -----------------------------------------------------------------
    # Order Execution & Accounting Detail
    # -----------------------------------------------------------------
    def _execute_trade(self, date_idx: int, ticker_idx: int, qty: float, price: float,
                       state: PortfolioState, close_prices_arr: np.ndarray, rate: float):
        """Checks constraints, calculates costs/taxes, and executes a single asset transaction."""
        old_h = state.holdings[ticker_idx]
        new_h = old_h + qty
        
        # Detect position flip (long to short or short to long)
        is_flip = (old_h > 0.0 and new_h < 0.0) or (old_h < 0.0 and new_h > 0.0)
        
        price_eff = self.slippage_model.apply(price, qty)
        
        if is_flip:
            qty_close = -old_h
            qty_open = new_h
            
            # Position flips are split so closing (risk-reducing) trades bypass margin clamps
            total_comm = self._calculate_commission(qty, price_eff)
            comm_close = total_comm * (abs(qty_close) / abs(qty))
            comm_open = total_comm * (abs(qty_open) / abs(qty))
            
            # Execute Closing Portion (no clamping required)
            self._process_transaction(date_idx, ticker_idx, qty_close, price_eff, comm_close, state, close_prices_arr, rate)
            
            # Execute Opening Portion (subject to margin clamps)
            self._process_transaction(date_idx, ticker_idx, qty_open, price_eff, comm_open, state, close_prices_arr, rate)
        else:
            comm = self._calculate_commission(qty, price_eff)
            self._process_transaction(date_idx, ticker_idx, qty, price_eff, comm, state, close_prices_arr, rate)

    def _process_transaction(self, date_idx: int, ticker_idx: int, qty: float, price_eff: float,
                             comm: float, state: PortfolioState, close_prices_arr: np.ndarray, rate: float):
        """Processes a single transaction line checking bounds and rounding constraints."""
        if qty == 0.0:
            return
            
        original_qty = qty
        val_idx = max(0, date_idx - 1) if self.config.execution_price_type == 'Open' else date_idx
        
        qty = self._apply_short_restrictions(ticker_idx, qty, state)
        if qty == 0.0:
            return
            
        qty = self._apply_margin_clamps(val_idx, ticker_idx, qty, price_eff, state, close_prices_arr, rate)
        if qty == 0.0:
            return
            
        is_liquidating = math.isclose(qty, -state.holdings[ticker_idx], rel_tol=1e-9, abs_tol=1e-9)
        qty = self._apply_fractional_rounding(qty, is_liquidating)
        if qty == 0.0:
            return
            
        if qty != original_qty:
            comm = self._calculate_commission(qty, price_eff)
            
        comm_tax = comm if is_liquidating or (state.holdings[ticker_idx] != 0.0 and (state.holdings[ticker_idx] + qty) == 0.0) else 0.0
        realized_lots = state.update_position(ticker_idx, qty, price_eff, comm, date=self.data.index[date_idx], rate=rate)
        trade_gain = self.tax_model.calculate_realized_gain(realized_lots, comm_tax, rate)
        
        state.realized_gain_loss_ytd += trade_gain
        tax = self._process_immediate_tax(state)
        
        self._deduct_transaction_cost(qty, price_eff, comm, tax, state, rate)

    def _apply_short_restrictions(self, ticker_idx: int, qty: float, state: PortfolioState) -> float:
        if qty < 0.0 and not self.config.allow_short:
            current_holding = state.holdings[ticker_idx]
            if current_holding <= 0.0:
                return 0.0
            return max(qty, -current_holding)
        return qty

    def _apply_margin_clamps(self, val_idx: int, ticker_idx: int, qty: float, price_eff: float,
                             state: PortfolioState, close_prices_arr: np.ndarray, rate: float) -> float:
        if qty > 0.0:
            if state.holdings[ticker_idx] >= 0.0:
                return self._clamp_buy_quantity(val_idx, ticker_idx, qty, price_eff, state, close_prices_arr, rate)
        elif qty < 0.0 and state.holdings[ticker_idx] <= 0.0:
            return self._clamp_short_quantity(val_idx, qty, price_eff, state, close_prices_arr, rate)
        return qty

    def _apply_fractional_rounding(self, qty: float, is_liquidating: bool) -> float:
        if not self.config.allow_fractional and not is_liquidating:
            return float(math.floor(qty)) if qty > 0.0 else float(math.ceil(qty))
        return qty

    def _process_immediate_tax(self, state: PortfolioState) -> float:
        if self.tax_model.deferred:
            return 0.0
        tax = 0.0
        if state.realized_gain_loss_ytd > 0.0:
            taxable_gain = max(0.0, state.realized_gain_loss_ytd - state.tax_loss_carry_forward)
            state.tax_loss_carry_forward = max(0.0, state.tax_loss_carry_forward - state.realized_gain_loss_ytd)
            tax = taxable_gain * self.tax_model.rate
            state.realized_gain_loss_ytd = 0.0
        elif state.realized_gain_loss_ytd < 0.0:
            state.tax_loss_carry_forward += abs(state.realized_gain_loss_ytd)
            state.realized_gain_loss_ytd = 0.0
        return tax

    def _deduct_transaction_cost(self, qty: float, price_eff: float, comm: float, tax: float,
                                 state: PortfolioState, rate: float):
        val_usd = qty * price_eff
        comm_usd = comm
        if self.config.account_currency == 'USD':
            val_account = val_usd
            comm_account = comm_usd
        else:
            is_buy = qty > 0.0
            val_account = self.currency_converter.convert_trade(val_usd, None, is_buy, rate)
            comm_account = comm_usd / rate
        trade_cost_account = val_account + comm_account + tax
        state.cash -= trade_cost_account

    def _clamp_buy_quantity(self, val_idx: int, ticker_idx: int, qty: float, price_eff: float,
                             state: PortfolioState, close_prices_arr: np.ndarray, rate: float) -> float:
        """Clamps requested purchase quantity based on available cash and margin limits."""
        # Convert state cash (in account currency) to USD
        cash_usd = state.cash * rate
        # Account for FX markup on transaction value by adjusting effective price
        price_eff_fx = price_eff / (1.0 - self.fx_pct)
        max_cash_qty = self._get_max_qty(cash_usd, price_eff_fx)
        
        if self.config.allow_margin:
            # Leverage limits computed in account currency
            gross_exposure = state.get_gross_exposure(close_prices_arr[val_idx], rate)
            net_holdings_val = state.get_holdings_value(close_prices_arr[val_idx]) / rate
            equity = state.cash + net_holdings_val
            
            # C_eff = MaxLeverage * Equity - GrossExposure
            C_eff = self.config.max_leverage * equity - gross_exposure
            C_eff_usd = C_eff * rate
            max_margin_qty = self._get_max_qty(C_eff_usd, price_eff_fx)
            q_limit = max_margin_qty
        else:
            q_limit = max_cash_qty
            
        return min(qty, q_limit)

    def _clamp_short_quantity(self, val_idx: int, qty: float, price_eff: float,
                               state: PortfolioState, close_prices_arr: np.ndarray, rate: float) -> float:
        """Clamps requested short sell quantity to prevent margin leverage limit violations."""
        net_holdings_val = state.get_holdings_value(close_prices_arr[val_idx]) / rate
        equity = state.cash + net_holdings_val
        
        if equity <= 0.0 or not self.config.allow_margin:
            return 0.0
            
        gross_exposure = state.get_gross_exposure(close_prices_arr[val_idx], rate)
        C_eff = self.config.max_leverage * equity - gross_exposure
        C_eff_usd = C_eff * rate
        
        # Account for FX markup on transaction value by adjusting effective price
        price_eff_fx = price_eff / (1.0 - self.fx_pct)
        max_short_qty = self._get_max_qty(C_eff_usd, price_eff_fx)
        return max(qty, -max_short_qty)

    # -----------------------------------------------------------------
    # Margin Risk Controls
    # -----------------------------------------------------------------
    def _check_maintenance_margin(self, date_idx: int, state: PortfolioState, close_prices_arr: np.ndarray,
                                  low_prices_arr: Optional[np.ndarray], high_prices_arr: Optional[np.ndarray],
                                  tickers: List[str], rate: float):
        """Verifies account equity is above the maintenance margin and triggers liquidation if breached."""
        if not self.config.allow_margin:
            if state.cash < -1e-5:
                self._liquidate_portfolio(date_idx, state, close_prices_arr, tickers, rate)
            return
            
        if low_prices_arr is not None and high_prices_arr is not None:
            self._check_intraday_margin(date_idx, state, close_prices_arr, low_prices_arr, high_prices_arr, tickers, rate)
        else:
            self._check_close_margin(date_idx, state, close_prices_arr, tickers, rate)

    def _check_intraday_margin(self, date_idx: int, state: PortfolioState, close_prices_arr: np.ndarray,
                               low_prices_arr: np.ndarray, high_prices_arr: np.ndarray, tickers: List[str], rate: float):
        """Checks for maintenance margin violations using intraday Low/High prices to prevent liquidation delay."""
        worst_prices = np.zeros(len(tickers))
        for j in range(len(tickers)):
            worst_prices[j] = low_prices_arr[date_idx, j] if state.holdings[j] >= 0.0 else high_prices_arr[date_idx, j]
        
        worst_holdings_val = float(np.nansum(state.holdings * worst_prices))
        worst_equity = state.cash + worst_holdings_val / rate
        worst_gross_exposure = float(np.nansum(np.abs(state.holdings) * worst_prices)) / rate
        
        if worst_gross_exposure > 0.0:
            margin_ratio = worst_equity / worst_gross_exposure
            if margin_ratio < self.config.maintenance_margin_pct or worst_equity <= 0.0:
                self._liquidate_portfolio(date_idx, state, close_prices_arr, tickers, rate)

    def _check_close_margin(self, date_idx: int, state: PortfolioState, close_prices_arr: np.ndarray,
                            tickers: List[str], rate: float):
        """Checks for maintenance margin violations using Close prices at end of day."""
        equity = state.get_equity(close_prices_arr[date_idx], rate)
        gross_exposure = state.get_gross_exposure(close_prices_arr[date_idx], rate)
        
        if gross_exposure > 0.0:
            margin_ratio = equity / gross_exposure
            if margin_ratio < self.config.maintenance_margin_pct or equity <= 0.0:
                self._liquidate_portfolio(date_idx, state, close_prices_arr, tickers, rate)

    def _liquidate_portfolio(self, date_idx: int, state: PortfolioState, close_prices_arr: np.ndarray, tickers: List[str], rate: float):
        """Forcibly liquidates all open positions at Close prices due to margin call."""
        state.is_liquidated = True
        date = self.data.index[date_idx]
        for ticker_idx in range(len(tickers)):
            qty = state.holdings[ticker_idx]
            if qty != 0.0:
                self._liquidate_ticker(date_idx, ticker_idx, qty, date, state, close_prices_arr, rate)

    def _liquidate_ticker(self, date_idx: int, ticker_idx: int, qty: float, date: pd.Timestamp,
                          state: PortfolioState, close_prices_arr: np.ndarray, rate: float):
        """Liquidates a single ticker position at Close price, updating cash and realized gains/losses."""
        price = close_prices_arr[date_idx, ticker_idx]
        if pd.isna(price) or price <= 0.0:
            return
            
        price_eff = self.slippage_model.apply(price, -qty)
        comm = self._calculate_commission(-qty, price_eff)
        
        proceeds = self._calculate_liquidation_proceeds(qty, qty * price_eff, comm, rate)
        state.cash += proceeds
        
        realized_lots = state.update_position(ticker_idx, -qty, price_eff, comm, date=date, rate=rate)
        trade_gain = self.tax_model.calculate_realized_gain(realized_lots, comm, rate)
        state.realized_gain_loss_ytd += trade_gain

    def _calculate_liquidation_proceeds(self, qty: float, val_usd: float, comm_usd: float, rate: float) -> float:
        """Calculates the net proceeds from liquidating a position, accounting for FX conversion and commission.

        Terms:
        - Proceeds: The net cash received from closing a position after deducting commissions and converting to account currency.

        Args:
        - qty: The quantity of the position being liquidated (positive for long, negative for short).
            - qty < 0 indicates a short position being closed (buying back shares).
            - qty > 0 indicates a long position being closed (selling shares).
        - rate: The FX conversion rate from USD to account currency (account_currency / USD).

        High-level process:
        1. Calculate the gross liquidation value in USD (val_usd).
        2. Subtract the commission in USD (comm_usd).
        3. If the account currency is not USD, convert the net proceeds to account currency using the provided FX rate.
        4. Return the net proceeds in account currency.
        
        Reference: https://legal-resources.uslegalforms.com/l/liquidation-proceeds"""
        if self.config.account_currency == 'USD':
            return val_usd - comm_usd
        is_buy = qty < 0.0
        val_account = self.currency_converter.convert_trade(val_usd, None, is_buy, rate)
        return val_account - comm_usd / rate

    def _finalize_results(self, cash_hist: np.ndarray, val_hist: np.ndarray, deposit_hist: np.ndarray) -> pd.DataFrame:
        """Formats the simulated historical timelines into a final output DataFrame with net returns."""
        res = pd.DataFrame(index=self.data.index)
        res['cash'] = cash_hist
        res['holdings_value'] = val_hist
        res['total_equity'] = res['cash'] + res['holdings_value']
        
        # Calculate daily returns net of capital deposits (deposits are inflows at the start of the day)
        # return_t = (Equity_t - Deposit_t - Equity_t-1) / Equity_t-1
        prev_equity = res['total_equity'].shift(1).fillna(self.config.initial_capital)
        res['returns'] = (res['total_equity'] - deposit_hist - prev_equity) / prev_equity
        res['returns'] = res['returns'].fillna(0.0)
        
        return res