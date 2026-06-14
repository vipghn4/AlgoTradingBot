import math
import pandas as pd
from data_structures import Strategy

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
                 execution_price_type: str = 'Open'):
        """
        A realistic backtester designed to model constraints faced by retail investors
        (e.g., full-time employees with small starting capital).

        Args:
            data: Historical market data (MultiIndex DataFrame).
            strategy: The Strategy instance to backtest.
            initial_capital: Starting cash balance.
            monthly_deposit: Periodic savings added to the cash balance monthly.
            commission_per_share: Brokerage fee per share traded.
            commission_pct: Brokerage fee as a percentage of trade value (e.g. 0.001 for 0.1%).
            commission_flat: Flat fee charged per trade transaction (e.g. $4.95).
            commission_min: Minimum brokerage fee charged per trade transaction.
            tax_rate: Capital gains tax rate (e.g. 0.15 for 15%).
            tax_deferred: If True, tax is accumulated and paid at the end of each calendar year.
                          If False, tax is paid immediately upon trade execution.
            annual_margin_rate: Annual interest rate paid on negative cash balances.
            annual_borrow_rate: Annual borrow fee rate paid on short stock positions.
            allow_margin: If False, trades are strictly limited by available cash.
                          If True, trades can exceed cash up to the max_leverage limit.
            max_leverage: Maximum allowed leverage ratio (Holdings Value / Net Equity) under margin.
            allow_short: If False, short selling is disabled; selling is capped at current holdings.
            allow_fractional: If False, trade quantities are rounded down to integer shares.
            slippage_pct: Slippage / bid-ask spread fraction (e.g., 0.0005 for 0.05% price penalty).
            execution_delay: Number of rows to delay trade signals (e.g., 1 day delay to represent
                             placing trades the morning after a signal is calculated).
            execution_price_type: Price metric to execute trades at (e.g., 'Open' or 'Close').
        """
        self.data = data
        self.strategy = strategy
        self.initial_capital = initial_capital
        self.monthly_deposit = monthly_deposit
        self.commission_per_share = commission_per_share
        self.commission_pct = commission_pct
        self.commission_flat = commission_flat
        self.commission_min = commission_min
        self.tax_rate = tax_rate
        self.tax_deferred = tax_deferred
        self.margin_rate_daily = annual_margin_rate / 365.25
        self.borrow_rate_daily = annual_borrow_rate / 365.25
        self.allow_margin = allow_margin
        self.max_leverage = max_leverage
        self.allow_short = allow_short
        self.allow_fractional = allow_fractional
        self.slippage_pct = slippage_pct
        self.execution_delay = execution_delay
        self.execution_price_type = execution_price_type
        
        self._current_date = None

    def run(self) -> pd.DataFrame:
        trades = self.strategy.generate_trades(self.data)
        
        # Apply execution signal delay (e.g., 1-day delay for next-day open orders)
        if self.execution_delay > 0:
            trades = trades.shift(self.execution_delay).fillna(0.0)
            
        state = self._init_state(trades.columns)
        return self._simulate(trades, state)

    def _init_state(self, tickers):
        return {
            'cash': float(self.initial_capital),
            'holdings': {t: 0.0 for t in tickers},
            'cost_basis': {t: 0.0 for t in tickers},
            'prev_month': None,
            'prev_year': None,
            'accumulated_tax': 0.0
        }

    def _simulate(self, trades, state):
        cash_hist, val_hist = [], []
        for date, trade_row in trades.iterrows():
            self._current_date = date
            self._apply_daily_frictions(date, state)
            daily_val = self._process_row(date, trade_row, state)
            cash_hist.append(state['cash'])
            val_hist.append(daily_val)
            
        # Deduct any remaining deferred tax at the end of the simulation
        if self.tax_deferred and state['accumulated_tax'] > 0:
            state['cash'] -= state['accumulated_tax']
            state['accumulated_tax'] = 0.0
            if cash_hist:
                cash_hist[-1] = state['cash']
                
        return self._finalize_results(cash_hist, val_hist)

    def _apply_daily_frictions(self, date, state):
        self._apply_deposit(date, state)
        self._apply_interest(state)
        self._apply_tax_deduction(date, state)

    def _apply_deposit(self, date, state):
        if state['prev_month'] is not None and date.month != state['prev_month']:
            state['cash'] += self.monthly_deposit
        state['prev_month'] = date.month

    def _apply_interest(self, state):
        if state['cash'] < 0:
            state['cash'] *= (1 + self.margin_rate_daily)

    def _apply_tax_deduction(self, date, state):
        if self.tax_deferred and state['prev_year'] is not None and date.year != state['prev_year']:
            # Deduct accumulated capital gains tax for the past year
            state['cash'] -= state['accumulated_tax']
            state['accumulated_tax'] = 0.0
        state['prev_year'] = date.year

    def _process_row(self, date, trade_row, state):
        row_val = 0.0
        for ticker, qty in trade_row.items():
            price = self.data.loc[date, (ticker, self.execution_price_type)]
            
            # Skip trading if price is missing or invalid
            if pd.isna(price) or price <= 0:
                close_price = self.data.loc[date, (ticker, 'Close')]
                if not pd.isna(close_price):
                    row_val += state['holdings'][ticker] * close_price
                continue
                
            self._apply_borrow_fee(ticker, price, state)
            
            if qty != 0:
                self._execute_trade(ticker, qty, price, state)
                
            close_price = self.data.loc[date, (ticker, 'Close')]
            if not pd.isna(close_price):
                row_val += state['holdings'][ticker] * close_price
                
        return row_val

    def _apply_borrow_fee(self, ticker, price, state):
        if state['holdings'][ticker] < 0:
            short_val = abs(state['holdings'][ticker] * price)
            state['cash'] -= short_val * self.borrow_rate_daily

    def _execute_trade(self, ticker, qty, price, state):
        # 1. Apply slippage
        price_eff = price * (1 + self.slippage_pct) if qty > 0 else price * (1 - self.slippage_pct)
        
        # 2. Check short selling restriction
        if qty < 0 and not self.allow_short:
            current_holding = state['holdings'][ticker]
            if current_holding <= 0:
                qty = 0.0
            else:
                qty = max(qty, -current_holding)
                
        if qty == 0:
            return
            
        # 3. Check cash / margin constraints for buys
        if qty > 0:
            f_flat = self.commission_flat
            f_pct = self.commission_pct
            f_share = self.commission_per_share
            f_min = self.commission_min
            
            q_cash_limit = self._get_max_qty(state['cash'], price_eff, f_flat, f_pct, f_share, f_min)
            
            if self.allow_margin:
                # E = cash + holdings_value
                holdings_val = sum(
                    state['holdings'][t] * self.data.loc[self._current_date, (t, 'Close')]
                    for t in state['holdings'] if not pd.isna(self.data.loc[self._current_date, (t, 'Close')])
                )
                equity = state['cash'] + holdings_val
                C_eff = self.max_leverage * equity - holdings_val
                q_margin_limit = self._get_max_qty(
                    C_eff, price_eff,
                    self.max_leverage * f_flat,
                    self.max_leverage * f_pct,
                    self.max_leverage * f_share,
                    self.max_leverage * f_min
                )
                q_limit = q_margin_limit
            else:
                q_limit = q_cash_limit
                
            qty = min(qty, q_limit)
            
        # 4. Enforce integer shares if fractional shares are not allowed
        if not self.allow_fractional:
            if qty > 0:
                qty = float(math.floor(qty))
            else:
                qty = float(math.ceil(qty))
                
        if qty == 0:
            return
            
        # 5. Calculate final commission & tax
        comm = self._calculate_commission(qty, price_eff)
        tax = self._get_tax(ticker, qty, price_eff, comm, state)
        
        # 6. Adjust cash
        if qty > 0:
            cost = (qty * price_eff) + comm + tax
            state['cash'] -= cost
        else:
            state['cash'] -= (qty * price_eff) + comm + tax
            
        # 7. Update accounting (holdings and cost basis)
        self._update_accounting(ticker, qty, price_eff, comm, state)

    def _get_max_qty(self, C, P, f_flat, f_pct, f_share, f_min) -> float:
        if C <= 0:
            return 0.0
            
        P_factor = P * (1 + f_pct) + f_share
        if P_factor > 0:
            q_candidate = (C - f_flat) / P_factor
            comm_candidate = q_candidate * (P * f_pct + f_share) + f_flat
            if comm_candidate >= f_min:
                q_max = q_candidate
            else:
                q_max = (C - f_min) / P if P > 0 else 0.0
        else:
            q_max = 0.0
            
        return max(0.0, q_max)

    def _calculate_commission(self, qty, price) -> float:
        if qty == 0:
            return 0.0
        comm = (abs(qty) * self.commission_per_share) + (abs(qty) * price * self.commission_pct) + self.commission_flat
        if comm < self.commission_min:
            comm = self.commission_min
        return comm

    def _get_tax(self, ticker, qty, price, comm, state) -> float:
        if qty < 0 and state['holdings'][ticker] > 0:
            realized_qty = min(abs(qty), state['holdings'][ticker])
            gain = realized_qty * (price - state['cost_basis'][ticker]) - comm
            return max(0, gain * self.tax_rate)
        return 0.0

    def _update_accounting(self, ticker, qty, price, comm, state):
        old_h = state['holdings'][ticker]
        new_h = old_h + qty
        
        is_flip = (old_h > 0 and new_h < 0) or (old_h < 0 and new_h > 0)
        is_closed = (new_h == 0)

        if is_flip:
            # Flipped: Start fresh basis for the new direction using the trade price + pro-rated comm
            state['cost_basis'][ticker] = price + (comm / abs(qty))
        elif (old_h >= 0 and qty > 0) or (old_h <= 0 and qty < 0):
            # Adding to existing position: Calculate new average cost including commissions
            cur_total_cost = abs(old_h * state['cost_basis'][ticker])
            new_trade_cost = abs(qty * price) + comm
            state['cost_basis'][ticker] = (cur_total_cost + new_trade_cost) / abs(new_h)
        elif is_closed:
            state['cost_basis'][ticker] = 0.0
            
        state['holdings'][ticker] = new_h

    def _finalize_results(self, cash_hist, val_hist):
        res = pd.DataFrame(index=self.data.index)
        res['cash'], res['holdings_value'] = cash_hist, val_hist
        res['total_equity'] = res['cash'] + res['holdings_value']
        res['returns'] = res['total_equity'].pct_change().fillna(0.0)
        return res