import numpy as np
import pandas as pd
from typing import Optional

class PortfolioState:
    """Domain model representing the portfolio's active financial state and transaction accounting logic."""

    def __init__(self, cash: float, num_tickers: int):
        """
        Initialize the PortfolioState with cash, holdings, and tax metrics.

        Args:
            cash: Initial cash balance in the account's base currency.
            num_tickers: Number of assets/tickers tracked in the portfolio.
        """
        self.cash = cash
        self.holdings = np.zeros(num_tickers, dtype=float)
        self.cost_basis = np.zeros(num_tickers, dtype=float)
        self.realized_gain_loss_ytd = 0.0
        self.tax_loss_carry_forward = 0.0
        self.prev_month: Optional[int] = None
        self.prev_year: Optional[int] = None
        self.prev_year_date: Optional[pd.Timestamp] = None
        self.lots = [[] for _ in range(num_tickers)]
        self.executed_trades_count = 0

    def get_holdings_value(self, close_prices: np.ndarray) -> float:
        """Return the mark-to-market value of all active holdings."""
        return float(np.nansum(self.holdings * close_prices))

    def get_equity(self, close_prices: np.ndarray, fx_rate: float = 1.0) -> float:
        """Return the total portfolio equity value in the account currency (cash plus converted holdings value)."""
        return self.cash + self.get_holdings_value(close_prices) / fx_rate

    def get_gross_exposure(self, close_prices: np.ndarray, fx_rate: float = 1.0) -> float:
        """Return the gross absolute holdings exposure value converted to the account currency."""
        return float(np.nansum(np.abs(self.holdings) * close_prices)) / fx_rate

    def apply_interest(self, margin_rate_daily: float, days_elapsed: int):
        """Charge compounding margin interest on any negative cash balances."""
        if self.cash < 0.0:
            self.cash *= (1.0 + margin_rate_daily) ** days_elapsed

    def apply_short_borrow_fee(self, ticker_idx: int, close_price: float, borrow_rate_daily: float, days_elapsed: int, fx_rate: float = 1.0):
        """Charge borrowing fees on active short positions."""
        short_value = abs(self.holdings[ticker_idx] * close_price)
        fee = short_value * ((1.0 + borrow_rate_daily) ** days_elapsed - 1.0)
        self.cash -= fee / fx_rate

    def update_position(self, ticker_idx: int, qty: float, price: float, comm: float,
                        date: Optional[pd.Timestamp] = None, rate: float = 1.0) -> list:
        """Update share quantity, cost basis, and tax lots after executing a transaction."""
        if qty == 0.0:
            return []
            
        self.executed_trades_count += 1
        
        if date is None:
            date = pd.Timestamp('2026-06-14')
            
        old_h = self.holdings[ticker_idx]
        new_h = old_h + qty
        realized_trades = []
        
        # Enforce tax lot matching
        if old_h == 0.0:
            # Opening a new position
            unit_comm = comm / abs(qty) if qty != 0.0 else 0.0
            lot_price = price + unit_comm if qty > 0.0 else price - unit_comm
            self.lots[ticker_idx] = [{'qty': qty, 'price': lot_price, 'date': date, 'rate': rate}]
            self.cost_basis[ticker_idx] = lot_price
        elif (old_h > 0.0 and qty > 0.0) or (old_h < 0.0 and qty < 0.0):
            # Increasing an existing position
            unit_comm = comm / abs(qty) if qty != 0.0 else 0.0
            lot_price = price + unit_comm if qty > 0.0 else price - unit_comm
            self.lots[ticker_idx].append({'qty': qty, 'price': lot_price, 'date': date, 'rate': rate})
            # Recalculate average cost basis of remaining position
            total_cost = sum(abs(lot['qty'] * lot['price']) for lot in self.lots[ticker_idx])
            self.cost_basis[ticker_idx] = total_cost / abs(new_h)
        else:
            # Decreasing or flipping a position
            qty_to_match = qty
            unit_comm = comm / abs(qty) if qty != 0.0 else 0.0
            
            while qty_to_match != 0.0 and len(self.lots[ticker_idx]) > 0:
                lot = self.lots[ticker_idx][0]
                # If we are long, lot['qty'] > 0 and qty_to_match < 0
                # If we are short, lot['qty'] < 0 and qty_to_match > 0
                match_qty = min(abs(qty_to_match), abs(lot['qty']))
                
                # Reconstruct match qty with correct sign
                match_qty_signed = -match_qty if old_h > 0.0 else match_qty
                
                # Calculate gain/loss
                holding_days = (date - lot['date']).days
                if old_h > 0.0:
                    # Closing a long position: sold at price (adjusted down by comm), bought at lot['price']
                    realized_trades.append((match_qty, lot['price'], lot.get('rate', 1.0), price - unit_comm, rate, holding_days))
                else:
                    # Closing a short position: covered at price (adjusted up by comm), sold at lot['price']
                    realized_trades.append((match_qty, price + unit_comm, rate, lot['price'], lot.get('rate', 1.0), holding_days))
                    
                # Update lot and remaining quantity to match
                lot['qty'] += match_qty_signed
                qty_to_match -= match_qty_signed
                
                if abs(lot['qty']) < 1e-9:
                    self.lots[ticker_idx].pop(0)
                    
            # If position flipped, open new lot with the remaining quantity
            if abs(qty_to_match) > 1e-9:
                lot_price = price + unit_comm if qty_to_match > 0.0 else price - unit_comm
                self.lots[ticker_idx] = [{'qty': qty_to_match, 'price': lot_price, 'date': date, 'rate': rate}]
                self.cost_basis[ticker_idx] = lot_price
            else:
                # Update average cost basis of remaining shares
                if abs(new_h) > 1e-9:
                    total_cost = sum(abs(lot['qty'] * lot['price']) for lot in self.lots[ticker_idx])
                    self.cost_basis[ticker_idx] = total_cost / abs(new_h)
                else:
                    self.cost_basis[ticker_idx] = 0.0
                    
        self.holdings[ticker_idx] = new_h
        return realized_trades
