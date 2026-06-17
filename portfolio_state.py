from dataclasses import dataclass
import numpy as np
import pandas as pd
from typing import Optional

@dataclass
class TaxLot:
    """Data structure representing an individual buy or sell transaction lot for tax calculations.
    
    Args:
    - qty: Quantity of shares in the lot (positive for buys, negative for sells).
    - price: Price per share for the lot, including transaction costs.
    - date: Timestamp of the transaction for holding period calculations.
    - rate: FX conversion rate to account currency at the time of the transaction (default is 1.0 for same-currency trades).
    """
    qty: float
    price: float
    date: pd.Timestamp
    rate: float = 1.0


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
        self.is_liquidated = False

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
                        date: pd.Timestamp, rate: float = 1.0) -> list:
        """Update share quantity, cost basis, and tax lots after executing a transaction.

        Assertion: The transaction date must be provided for tax lot accounting.

        Terms:
        - Realized trades: Trades that close or reduce a position, generating realized gains or losses.
        
        High-level logic:
        1. If the transaction quantity is zero, return an empty list (no action).
        2. Increment the executed trades count.
        3. Calculate the new holdings quantity and the adjusted lot price including per-unit commission.
        4. Depending on the current holdings and transaction direction:
            a. If there are no existing holdings, open a new position and create a new tax lot.
            b. If the transaction is in the same direction as existing holdings, increase the position and append a new tax lot.
            c. If the transaction is in the opposite direction, reduce or flip the position, performing FIFO tax lot matching and calculating realized gains/losses.
        5. Update the holdings and return a list of realized trades (if any) for further processing (e.g., tax calculations).
        """
        if qty == 0.0:
            return []
            
        self.executed_trades_count += 1
        assert date is not None, "Transaction date must be provided for tax lot accounting."
        
        old_h = self.holdings[ticker_idx]
        new_h = old_h + qty
        unit_comm = comm / abs(qty)
        lot_price = price + unit_comm if qty > 0.0 else price - unit_comm
        
        if old_h == 0.0:
            self._open_position(ticker_idx, qty, lot_price, date, rate)
            realized_trades = []
        elif (old_h > 0.0 and qty > 0.0) or (old_h < 0.0 and qty < 0.0):
            self._increase_position(ticker_idx, qty, lot_price, date, rate, new_h)
            realized_trades = []
        else:
            realized_trades = self._reduce_or_flip_position(
                ticker_idx, qty, price, unit_comm, date, rate, old_h, new_h
            )
            
        self.holdings[ticker_idx] = new_h
        return realized_trades

    def _open_position(self, ticker_idx: int, qty: float, lot_price: float, date: pd.Timestamp, rate: float):
        """Record a new initial transaction lot and set the starting cost basis."""
        self.lots[ticker_idx] = [TaxLot(qty=qty, price=lot_price, date=date, rate=rate)]
        self.cost_basis[ticker_idx] = lot_price

    def _increase_position(self, ticker_idx: int, qty: float, lot_price: float, date: pd.Timestamp, rate: float, new_h: float):
        """Append a new transaction lot and recalculate the average cost basis of the position."""
        self.lots[ticker_idx].append(TaxLot(qty=qty, price=lot_price, date=date, rate=rate))
        self.cost_basis[ticker_idx] = self._calculate_average_cost(ticker_idx, new_h)

    def _calculate_average_cost(self, ticker_idx: int, holdings_qty: float) -> float:
        """Calculate the weighted average cost basis of the remaining lots for a ticker."""
        if abs(holdings_qty) < 1e-9:
            return 0.0
        total_cost = sum(abs(lot.qty * lot.price) for lot in self.lots[ticker_idx])
        return total_cost / abs(holdings_qty)

    def _reduce_or_flip_position(self, ticker_idx: int, qty: float, price: float, unit_comm: float,
                                 date: pd.Timestamp, rate: float, old_h: float, new_h: float) -> list:
        """Reduce a position or flip it between long and short, performing FIFO tax lot matching.
        
        High-level logic:
        1. While there is remaining quantity to match and there are existing lots, match the transaction against the oldest lot.
        2. For each matched lot, calculate the realized trade details and update the lot's quantity.
        3. If the lot is fully matched, remove it from the list of lots.
        4. If there is remaining unmatched quantity after all lots are exhausted, create a new lot for the remaining quantity and set the cost basis to the transaction price.
        5. If all lots are matched, recalculate the average cost basis of the remaining position.
        6. Return the list of realized trades generated from the matching process.
        """
        realized_trades = []
        qty_to_match = qty
        
        while qty_to_match != 0.0 and len(self.lots[ticker_idx]) > 0:
            lot = self.lots[ticker_idx][0]
            match_qty, match_qty_signed, trade = self._match_lot(
                lot, qty_to_match, old_h, price, unit_comm, date, rate
            )
            realized_trades.append(trade)
            
            lot.qty += match_qty_signed
            qty_to_match -= match_qty_signed
            
            if abs(lot.qty) < 1e-9:
                self.lots[ticker_idx].pop(0)
                
        if abs(qty_to_match) > 1e-9:
            lot_price = price + unit_comm if qty_to_match > 0.0 else price - unit_comm
            self.lots[ticker_idx] = [TaxLot(qty=qty_to_match, price=lot_price, date=date, rate=rate)]
            self.cost_basis[ticker_idx] = lot_price
        else:
            self.cost_basis[ticker_idx] = self._calculate_average_cost(ticker_idx, new_h)
            
        return realized_trades

    def _match_lot(self, lot: TaxLot, qty_to_match: float, old_h: float, price: float, unit_comm: float,
                   date: pd.Timestamp, rate: float) -> tuple[float, float, tuple]:
        """Match a transaction against a specific tax lot to calculate realized trade parameters.
        
        High-level logic:
        1. Determine the quantity to match based on the smaller of the lot's quantity and the remaining quantity to match.
        2. Calculate the signed match quantity based on whether the position is long or short.
        3. Calculate the holding period in days for the matched lot.
        4. Depending on whether the position is long or short, calculate the realized trade details including matched quantity, prices, rates, and holding days.
        5. Return the matched quantity, signed matched quantity, and a tuple representing the realized trade details.

        Trade tuple structure:
        - For closing a long position: (matched_qty, lot_price, lot_rate, sell_price, sell_rate, holding_days)
        - For closing a short position: (matched_qty, cover_price, cover_rate, lot_price, lot_rate, holding_days)
        """
        match_qty = min(abs(qty_to_match), abs(lot.qty))
        match_qty_signed = -match_qty if old_h > 0.0 else match_qty
        holding_days = (date - lot.date).days
        
        if old_h > 0.0:
            # Closing a long position: sold at price (adjusted down by comm), bought at lot.price
            trade = (match_qty, lot.price, lot.rate, price - unit_comm, rate, holding_days)
        else:
            # Closing a short position: covered at price (adjusted up by comm), sold at lot.price
            trade = (match_qty, price + unit_comm, rate, lot.price, lot.rate, holding_days)
            
        return match_qty, match_qty_signed, trade
