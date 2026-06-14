from abc import ABC, abstractmethod
import pandas as pd
from typing import Optional

class CurrencyConverter:
    """Handles currency conversions and applies FX markups exactly once."""

    def __init__(self, fx_rates: pd.Series, fx_pct: float = 0.0):
        self.fx_rates = fx_rates
        self.fx_pct = fx_pct

    def convert_trade(self, amount_usd: float, date: pd.Timestamp, is_buy: bool, rate: float) -> float:
        """Convert trade transaction value from USD to account currency, applying FX fee."""
        if self.fx_pct == 0.0:
            return amount_usd / rate
        if is_buy:
            return amount_usd / (rate * (1.0 - self.fx_pct))
        else:
            return amount_usd / (rate * (1.0 + self.fx_pct))

    def convert_deposit(self, amount: float, deposit_curr: str, account_curr: str, rate: float) -> float:
        """Convert deposit from deposit currency to account base currency, applying FX fee."""
        if deposit_curr == account_curr:
            return amount
        if deposit_curr == 'GBP' and account_curr == 'USD':
            return amount * rate * (1.0 - self.fx_pct)
        else:
            return (amount / rate) * (1.0 - self.fx_pct)


class CommissionModel(ABC):
    """Abstract base class representing transaction commission models."""

    @abstractmethod
    def calculate(self, qty: float, price: float) -> float:
        """Calculate the total transaction commission."""
        pass

    @abstractmethod
    def get_max_qty(self, cash: float, price: float) -> float:
        """Calculate the maximum quantity purchaseable given a cash limit."""
        pass


class DefaultCommissionModel(CommissionModel):
    """Default commission model separating per-share, percentage, flat, and minimum commissions."""

    def __init__(self, per_share: float = 0.0, pct: float = 0.0, flat: float = 0.0, minimum: float = 0.0):
        """
        Initialize the commission model with broker commission settings.

        Args:
            per_share: Fixed fee charged per individual share traded.
            pct: Percentage-based fee of the total transaction value.
            flat: Flat fixed fee charged per execution.
            minimum: Minimum brokerage commission threshold.
        """
        self.per_share = per_share
        self.pct = pct
        self.flat = flat
        self.minimum = minimum

    def calculate(self, qty: float, price: float) -> float:
        """Calculate commission for a specific trade."""
        if qty == 0.0:
            return 0.0
        broker_comm = (abs(qty) * self.per_share) + (abs(qty) * price * self.pct) + self.flat
        return max(broker_comm, self.minimum)

    def get_max_qty(self, cash: float, price: float) -> float:
        """Determine the maximum purchaseable quantity for a given cash limit."""
        if cash <= 0.0:
            return 0.0
        
        # Cost equation: q * P * (1 + pct) + q * per_share + flat = cash
        price_factor = price * (1.0 + self.pct) + self.per_share
        if price_factor > 0.0:
            q_candidate = (cash - self.flat) / price_factor
            broker_comm_candidate = q_candidate * (price * self.pct + self.per_share) + self.flat
            if broker_comm_candidate >= self.minimum:
                q_max = q_candidate
            else:
                # Minimum commission is active: cost = q * P + minimum = cash
                q_max = (cash - self.minimum) / price if price > 0.0 else 0.0
        else:
            q_max = 0.0
            
        return max(0.0, q_max)


class SlippageModel(ABC):
    """Abstract base class representing transaction price slippage models."""

    @abstractmethod
    def apply(self, price: float, qty: float) -> float:
        """Apply transaction execution slippage to a price."""
        pass


class DefaultSlippageModel(SlippageModel):
    """Default slippage model applying a constant percentage bid-ask execution spread."""

    def __init__(self, pct: float = 0.0005):
        """
        Initialize the slippage model with a fixed percentage spread.

        Args:
            pct: The bid-ask execution spread markup/markdown percentage.
        """
        self.pct = pct

    def apply(self, price: float, qty: float) -> float:
        """Apply percentage-based execution slippage to the stock price."""
        if qty > 0.0:
            return price * (1.0 + self.pct)
        elif qty < 0.0:
            return price * (1.0 - self.pct)
        return price


class TaxModel(ABC):
    """Abstract base class representing tax models for realized capital gains and losses."""

    @abstractmethod
    def calculate_tax(self, qty: float, price: float, cost_basis: float, holdings: float, comm: float) -> float:
        """Calculate the capital gains tax for a trade (legacy/compatibility)."""
        pass

    @abstractmethod
    def calculate_realized_gain(self, realized_lots: list, comm: float, rate: float = 1.0) -> float:
        """Calculate the net realized capital gains from matched tax lots."""
        pass

    @abstractmethod
    def calculate_annual_tax(self, realized_gain_ytd: float, tax_loss_carry_forward: float) -> tuple[float, float]:
        """
        Calculate annual tax due and update the tax loss carry forward.
        Returns (tax_due, new_tax_loss_carry_forward).
        """
        pass

    @abstractmethod
    def is_tax_year_end(self, date: pd.Timestamp, prev_date: Optional[pd.Timestamp]) -> bool:
        """Check if the current date represents a tax year-end transition."""
        pass

    def reset(self):
        """Reset stateful trackers if any."""
        pass


class DefaultTaxModel(TaxModel):
    """Default tax model calculating capital gains and losses on long and short liquidations."""

    def __init__(self, rate: float = 0.0, deferred: bool = True):
        """
        Initialize the capital gains tax model.

        Args:
            rate: The capital gains tax percentage rate.
            deferred: True if tax is deferred and settled at year-end, False otherwise.
        """
        self.rate = rate
        self.deferred = deferred

    def calculate_tax(self, qty: float, price: float, cost_basis: float, holdings: float, comm: float) -> float:
        """Calculate the gains or losses realized on a long or short trade (legacy)."""
        if qty < 0.0 and holdings > 0.0:
            realized_qty = min(abs(qty), holdings)
            return realized_qty * (price - cost_basis) - comm
        elif qty > 0.0 and holdings < 0.0:
            realized_qty = min(qty, abs(holdings))
            return realized_qty * (cost_basis - price) - comm
        return 0.0

    def calculate_realized_gain(self, realized_lots: list, comm: float, rate: float = 1.0) -> float:
        """Calculate the net realized capital gains from matched tax lots converted to the account currency."""
        # Convert USD gain to account currency using the disposal rate
        gain_usd = sum(qty * (sell_p - buy_p) for qty, buy_p, buy_rate, sell_p, sell_rate, days in realized_lots)
        return gain_usd / rate

    def calculate_annual_tax(self, realized_gain_ytd: float, tax_loss_carry_forward: float) -> tuple[float, float]:
        """Standard calendar year capital gains tax calculation."""
        net_gain = realized_gain_ytd - tax_loss_carry_forward
        if net_gain > 0.0:
            tax_due = net_gain * self.rate
            new_carry_forward = 0.0
        else:
            tax_due = 0.0
            new_carry_forward = abs(net_gain)
        return tax_due, new_carry_forward

    def is_tax_year_end(self, date: pd.Timestamp, prev_date: Optional[pd.Timestamp]) -> bool:
        """Triggers at calendar year end."""
        if prev_date is None:
            return False
        return date.year != prev_date.year


class UKTaxModel(TaxModel):
    """UK HMRC-compliant tax model including annual exempt allowance and April 6th tax year alignment."""

    def __init__(self, rate: float = 0.20, annual_allowance: float = 3000.0, deferred: bool = True):
        self.rate = rate
        self.annual_allowance = annual_allowance
        self.deferred = deferred

    def calculate_tax(self, qty: float, price: float, cost_basis: float, holdings: float, comm: float) -> float:
        """Fallback legacy method."""
        if qty < 0.0 and holdings > 0.0:
            realized_qty = min(abs(qty), holdings)
            return realized_qty * (price - cost_basis) - comm
        elif qty > 0.0 and holdings < 0.0:
            realized_qty = min(qty, abs(holdings))
            return realized_qty * (cost_basis - price) - comm
        return 0.0

    def calculate_realized_gain(self, realized_lots: list, comm: float, rate: float = 1.0) -> float:
        # UK HMRC rules require converting purchase price using purchase-date rate and sale price using sale-date rate
        return sum(qty * (sell_p / sell_rate - buy_p / buy_rate) for qty, buy_p, buy_rate, sell_p, sell_rate, days in realized_lots)

    def calculate_annual_tax(self, realized_gain_ytd: float, tax_loss_carry_forward: float) -> tuple[float, float]:
        # Carry-forward losses are only used to reduce net gain to the allowance threshold, preserving remaining losses
        if realized_gain_ytd < 0.0:
            tax_due = 0.0
            new_carry_forward = tax_loss_carry_forward + abs(realized_gain_ytd)
        elif realized_gain_ytd <= self.annual_allowance:
            tax_due = 0.0
            new_carry_forward = tax_loss_carry_forward
        else:
            excess_gain = realized_gain_ytd - self.annual_allowance
            used_loss = min(excess_gain, tax_loss_carry_forward)
            taxable_gain = excess_gain - used_loss
            tax_due = taxable_gain * self.rate
            new_carry_forward = tax_loss_carry_forward - used_loss
        return tax_due, new_carry_forward

    def _get_tax_year(self, date: pd.Timestamp) -> int:
        return date.year if (date.month > 4 or (date.month == 4 and date.day >= 6)) else date.year - 1

    def is_tax_year_end(self, date: pd.Timestamp, prev_date: Optional[pd.Timestamp]) -> bool:
        if prev_date is None:
            return False
        return self._get_tax_year(date) != self._get_tax_year(prev_date)


class USTaxModel(TaxModel):
    """US IRS-compliant tax model separating short-term and long-term capital gains."""

    def __init__(self, short_term_rate: float = 0.24, long_term_rate: float = 0.15, deferred: bool = True):
        self.short_term_rate = short_term_rate
        self.long_term_rate = long_term_rate
        self.deferred = deferred
        self.st_gains_ytd = 0.0
        self.lt_gains_ytd = 0.0

    def calculate_tax(self, qty: float, price: float, cost_basis: float, holdings: float, comm: float) -> float:
        if qty < 0.0 and holdings > 0.0:
            realized_qty = min(abs(qty), holdings)
            return realized_qty * (price - cost_basis) - comm
        elif qty > 0.0 and holdings < 0.0:
            realized_qty = min(qty, abs(holdings))
            return realized_qty * (cost_basis - price) - comm
        return 0.0

    def calculate_realized_gain(self, realized_lots: list, comm: float, rate: float = 1.0) -> float:
        net_gain_account = 0.0
        for qty, buy_p, buy_rate, sell_p, sell_rate, days in realized_lots:
            # US IRS cost basis and proceeds are converted using acquisition-date rate and disposal-date rate
            buy_cost_account = qty * buy_p / buy_rate
            sell_proceeds_account = qty * sell_p / sell_rate
            gain_account = sell_proceeds_account - buy_cost_account
            net_gain_account += gain_account
            
            if days > 365:
                self.lt_gains_ytd += gain_account
            else:
                self.st_gains_ytd += gain_account
        return net_gain_account

    def calculate_annual_tax(self, realized_gain_ytd: float, tax_loss_carry_forward: float) -> tuple[float, float]:
        st = self.st_gains_ytd
        lt = self.lt_gains_ytd
        
        # Offset with loss carry forwards
        if tax_loss_carry_forward > 0.0:
            if st > 0.0:
                offset = min(st, tax_loss_carry_forward)
                st -= offset
                tax_loss_carry_forward -= offset
            if lt > 0.0 and tax_loss_carry_forward > 0.0:
                offset = min(lt, tax_loss_carry_forward)
                lt -= offset
                tax_loss_carry_forward -= offset
                
        net_ytd = st + lt
        if net_ytd < 0.0:
            new_carry_forward = tax_loss_carry_forward + abs(net_ytd)
            tax_due = 0.0
        else:
            tax_due = max(0.0, st) * self.short_term_rate + max(0.0, lt) * self.long_term_rate
            new_carry_forward = tax_loss_carry_forward
            
        self.st_gains_ytd = 0.0
        self.lt_gains_ytd = 0.0
        return tax_due, new_carry_forward

    def is_tax_year_end(self, date: pd.Timestamp, prev_date: Optional[pd.Timestamp]) -> bool:
        if prev_date is None:
            return False
        return date.year != prev_date.year

    def reset(self):
        self.st_gains_ytd = 0.0
        self.lt_gains_ytd = 0.0


class TaxFreeModel(TaxModel):
    """Tax-exempt model representing UK ISA or US Roth IRA accounts."""

    def __init__(self, deferred: bool = True):
        self.deferred = deferred

    def calculate_tax(self, qty: float, price: float, cost_basis: float, holdings: float, comm: float) -> float:
        return 0.0

    def calculate_realized_gain(self, realized_lots: list, comm: float, rate: float = 1.0) -> float:
        return 0.0

    def calculate_annual_tax(self, realized_gain_ytd: float, tax_loss_carry_forward: float) -> tuple[float, float]:
        return 0.0, 0.0

    def is_tax_year_end(self, date: pd.Timestamp, prev_date: Optional[pd.Timestamp]) -> bool:
        return False
