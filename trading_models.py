from abc import ABC, abstractmethod

class CommissionModel(ABC):
    @abstractmethod
    def calculate(self, qty: float, price: float) -> float:
        """Calculate the total transaction commission."""
        pass

    @abstractmethod
    def get_max_qty(self, cash: float, price: float) -> float:
        """Calculate the maximum quantity purchaseable given a cash limit."""
        pass


class DefaultCommissionModel(CommissionModel):
    def __init__(self, per_share: float = 0.0, pct: float = 0.0, flat: float = 0.0, minimum: float = 0.0):
        self.per_share = per_share
        self.pct = pct
        self.flat = flat
        self.minimum = minimum

    def calculate(self, qty: float, price: float) -> float:
        if qty == 0.0:
            return 0.0
        comm = (abs(qty) * self.per_share) + (abs(qty) * price * self.pct) + self.flat
        return max(comm, self.minimum)

    def get_max_qty(self, cash: float, price: float) -> float:
        if cash <= 0.0:
            return 0.0
        
        # Cost equation: q * P * (1 + pct) + q * per_share + flat = cash
        price_factor = price * (1.0 + self.pct) + self.per_share
        if price_factor > 0.0:
            q_candidate = (cash - self.flat) / price_factor
            comm_candidate = q_candidate * (price * self.pct + self.per_share) + self.flat
            if comm_candidate >= self.minimum:
                q_max = q_candidate
            else:
                q_max = (cash - self.minimum) / price if price > 0.0 else 0.0
        else:
            q_max = 0.0
            
        return max(0.0, q_max)


class SlippageModel(ABC):
    @abstractmethod
    def apply(self, price: float, qty: float) -> float:
        """Apply transaction execution slippage to a price."""
        pass


class DefaultSlippageModel(SlippageModel):
    def __init__(self, pct: float = 0.0005):
        self.pct = pct

    def apply(self, price: float, qty: float) -> float:
        if qty > 0.0:
            return price * (1.0 + self.pct)
        elif qty < 0.0:
            return price * (1.0 - self.pct)
        return price


class TaxModel(ABC):
    @abstractmethod
    def calculate_tax(self, qty: float, price: float, cost_basis: float, holdings: float, comm: float) -> float:
        """Calculate the capital gains tax for a trade."""
        pass


class DefaultTaxModel(TaxModel):
    def __init__(self, rate: float = 0.0, deferred: bool = True):
        self.rate = rate
        self.deferred = deferred

    def calculate_tax(self, qty: float, price: float, cost_basis: float, holdings: float, comm: float) -> float:
        # Long position liquidation/reduction
        if qty < 0.0 and holdings > 0.0:
            realized_qty = min(abs(qty), holdings)
            gain = realized_qty * (price - cost_basis) - comm
            return max(0.0, gain * self.rate)
            
        # Short position coverage/reduction
        elif qty > 0.0 and holdings < 0.0:
            realized_qty = min(qty, abs(holdings))
            gain = realized_qty * (cost_basis - price) - comm
            return max(0.0, gain * self.rate)
            
        return 0.0
