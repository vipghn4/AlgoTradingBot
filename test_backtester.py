import unittest
import numpy as np
import pandas as pd
from backtester import DefaultCommissionModel, DefaultSlippageModel, DefaultTaxModel, PortfolioState

class TestCommissionModel(unittest.TestCase):
    """Unit tests for the Commission Model using Arrange-Act-Assert (AAA)."""

    def test_commission_calculation_with_flat_and_percentage_fees(self):
        # Arrange
        model = DefaultCommissionModel(per_share=0.0, pct=0.01, flat=2.0, minimum=0.0)
        qty = 10.0
        price = 100.0
        expected_commission = 12.0  # 10 * 100 * 0.01 + 2.0

        # Act
        actual_commission = model.calculate(qty, price)

        # Assert
        self.assertEqual(actual_commission, expected_commission)

    def test_commission_enforces_minimum_fee(self):
        # Arrange
        model = DefaultCommissionModel(per_share=0.0, pct=0.0, flat=0.0, minimum=5.0)
        qty = 1.0
        price = 10.0
        expected_commission = 5.0  # Under minimum, should clamp to 5.0

        # Act
        actual_commission = model.calculate(qty, price)

        # Assert
        self.assertEqual(actual_commission, expected_commission)

    def test_get_max_qty_calculates_correct_capacity(self):
        # Arrange
        model = DefaultCommissionModel(per_share=0.0, pct=0.05, flat=5.0, minimum=0.0)
        cash = 110.0
        price = 10.0
        # Cost eq: q * 10 * 1.05 + 5.0 <= 110 => q * 10.5 <= 105 => q <= 10.0
        expected_qty = 10.0

        # Act
        actual_qty = model.get_max_qty(cash, price)

        # Assert
        self.assertAlmostEqual(actual_qty, expected_qty)


class TestSlippageModel(unittest.TestCase):
    """Unit tests for the Slippage Model using Arrange-Act-Assert (AAA)."""

    def test_slippage_increases_buy_execution_price(self):
        # Arrange
        model = DefaultSlippageModel(pct=0.001)  # 0.1% slippage
        price = 100.0
        qty = 5.0
        expected_price = 100.1

        # Act
        actual_price = model.apply(price, qty)

        # Assert
        self.assertAlmostEqual(actual_price, expected_price)

    def test_slippage_decreases_sell_execution_price(self):
        # Arrange
        model = DefaultSlippageModel(pct=0.001)  # 0.1% slippage
        price = 100.0
        qty = -5.0
        expected_price = 99.9

        # Act
        actual_price = model.apply(price, qty)

        # Assert
        self.assertAlmostEqual(actual_price, expected_price)


class TestTaxModel(unittest.TestCase):
    """Unit tests for the Capital Gains Tax Model using Arrange-Act-Assert (AAA)."""

    def test_tax_calculation_for_long_position_gain(self):
        # Arrange
        model = DefaultTaxModel(rate=0.20, deferred=False)
        qty = -10.0  # Selling
        price = 150.0  # Exit price
        cost_basis = 100.0  # Entry price
        holdings = 10.0
        comm = 5.0
        # Gain = 10 * (150 - 100) - 5 = 495. Tax = 495 * 0.20 = 99.0
        expected_tax = 99.0

        # Act
        actual_tax = model.calculate_tax(qty, price, cost_basis, holdings, comm)

        # Assert
        self.assertAlmostEqual(actual_tax, expected_tax)

    def test_tax_calculation_is_zero_for_losses(self):
        # Arrange
        model = DefaultTaxModel(rate=0.20, deferred=False)
        qty = -10.0
        price = 80.0  # Selling at a loss
        cost_basis = 100.0
        holdings = 10.0
        comm = 5.0
        expected_tax = 0.0

        # Act
        actual_tax = model.calculate_tax(qty, price, cost_basis, holdings, comm)

        # Assert
        self.assertEqual(actual_tax, expected_tax)

    def test_tax_calculation_for_short_position_gain(self):
        # Arrange
        model = DefaultTaxModel(rate=0.15, deferred=False)
        qty = 5.0  # Buying to cover short
        price = 80.0  # Exit price (lower, so gain)
        cost_basis = 100.0  # Entry price
        holdings = -5.0
        comm = 2.0
        # Gain = 5 * (100 - 80) - 2 = 98. Tax = 98 * 0.15 = 14.7
        expected_tax = 14.7

        # Act
        actual_tax = model.calculate_tax(qty, price, cost_basis, holdings, comm)

        # Assert
        self.assertAlmostEqual(actual_tax, expected_tax)


if __name__ == '__main__':
    unittest.main()
