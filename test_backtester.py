import unittest
import numpy as np
import pandas as pd
from data_structures import Strategy
from portfolio_state import PortfolioState
from trading_models import DefaultCommissionModel, DefaultSlippageModel, DefaultTaxModel, UKTaxModel, USTaxModel, CurrencyConverter
from backtester import Backtester

class MockStrategy(Strategy):
    def __init__(self, signals_df: pd.DataFrame):
        self.signals_df = signals_df

    def generate_trades(self, data: pd.DataFrame) -> pd.DataFrame:
        return self.signals_df


class TestCommissionModel(unittest.TestCase):
    """Unit tests for the Commission Model using Arrange-Act-Assert (AAA)."""

    def test_commission_calculation_calculates_brokerage_fee(self):
        # Arrange
        # per_share = 0.5, pct = 0.01, flat = 2.0, minimum = 0.0
        model = DefaultCommissionModel(per_share=0.5, pct=0.01, flat=2.0, minimum=0.0)
        qty = 10.0
        price = 100.0
        # broker_comm = 10 * 0.5 + 10 * 100 * 0.01 + 2.0 = 5.0 + 10.0 + 2.0 = 17.0
        expected_commission = 17.0

        # Act
        actual_commission = model.calculate(qty, price)

        # Assert
        self.assertEqual(actual_commission, expected_commission)

    def test_commission_enforces_minimum_fee(self):
        # Arrange
        model = DefaultCommissionModel(per_share=0.0, pct=0.0, flat=0.0, minimum=5.0)
        qty = 1.0
        price = 10.0
        # broker_comm = 0.0 -> clamped to minimum 5.0
        expected_commission = 5.0

        # Act
        actual_commission = model.calculate(qty, price)

        # Assert
        self.assertAlmostEqual(actual_commission, expected_commission)


class TestCurrencyConverter(unittest.TestCase):
    """Unit tests for the CurrencyConverter using Arrange-Act-Assert (AAA)."""

    def test_convert_trade_applies_markup_correctly(self):
        # Arrange
        rates = pd.Series([1.25], index=[pd.Timestamp('2026-06-14')])
        converter = CurrencyConverter(fx_rates=rates, fx_pct=0.005)
        amount_usd = 1000.0
        rate = 1.25

        # Buy transaction: amount / (rate * (1 - fx_pct))
        expected_buy = 1000.0 / (1.25 * (1.0 - 0.005))
        
        # Sell transaction: amount / (rate * (1 + fx_pct))
        expected_sell = 1000.0 / (1.25 * (1.0 + 0.005))

        # Act
        actual_buy = converter.convert_trade(amount_usd, pd.Timestamp('2026-06-14'), is_buy=True, rate=rate)
        actual_sell = converter.convert_trade(amount_usd, pd.Timestamp('2026-06-14'), is_buy=False, rate=rate)

        # Assert
        self.assertAlmostEqual(actual_buy, expected_buy)
        self.assertAlmostEqual(actual_sell, expected_sell)

    def test_convert_deposit_applies_markup_correctly(self):
        # Arrange
        rates = pd.Series([1.25], index=[pd.Timestamp('2026-06-14')])
        converter = CurrencyConverter(fx_rates=rates, fx_pct=0.005)

        # Act & Assert
        # Case 1: Same currency -> no conversion
        self.assertEqual(converter.convert_deposit(100.0, 'USD', 'USD', 1.25), 100.0)

        # Case 2: GBP to USD -> amount * rate * (1 - fx_pct)
        self.assertAlmostEqual(converter.convert_deposit(100.0, 'GBP', 'USD', 1.25), 124.375)

        # Case 3: USD to GBP -> (amount / rate) * (1 - fx_pct)
        self.assertAlmostEqual(converter.convert_deposit(100.0, 'USD', 'GBP', 1.25), 79.6)

        # Case 4: EUR to USD (Arbitrary pair with USD account) -> amount * rate * (1 - fx_pct)
        self.assertAlmostEqual(converter.convert_deposit(100.0, 'EUR', 'USD', 1.10), 109.45)

        # Case 5: USD to EUR (Arbitrary pair with USD deposit) -> (amount / rate) * (1 - fx_pct)
        self.assertAlmostEqual(converter.convert_deposit(100.0, 'USD', 'EUR', 1.10), 90.45454545)

        # Case 6: EUR to GBP (No USD involved) -> raises ValueError
        with self.assertRaises(ValueError):
            converter.convert_deposit(100.0, 'EUR', 'GBP', 0.85)


class TestSlippageModel(unittest.TestCase):
    """Unit tests for the Slippage Model using Arrange-Act-Assert (AAA)."""

    def test_slippage_increases_buy_execution_price(self):
        # Arrange
        model = DefaultSlippageModel(pct=0.001)
        price = 100.0
        qty = 5.0
        expected_price = 100.1

        # Act
        actual_price = model.apply(price, qty)

        # Assert
        self.assertAlmostEqual(actual_price, expected_price)

    def test_slippage_decreases_sell_execution_price(self):
        # Arrange
        model = DefaultSlippageModel(pct=0.001)
        price = 100.0
        qty = -5.0
        expected_price = 99.9

        # Act
        actual_price = model.apply(price, qty)

        # Assert
        self.assertAlmostEqual(actual_price, expected_price)


class TestTaxModel(unittest.TestCase):
    """Unit tests for the Capital Gains Tax Model using Arrange-Act-Assert (AAA)."""

    def test_tax_model_returns_raw_realized_gain(self):
        # Arrange
        model = DefaultTaxModel(rate=0.20, deferred=False)
        qty = -10.0
        price = 150.0
        cost_basis = 100.0
        holdings = 10.0
        comm = 5.0
        # Gain = 10 * (150 - 100) - 5 = 495.0 (returns raw gain, not tax)
        expected_gain = 495.0

        # Act
        actual_gain = model.calculate_tax(qty, price, cost_basis, holdings, comm)

        # Assert
        self.assertAlmostEqual(actual_gain, expected_gain)

    def test_tax_model_returns_negative_value_for_realized_losses(self):
        # Arrange
        model = DefaultTaxModel(rate=0.20, deferred=False)
        qty = -10.0
        price = 80.0
        cost_basis = 100.0
        holdings = 10.0
        comm = 5.0
        # Loss = 10 * (80 - 100) - 5 = -205.0
        expected_loss = -205.0

        # Act
        actual_loss = model.calculate_tax(qty, price, cost_basis, holdings, comm)

        # Assert
        self.assertAlmostEqual(actual_loss, expected_loss)


class TestBacktesterEngine(unittest.TestCase):
    """Unit tests for the Backtester simulation engine using Arrange-Act-Assert (AAA)."""

    def setUp(self):
        # Prepare mock pricing data (MultiIndex columns)
        dates = pd.date_range(start='2026-06-01', periods=3, freq='D')
        columns = pd.MultiIndex.from_tuples([('ABC', 'Open'), ('ABC', 'Close')], names=['Ticker', 'Price'])
        self.data = pd.DataFrame(100.0, index=dates, columns=columns)
        self.data.loc[dates[0], ('ABC', 'Open')] = 100.0
        self.data.loc[dates[0], ('ABC', 'Close')] = 105.0
        self.data.loc[dates[1], ('ABC', 'Open')] = 110.0
        self.data.loc[dates[1], ('ABC', 'Close')] = 115.0
        self.data.loc[dates[2], ('ABC', 'Open')] = 120.0
        self.data.loc[dates[2], ('ABC', 'Close')] = 125.0

    def test_backtester_executes_buys_and_retains_correct_cash(self):
        # Arrange
        # A strategy signaling to buy 10 shares on the first day
        signals = pd.DataFrame(0.0, index=self.data.index, columns=['ABC'])
        signals.iloc[0] = 10.0
        strategy = MockStrategy(signals)
        
        # Backtester (execution_delay=1 to execute on the next day's Open, zero commission)
        tester = Backtester(
            data=self.data,
            strategy=strategy,
            initial_capital=5000.0,
            monthly_deposit=0.0,
            slippage_pct=0.0,
            execution_delay=1,
            execution_price_type='Open'
        )

        # Act
        results = tester.run()

        # ABC Open on Day 1 = 110.0. Cost of 10 shares = 1100.0.
        # Remaining cash should be 5000 - 1100 = 3900.0
        self.assertAlmostEqual(results['cash'].iloc[1], 3900.0)
        # ABC Close on Day 1 = 115.0. Holdings value = 10 * 115 = 1150.0
        self.assertAlmostEqual(results['holdings_value'].iloc[1], 1150.0)
        # Total equity = 3900 + 1150 = 5050.0
        self.assertAlmostEqual(results['total_equity'].iloc[1], 5050.0)

    def test_backtester_clamps_buy_orders_when_insufficient_cash(self):
        # Arrange
        # Attempt to buy 100 shares (costs 10,000) on 5,000 capital
        signals = pd.DataFrame(0.0, index=self.data.index, columns=['ABC'])
        signals.iloc[0] = 100.0
        strategy = MockStrategy(signals)
        
        tester = Backtester(
            data=self.data,
            strategy=strategy,
            initial_capital=5000.0,
            monthly_deposit=0.0,
            allow_margin=False,
            slippage_pct=0.0,
            execution_delay=1,
            execution_price_type='Open'
        )

        # Act
        results = tester.run()

        # Cash = 5000.0, ABC Open on Day 1 = 110.0. Max buy quantity should be clamped to 5000 / 110 = 45.4545 shares.
        # Remaining cash should be 0.0
        self.assertAlmostEqual(results['cash'].iloc[1], 0.0)
        self.assertAlmostEqual(results['holdings_value'].iloc[1], (5000.0 / 110.0) * 115.0)

    def test_backtester_splits_position_flips_and_applies_short_margin_checks(self):
        # Arrange
        # Start with 10 long shares, then signal -30 (flip to -20 short)
        signals = pd.DataFrame(0.0, index=self.data.index, columns=['ABC'])
        signals.iloc[0] = 10.0
        signals.iloc[1] = -30.0 # Position flip
        strategy = MockStrategy(signals)
        
        # Cash account (allow_margin=False), short selling disallowed
        tester = Backtester(
            data=self.data,
            strategy=strategy,
            initial_capital=5000.0,
            monthly_deposit=0.0,
            allow_margin=False,
            allow_short=False,
            slippage_pct=0.0,
            execution_delay=1,
            execution_price_type='Open'
        )

        # Act
        results = tester.run()

        # Day 1: Buy 10 shares at Day 1 Open (110.0). Cash = 3900.0. Holdings = 10.0
        # Day 2: Try to sell 30 shares at Day 2 Open (120.0). Since allow_short=False, it closes the long position (sells 10)
        # but is restricted from shorting the remaining 20. Net qty = -10.0.
        # Cash received = 10 * 120 = 1200.0. Cash should be 3900 + 1200 = 5100.0.
        # Holdings should be 0.0
        self.assertAlmostEqual(results['cash'].iloc[2], 5100.0)
        self.assertAlmostEqual(results['holdings_value'].iloc[2], 0.0)


class TestNewBacktesterFeatures(unittest.TestCase):
    def setUp(self):
        dates = pd.date_range(start='2026-06-01', periods=5, freq='D')
        columns = pd.MultiIndex.from_tuples([('ABC', 'Open'), ('ABC', 'Close')], names=['Ticker', 'Price'])
        self.data = pd.DataFrame(100.0, index=dates, columns=columns)
        self.data.loc[dates[0], ('ABC', 'Open')] = 100.0
        self.data.loc[dates[0], ('ABC', 'Close')] = 105.0
        self.data.loc[dates[1], ('ABC', 'Open')] = 110.0
        self.data.loc[dates[1], ('ABC', 'Close')] = 115.0
        self.data.loc[dates[2], ('ABC', 'Open')] = 120.0
        self.data.loc[dates[2], ('ABC', 'Close')] = 125.0
        self.data.loc[dates[3], ('ABC', 'Open')] = 130.0
        self.data.loc[dates[3], ('ABC', 'Close')] = 135.0
        self.data.loc[dates[4], ('ABC', 'Open')] = 140.0
        self.data.loc[dates[4], ('ABC', 'Close')] = 145.0

    def test_short_borrow_fee_is_deducted_correctly(self):
        # A strategy signaling to sell short 10 shares on Day 0
        signals = pd.DataFrame(0.0, index=self.data.index, columns=['ABC'])
        signals.iloc[0] = -10.0
        strategy = MockStrategy(signals)
        
        # Backtester (allow_short=True, allow_margin=True, borrow_rate=10% annual, execution_delay=1)
        tester = Backtester(
            data=self.data,
            strategy=strategy,
            initial_capital=5000.0,
            monthly_deposit=0.0,
            allow_short=True,
            allow_margin=True,
            annual_borrow_rate=0.10,
            slippage_pct=0.0,
            execution_delay=1,
            execution_price_type='Open'
        )

        results = tester.run()

        # Day 1: Short entry executes at Day 1 Open (110.0)
        # Cash receives proceeds: 10 * 110 = 1100.0. New Cash = 5000 + 1100 = 6100.0.
        self.assertAlmostEqual(results['cash'].iloc[1], 6100.0)

        # Day 2: Short position is carried from Day 1 to Day 2 (1 day elapsed)
        # Borrow fee is charged on the previous day (Day 1) Close value (10 * 115.0) to prevent lookahead bias
        expected_cash = 6100.0 - (10.0 * 115.0 * ((1.0 + 0.10/360.0) ** 1 - 1.0))
        self.assertAlmostEqual(results['cash'].iloc[2], expected_cash)

    def test_multi_currency_accounting(self):
        # A strategy signaling to buy 10 shares on Day 0
        signals = pd.DataFrame(0.0, index=self.data.index, columns=['ABC'])
        signals.iloc[0] = 10.0
        strategy = MockStrategy(signals)
        
        # FX rates: 1 GBP = 1.25 USD on all days
        fx_rates = pd.Series(1.25, index=self.data.index)
        
        # Account in GBP, stocks in USD, FX fee = 0.35%
        tester = Backtester(
            data=self.data,
            strategy=strategy,
            initial_capital=4000.0, # GBP
            monthly_deposit=0.0,
            account_currency='GBP',
            fx_rates=fx_rates,
            fx_pct=0.0035,
            slippage_pct=0.0,
            execution_delay=1,
            execution_price_type='Open'
        )

        results = tester.run()

        # Day 1: Buy 10 shares at Day 1 Open (110.0 USD)
        # Cost in USD = 1100.0 USD
        # Cost in GBP (including FX fee) = 1100.0 / (1.25 * (1 - 0.0035)) = 1100.0 / 1.245625 = 883.0908 GBP
        # FX fee is charged in the currency conversion step, and commission model fx_pct is set to 0.0 to prevent double charging.
        expected_cash = 4000.0 - 1100.0 / (1.25 * (1.0 - 0.0035))
        self.assertAlmostEqual(results['cash'].iloc[1], expected_cash)


class TestUKTaxModel(unittest.TestCase):
    """Unit tests for the UK CGT Tax Model using HMRC rules."""

    def test_uk_tax_gain_conversion_uses_date_specific_rates(self):
        # Arrange: 10 shares bought at $10 (FX rate 1.25) and sold at $15 (FX rate 1.50)
        # cost in GBP = 10 * 10 / 1.25 = 80 GBP
        # proceeds in GBP = 10 * 15 / 1.50 = 100 GBP
        # realized gain = 100 - 80 = 20 GBP
        model = UKTaxModel(rate=0.20, annual_allowance=3000.0)
        # realized_lots elements: (qty, buy_p, buy_rate, sell_p, sell_rate, days)
        realized_lots = [(10.0, 10.0, 1.25, 15.0, 1.50, 50)]

        # Act
        gain = model.calculate_realized_gain(realized_lots, comm=0.0)

        # Assert
        self.assertAlmostEqual(gain, 20.0)

    def test_uk_tax_annual_allowance_preserves_loss_carry_forward(self):
        # Arrange
        model = UKTaxModel(rate=0.20, annual_allowance=3000.0)
        
        # Scenario A: YTD gains <= annual allowance
        tax_due, new_carry = model.calculate_annual_tax(realized_gain_ytd=2000.0, tax_loss_carry_forward=1000.0)
        self.assertEqual(tax_due, 0.0)
        self.assertEqual(new_carry, 1000.0)  # Loss carry forward is not wasted

        # Scenario B: YTD gains > annual allowance, partially offset by carry forward
        # realized_gain_ytd = 5000.0, allowance = 3000.0 -> excess = 2000.0
        # carry forward is 1500.0 -> taxable = 2000 - 1500 = 500.0
        # tax = 500 * 20% = 100.0. new carry forward = 0.0
        tax_due, new_carry = model.calculate_annual_tax(realized_gain_ytd=5000.0, tax_loss_carry_forward=1500.0)
        self.assertEqual(tax_due, 100.0)
        self.assertEqual(new_carry, 0.0)

        # Scenario C: YTD gains > annual allowance, fully offset by carry forward
        # realized_gain_ytd = 5000.0, allowance = 3000.0 -> excess = 2000.0
        # carry forward is 3000.0 -> taxable = 0.0. new carry forward = 1000.0
        tax_due, new_carry = model.calculate_annual_tax(realized_gain_ytd=5000.0, tax_loss_carry_forward=3000.0)
        self.assertEqual(tax_due, 0.0)
        self.assertEqual(new_carry, 1000.0)

    def test_uk_tax_year_transition(self):
        model = UKTaxModel()
        # UK tax year ends April 5. April 5 to April 6 is a tax year transition.
        self.assertTrue(model.is_tax_year_end(pd.Timestamp('2026-04-06'), pd.Timestamp('2026-04-05')))
        # Mid-year or normal transition is not a tax year end.
        self.assertFalse(model.is_tax_year_end(pd.Timestamp('2026-04-05'), pd.Timestamp('2026-04-04')))
        self.assertFalse(model.is_tax_year_end(pd.Timestamp('2026-05-01'), pd.Timestamp('2026-04-30')))


class TestUSTaxModel(unittest.TestCase):
    """Unit tests for the US Tax Model distinguishing ST and LT gains."""

    def test_us_tax_splits_short_term_and_long_term_gains(self):
        # Arrange
        model = USTaxModel(short_term_rate=0.24, long_term_rate=0.15)
        # 10 shares held for 50 days (Short Term)
        # Buy: 10 * 10 / 1.0 = 100 USD. Sell: 10 * 15 / 1.0 = 150 USD. Gain: 50 USD.
        # 10 shares held for 400 days (Long Term)
        # Buy: 10 * 100 / 1.0 = 1000 USD. Sell: 10 * 120 / 1.0 = 1200 USD. Gain: 200 USD.
        realized_lots = [
            (10.0, 10.0, 1.0, 15.0, 1.0, 50),
            (10.0, 100.0, 1.0, 120.0, 1.0, 400)
        ]

        # Act
        gain = model.calculate_realized_gain(realized_lots, comm=0.0)

        # Assert
        self.assertAlmostEqual(gain, 250.0)
        self.assertEqual(model.st_gains_ytd, 50.0)
        self.assertEqual(model.lt_gains_ytd, 200.0)

        # Act: Calculate annual tax
        tax_due, new_carry = model.calculate_annual_tax(realized_gain_ytd=250.0, tax_loss_carry_forward=0.0)
        # Tax = 50 * 0.24 + 200 * 0.15 = 12 + 30 = 42.0
        self.assertEqual(tax_due, 42.0)
        self.assertEqual(new_carry, 0.0)


class TestMarginRiskTiming(unittest.TestCase):
    """Unit tests verifying margin risk controls execution timing and liquidation logic."""

    def test_close_price_execution_does_not_trigger_false_intraday_liquidation(self):
        # Arrange
        # Day 0: Open=100, High=100, Low=50, Close=100.
        # This has a massive intraday drop, but the user is buying at the Close.
        # They should not be liquidated because they did not hold the position during the day.
        dates = pd.date_range(start='2026-06-01', periods=1, freq='D')
        columns = pd.MultiIndex.from_tuples([
            ('ABC', 'Open'), ('ABC', 'High'), ('ABC', 'Low'), ('ABC', 'Close')
        ], names=['Ticker', 'Price'])
        data = pd.DataFrame(100.0, index=dates, columns=columns)
        data.loc[dates[0], ('ABC', 'Low')] = 50.0  # Big intraday drop
        
        signals = pd.DataFrame(0.0, index=data.index, columns=['ABC'])
        signals.iloc[0] = 10.0  # Buy 10 shares
        strategy = MockStrategy(signals)
        
        # Configure with max leverage 2.0, maintenance margin 25%, Close execution
        tester = Backtester(
            data=data,
            strategy=strategy,
            initial_capital=1000.0,
            monthly_deposit=0.0,
            allow_margin=True,
            max_leverage=2.0,
            maintenance_margin_pct=0.25,
            slippage_pct=0.0,
            execution_delay=0,  # Execute on same day
            execution_price_type='Close'
        )
        
        # Act
        results = tester.run()
        
        # Assert
        # They should successfully buy 10 shares without triggering liquidation
        self.assertFalse(tester.state.is_liquidated)
        self.assertEqual(tester.state.holdings[0], 10.0)
        # Cash should be 1000 - 10 * 100 = 0.0
        self.assertAlmostEqual(results['cash'].iloc[0], 0.0)
        self.assertAlmostEqual(results['holdings_value'].iloc[0], 1000.0)


if __name__ == '__main__':
    unittest.main()
