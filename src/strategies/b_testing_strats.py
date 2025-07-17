from datetime import timedelta

import numpy as np
import pandas as pd
from backtesting import Strategy
from backtesting.lib import crossover
from ta.momentum import RSIIndicator

from src import indicators


class RSICross(Strategy):
    """RSI crossover strategy. Long only by default."""

    low_threshold = 30
    high_threshold = 80
    long_only = True

    def init(self):
        # Precompute RSI
        self.rsi = self.I(lambda x: RSIIndicator(pd.Series(x)).rsi(), self.data.Close)

    def next(self):
        """
        If RSI is below low_threshold - buy.
        If RSI is above high_threshold - sell.
        """
        if self.rsi <= self.low_threshold:
            self.position.close()
            self.buy()

        elif self.rsi >= self.high_threshold:
            self.position.close()
            if not self.long_only:
                self.sell()


class DCA(Strategy):
    """Simple Dollar Cost Averaging Strategy"""

    buy_frequency_days = 30

    def init(self):
        # Calculate the amount per trade (fixed cash amount)
        no_trades = len(self.data.Close) / self.buy_frequency_days  # Number of trades
        amount_per_trade = self._broker._cash / no_trades  # Cash per trade
        self.percentage_trade = amount_per_trade / self._broker._cash

        print(no_trades)
        print(amount_per_trade)
        print(self.percentage_trade)

        # Store the last buy time (None at the beginning)
        self.last_buy_time = None

    def next(self):
        # Get the current time from the data index
        current_time = self.data.index[-1]

        # Buy if it's the first trade or enough days have passed
        if self.last_buy_time is None or (current_time - self.trades[0].entry_time >= timedelta(days=self.buy_frequency_days)):
            self.buy(size=self.percentage_trade)
            self.last_buy_time = current_time  # Update last buy time
            print(f"BOUGHT {self.percentage_trade} units at {self.data.Close[-1]} on {current_time}")


class PriceStrength(Strategy):
    """Simple Price Strength strategy. Long only by default."""
    lookback_period_days = 5
    hold_period_in_days = 2

    def init(self):
        # Precompute the Price Strength
        self.price_strength = self.I(
            indicators.recent_all_time_high,
            self.data.Close,
            self.lookback_period_days,
        )

    def next(self):
        """
        If last Price is >= than the lookback window's max Price - buy and then sell after 'hold_period_days' days.
        """
        if self.price_strength:
            if not self.position.is_long:
                self.buy()

        elif self.trades and self.data.index[-1] - self.trades[0].entry_time >= timedelta(
            days=self.hold_period_in_days
        ):
            self.position.close()


class VolumeSpike(Strategy):
    """Simple Volume Spike strategy. Long only by default."""

    ma_window = 7  # Volume average lag
    vol_multiplier = 4  # Volume multiplier
    hold_period_in_days = 1
    long_only = True

    def init(self):
        # Precompute the Volume Spike
        self.vol_level = self.I(
            indicators.simple_moving_average,
            self.data.Volume * self.vol_multiplier,
            self.ma_window,
        )

    def next(self):
        """
        If Volume is more than Spiked Volume - buy and then sell the next day.
        """
        if self.data.Volume > self.vol_level:
            self.position.close()
            self.buy()

        elif self.trades and self.data.index[-1] - self.trades[0].entry_time >= timedelta(
            days=self.hold_period_in_days
        ):
            self.position.close()
            if not self.long_only:
                self.sell()


class SmaCross(Strategy):
    """Simple moving average with crossover strategy. Long only by default."""

    n1 = 42  # 1st moving average lag
    n2 = 252  # 2nd moving average lag
    long_only = True

    def init(self):
        # Precompute the two moving averages
        self.sma1 = self.I(indicators.simple_moving_average, self.data.Close, self.n1)
        self.sma2 = self.I(indicators.simple_moving_average, self.data.Close, self.n2)

    def next(self):
        """
        If sma1 crosses above sma2, close positions and buy.
        If sma1 crosses below sma2, close positions and sell.
        """
        if crossover(self.sma1, self.sma2):
            self.position.close()
            self.buy()

        elif crossover(self.sma2, self.sma1):
            self.position.close()
            if not self.long_only:
                self.sell()


class MomentumTimeSeries(Strategy):
    """Momentum time series strategy with variable lookback time period. Long only by default."""

    lookback = 3  # Define the lookback period
    long_only = True

    def init(self):
        # Precompute daily log returns
        self.returns = self.I(indicators.daily_log_returns, self.data.Close)
        self.long_short = self.I(np.sign, pd.Series(self.returns).rolling(self.lookback).mean())

    def next(self):
        """
        If the average of last lookback returns > 0, close positions and buy.
        If the average of last lookback returns < 0, close positions and sell.
        """
        if self.long_short > 0:
            self.position.close()
            self.buy()

        elif self.long_short < 0:
            self.position.close()
            if not self.long_only:
                self.sell()


class MeanReversionLongOnly(Strategy):
    """Mean reversion long only strategy. Long only by default."""

    n = 25
    threshold = 3.5
    long_only = True

    def init(self):
        # Precompute the two moving averages
        self.returns = self.I(indicators.daily_log_returns, self.data.Close)
        self.sma = self.I(indicators.simple_moving_average, self.data.Close, self.n)
        self.distance = self.I(indicators.substract_values, self.data.Close, self.sma)
        self.distance_change = self.I(indicators.multiply_values, self.distance, np.roll(self.distance, 1))

    def next(self):
        if self.distance < -self.threshold:
            self.buy()

        elif self.distance > 0:
            self.position.close()
            if not self.long_only:
                self.sell()
