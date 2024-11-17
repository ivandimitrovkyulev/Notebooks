from datetime import timedelta

import numpy as np
import pandas as pd
from backtesting.lib import crossover
from backtesting import Strategy
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
