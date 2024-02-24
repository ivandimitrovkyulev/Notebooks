import numpy as np
import pandas as pd
from backtesting.lib import crossover
from backtesting import Strategy
from src.indicators import (
    simple_moving_average,
    daily_log_returns,
)


class SmaCross(Strategy):
    # Define the two MA lags as *class variables*
    # for later optimization
    n1 = 42
    n2 = 252

    def init(self):
        # Precompute the two moving averages
        self.sma1 = self.I(simple_moving_average, self.data.Close, self.n1)
        self.sma2 = self.I(simple_moving_average, self.data.Close, self.n2)

    def next(self):
        # If sma1 crosses above sma2, close any existing
        # short trades, and buy the asset
        if crossover(self.sma1, self.sma2):
            self.position.close()
            self.buy()

        # Else, if sma1 crosses below sma2, close any existing
        # long trades, and sell the asset
        elif crossover(self.sma2, self.sma1):
            self.position.close()
            self.sell()


class MomentumTimeSeries(Strategy):
    # Define the lookback period
    lookback = 3

    def init(self):
        # Precompute the returns
        self.returns = self.I(daily_log_returns, self.data.Close)
        self.long_short = self.I(np.sign, pd.Series(self.returns).rolling(self.lookback).mean())

    def next(self):
        if self.long_short > 0:
            self.position.close()
            self.buy()

        elif self.long_short < 0:
            self.position.close()
            self.sell()
