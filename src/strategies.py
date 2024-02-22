import numpy as np
import pandas as pd
from backtesting import Strategy
from backtesting.lib import crossover


def daily_log_returns(values):
    """Calculate log returns for each day."""
    returns = np.log(values / np.roll(values, 1))
    returns[0] = None
    return returns


def simple_moving_average(values, n: int):
    """
    Return simple moving average of `values`, at
    each step taking into account `n` previous values.
    """
    return pd.Series(values).rolling(n).mean()


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
    lookback = 1

    def init(self):
        # Precompute the returns
        self.returns = self.I(daily_log_returns, self.data.Close)
        self.long_short = self.I(np.sign, np.roll(self.returns, -1))

    def next(self):
        if self.long_short > 0:
            self.position.close()
            self.buy()

        elif self.long_short < 0:
            self.position.close()
            self.sell()
