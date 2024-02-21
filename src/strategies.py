import pandas as pd
from backtesting import Strategy, Backtest
from backtesting.lib import crossover
from src.utils import load_data


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


import pandas as pd
from backtesting import Strategy, Backtest
from backtesting.lib import crossover
from src.utils import load_data


def momentum_returns(values):
    """Calculate returns for each day."""
    return np.log(values / np.roll(values, 1))


class MomentumTimeSeries(Strategy):
    # Define the lookback period
    lookback = 1

    def init(self):
        # Precompute the returns
        self.returns = self.I(momentum_returns, self.data.Close)

    def next(self):
        if self.returns > 0:
            self.position.close()
            self.buy()

        elif self.returns < 0:
            self.position.close()
            self.sell()


data = load_data("data/EUR-USD.csv")
bt = Backtest(data, MomentumTimeSeries, cash=10_000, commission=0)
stats = bt.run()
bt.plot()
