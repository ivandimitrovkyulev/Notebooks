import numpy as np
import pandas as pd


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
