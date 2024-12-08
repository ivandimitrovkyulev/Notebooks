import operator
from functools import reduce
from typing import Any

import numpy as np
import pandas as pd


def prod(iterable):
    return reduce(operator.mul, iterable)


def sub(iterable):
    return reduce(operator.sub, iterable)


def tdiv(iterable):
    return reduce(operator.truediv, iterable)


def daily_log_returns(values: Any) -> np.ndarray:
    """Calculate log returns for each day."""
    returns = np.log(values / np.roll(values, 1))
    returns[0] = None
    return returns


def z_score(df_1: pd.DataFrame, df_2: pd.DataFrame, lookback: int = 25, column: str = "Close") -> pd.Series:
    """Calculate the Z Score between 2 assets."""
    spread = df_1[column] - df_2[column]
    spread.dropna(inplace=True)

    mean_spread = spread.rolling(window=lookback).mean()
    std_spread = spread.rolling(window=lookback).std()

    # Calculate Z-Score
    z_score = (spread - mean_spread) / std_spread

    # Drop rows with NaN values (due to rolling calculations)
    z_score.dropna(inplace=True)

    return z_score


def simple_moving_average(values: Any, n: int) -> pd.Series:
    """
    Return simple moving average of `values`, at
    each step taking into account `n` previous values.
    """
    return pd.Series(values).rolling(n).mean()


def roll_array(array: np.array, n: int):
    rolled_array = np.roll(array, n)
    rolled_array[:n] = np.nan
    return rolled_array


def add_values(*args) -> Any:
    return sum(args)


def multiply_values(*args) -> Any:
    return prod(args)


def substract_values(*args) -> Any:
    return sub(args)


def divide_values(*args) -> Any:
    return tdiv(args)
