"""Plotting functions."""
from typing import List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from sklearn.linear_model import LinearRegression

from src.indicators import simple_moving_average


def plot_bolinger_bands(
        data: pd.DataFrame,
        n_lookback: int,
        n_std: int,
        y_axis: str = 'Close',
        hide_data: bool = False,
) -> None:
    """
    Plot Bollinger bands indicator.
    :param data: Stock history OHLC DataFrame
    :param n_lookback: Lookback period
    :param n_std: Number of standard deviations
    :param y_axis: Which data to plot, eg. 'Close', 'Open'
    :param hide_data: Hide data on the chart
    """
    plt.figure(1, figsize=(16, 10))
    plt.title(f"{y_axis} price")
    plt.xlabel('Date')
    plt.ylabel(f'{y_axis} Price')

    time_col = data.index
    data_y = data[y_axis]
    if not hide_data:
        plt.plot(time_col, data_y, label=f'{y_axis} Price')  # Plot data

    hlc3 = (data['High'] + data['Low'] + data['Close']) / 3
    mean = hlc3.rolling(n_lookback).mean()
    std = hlc3.rolling(n_lookback).std()
    upper_band = mean + n_std * std
    lower_band = mean - n_std * std

    plt.plot(time_col, upper_band, label='Upper Band')
    plt.plot(time_col, lower_band, label='Lower Band')

    plt.legend()
    plt.show()


def plot_sma(
        data: pd.DataFrame,
        smas: list,
        y_axis: str = 'Close',
        hide_data: bool = False,
) -> None:
    """
    Plot Simple Moving Averages.
    :param data: Stock history OHLC DataFrame
    :param smas: List of Simple Moving Averages
    :param y_axis: Which data to plot, eg. 'Close', 'Open'
    :param hide_data: Hide data on the chart and leave only SMAs
    """
    plt.figure(1, figsize=(16, 10))
    plt.title(f"{y_axis} price")
    plt.xlabel('Date')
    plt.ylabel(f'{y_axis} Price')

    time_col = data.index
    data_y = data[y_axis]
    if not hide_data:
        plt.plot(time_col, data_y, label=f'{y_axis} Price')  # Plot data

    for n in smas:
        sma = pd.Series(data_y).rolling(n).mean()
        plt.plot(time_col, sma, label=f'{n} SMA')  # Plot SMA

    plt.legend()
    plt.show()


def plot_distance(
        data: pd.DataFrame,
        n: int,
        threshold: int | float,
        y_axis: str = 'Close',
):
    plt.figure(1, figsize=(16, 10))
    plt.title(f"{y_axis} price")
    plt.xlabel('Date')
    plt.ylabel(f'{y_axis} Price')

    time_col = data.index
    price = data[y_axis]
    sma = pd.Series(price).rolling(n).mean()
    distance = price - sma

    plt.axhline(threshold, color='r')
    plt.axhline(-threshold, color='r')
    plt.axhline(0, color='r')
    
    plt.plot(time_col, distance, label="Distance")
    plt.legend()
    plt.show()


def plot_regression_line(
        data: pd.DataFrame,
        y_axis: str = 'Close',
        reg_line_count: int = 1,
        plot_vertical_line_separation: bool = True,
        log_scale: bool = False,
) -> None:
    """
    Plot data with Matplotlib and fit 1 overall regression line and n regression lines split into equally sized
    timeframes.
    :param data: Stock history OHLC DataFrame
    :param y_axis: Which data to plot, eg. 'Close', 'Open'
    :param reg_line_count: Number of timeframes lines to split and plot. If data has 9 rows and reg_line_count=3, then
        data will be split into [0, 1, 2], [3, 4, 5], [6, 7, 8]
    :param plot_vertical_line_separation: Plot vertical line to separate regression line splits?
    :param log_scale: Whether to plot Y axis logarithmically
    """
    data_clean = data[data[y_axis] >= 0]
    data_clean.dropna()  # Clean data
    plt.figure(1, figsize=(16, 10))
    # Reset index column so that we have integers to represent time for later analysis
    history = data_clean.reset_index()
    # Fit linear model using the train data set
    model = LinearRegression()

    time_col = history['Date']
    data_y = np.log(history[y_axis]) if log_scale else history[y_axis]

    split_range = int(len(history) / reg_line_count)
    for split_history in [history[i:i + split_range] for i in range(0, len(history), split_range)]:
        # Reshape index column to 2D array for .fit() method`
        time_as_int = np.array(split_history.index).reshape(-1, 1)
        split_data_y = np.log(split_history[y_axis]) if log_scale else split_history[y_axis]
        split_time_col = split_history['Date']

        model.fit(time_as_int, split_data_y)
        regression_line_split = model.predict(time_as_int)
        plt.plot(split_time_col, regression_line_split, color='r')  # Plot Regression line

        if plot_vertical_line_separation:
            plt.axvline(x=split_time_col.iloc[-1], color='y', linestyle='--')  # Plot vertical line

    # Reshape index column to 2D array for .fit() method`
    time_as_int = np.array(history.index).reshape(-1, 1)
    model.fit(time_as_int, data_y)
    regression_line = model.predict(time_as_int)

    # Graph
    plt.title(f"{y_axis} price Linear Regression")
    plt.plot(time_col, data_y, label=f'{y_axis} Price')  # Plot data
    plt.plot(time_col, regression_line, color='g', label='Full Regression line')  # Plot Regression line
    plt.xlabel('Date')
    plt.ylabel(f'{y_axis} Price')
    if log_scale:
        plt.gca().yaxis.set_major_formatter(FuncFormatter((lambda y, pos: "%.3f"%(np.exp(y)))))
    plt.legend()
    plt.show()


def plot_n_chart_comparison(
        charts: List[tuple[str, pd.DataFrame]],
        y_axis: str = 'Close',
        log_scale: bool = False,
        sma_n: int | None = None,
)-> None:
    """
    Plot a number of normalized charts on the same graph to visualise correlation.
    :param charts: List of tuples of chart name and chart data
    :param y_axis: Which data to plot, eg. 'Close', 'Open'
    :param log_scale: Whether to plot Y axis logarithmically
    :param sma_n: Smooth out graphs as SMAs
    """
    plt.figure(figsize=(10, 6))

    for name, data in charts:
        # Normalize the data
        data_normalized = data[y_axis] / data[y_axis].iloc[0]
        if log_scale:
            data_normalized = np.log(data_normalized)
        # Plotting the normalized data
        if sma_n:
            data_normalized = simple_moving_average(data_normalized, sma_n)
        plt.plot(data_normalized, label=name)

    plt.title('Normalized Stock Prices Comparison')
    plt.xlabel('Date')
    plt.ylabel('Normalized Price')
    if log_scale:
        plt.gca().yaxis.set_major_formatter(FuncFormatter((lambda y, pos: "%.3f"%(np.exp(y)))))
    plt.legend()
    plt.grid(True)
    plt.show()
