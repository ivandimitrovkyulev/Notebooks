"""Plotting functions."""
from datetime import datetime, timezone
from typing import List
from itertools import combinations

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from sklearn.linear_model import LinearRegression

from src.indicators import simple_moving_average


def _get_start_end_date(data: pd.DataFrame) -> tuple[str, str]:
    start_date = data.index[0].date().strftime("%Y/%-m/%-d")
    end_date = data.index[-1].date().strftime("%Y/%-m/%-d")
    return start_date, end_date


def plot_bolinger_bands(
        data: pd.DataFrame,
        n_lookback: int,
        n_std: int,
        y_axis: str = 'Close',
        hide_data: bool = False,
        figsize: tuple = (16, 10),
) -> None:
    """
    Plot Bollinger bands indicator.
    :param data: Stock history OHLC DataFrame
    :param n_lookback: Lookback period
    :param n_std: Number of standard deviations
    :param y_axis: Which data to plot, eg. 'Close', 'Open'
    :param hide_data: Hide data on the chart
    :param figsize: Plot figure size
    """
    start_date, end_date = _get_start_end_date(data)
    plt.figure(1, figsize=figsize)
    plt.title(f"{y_axis} price | {start_date} - {end_date}")
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
        figsize: tuple = (16, 10),
) -> None:
    """
    Plot Simple Moving Averages.
    :param data: Stock history OHLC DataFrame
    :param smas: List of Simple Moving Averages
    :param y_axis: Which data to plot, eg. 'Close', 'Open'
    :param hide_data: Hide data on the chart and leave only SMAs
    :param figsize: Plot figure size
    """
    start_date, end_date = _get_start_end_date(data)

    plt.figure(1, figsize=figsize)
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

    plt.title(f"SMA | {start_date} - {end_date}")
    plt.legend()
    plt.show()


def plot_distance(
        data: pd.DataFrame,
        n: int,
        threshold: int | float,
        y_axis: str = 'Close',
        figsize: tuple = (16, 10),
) -> None:
    """
    Plot distance away from threshold.
    :param data: Stock history OHLC DataFrame
    :param n: Number of data points to roll
    :param threshold: Threshold distance up and down to plot horizontally
    :param y_axis: Which data to plot, eg. 'Close', 'Open'
    :param figsize: Plot figure size
    """
    start_date, end_date = _get_start_end_date(data)

    plt.figure(1, figsize=figsize)
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

    plt.title(f"Distance | {start_date} - {end_date}")
    plt.plot(time_col, distance, label="Distance")
    plt.legend()
    plt.show()


def plot_regression_line(
        data: pd.DataFrame,
        y_axis: str = 'Close',
        reg_line_count: int = 1,
        plot_vertical_line_separation: bool = True,
        log_scale: bool = False,
        figsize: tuple = (16, 10),
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
    :param figsize: Plot figure size
    """
    start_date, end_date = _get_start_end_date(data)

    data_clean = data[data[y_axis] >= 0]
    data_clean.dropna()  # Clean data
    plt.figure(1, figsize=figsize)
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
    title = f"Log {y_axis} price Linear Regression" if log_scale else f"{y_axis} price Linear Regression"
    title += f" | {start_date} - {end_date}"
    plt.title(title)
    plt.plot(time_col, data_y, label=f'{y_axis} Price')  # Plot data
    plt.plot(time_col, regression_line, color='g', label='Full Regression line')  # Plot Regression line
    plt.xlabel('Date')
    plt.ylabel(f'{y_axis} Price')
    if log_scale:
        plt.gca().yaxis.set_major_formatter(FuncFormatter((lambda y, pos: "%.3f" % (np.exp(y)))))
    plt.legend()
    plt.show()


def plot_n_chart_comparison(
        charts: List[tuple[str, pd.DataFrame]],
        y_axis: str = 'Close',
        log_scale: bool = False,
        sma_n: int | None = None,
        figsize: tuple = (16, 10),
) -> plt:
    """
    Plot a number of normalized charts on the same graph to visualise correlation.
    :param charts: List of tuples of chart name and chart data
    :param y_axis: Which data to plot, eg. 'Close', 'Open'
    :param log_scale: Whether to plot Y axis logarithmically
    :param sma_n: Smooth out graphs as SMAs
    :param figsize: Plot figure size
    """
    plt.figure(figsize=figsize)
    start_date, end_date = _get_start_end_date(charts[0][1])

    # Plot all charts with their individual full data
    for name, data in charts:
        # Normalize the data
        data = data[y_axis] / data[y_axis].iloc[0]
        if log_scale:
            data = np.log(data)
        # Plotting the normalized data
        if sma_n:
            data = simple_moving_average(data, sma_n)
        plt.plot(data, label=name)

    title = 'Log Normalized Stock Prices Comparison' if log_scale else 'Normalized Stock Prices Comparison'
    title += f" | {start_date} - {end_date}"
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Normalized Price')
    if log_scale:
        plt.gca().yaxis.set_major_formatter(FuncFormatter((lambda y, pos: "%.3f" % (np.exp(y)))))
    plt.legend()
    plt.grid(True)
    plt.show()

    all_indexes_cleaned = []
    for name, data in charts:
        # Normalize index of all charts to NYSE and start of day
        data.index = data.index.tz_convert('America/New_York').normalize()
        all_indexes_cleaned.append(data.index)

    # Display correlation pairs
    corr_names = combinations(map(lambda c: c[0], charts), 2)
    corr_datas = combinations(map(lambda c: c[1], charts), 2)

    for corr_data, corr_name in zip(corr_datas, corr_names):
        coefficient = corr_data[0][y_axis].corr(corr_data[1][y_axis])
        print(f"{corr_name[0]} / {corr_name[1]} correlation:".ljust(26) + f"{coefficient:,.8f}")

    return plt


def compare_assets(
        tickers: list[str],
        start_date: tuple[int, int, int],
        end_date: tuple[int, int, int],
        log_scale: bool = False,
) -> plt:
    """
    Compare different tickets and plot normalized charts on the same graph to visualise their correlation.
    :param tickers: List of Tickers
    :param start_date: Start date of comparison
    :param end_date: End date of comparison
    :param log_scale: Whether to plot Y axis logarithmically
    :return:
    """
    start = datetime(*start_date, tzinfo=timezone.utc)
    end = datetime(*end_date, tzinfo=timezone.utc)

    tickers_history = {}
    for ticker in tickers:
        tickers_history[ticker] = yf.Ticker(ticker.upper()).history(period='max').dropna()

    charts = [(ticker, history[start:end]) for ticker, history in tickers_history.items()]
    plot_n_chart_comparison(
        charts=charts,
        log_scale=log_scale,
    )

    return plt
