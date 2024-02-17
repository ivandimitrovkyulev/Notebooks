"""Data and utilities for testing."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


class FileError(Exception):
    pass


def load_data(absolute_filepath: str) -> pd.DataFrame:
    """
    Read a CSV data file and return a DataFrame.
    :param absolute_filepath: Absolute file path. If file is saved in /data, pass 'data/file_name.csv' as argument.
    :returns: Pandas DataFrame object
    """
    try:
        file = pd.read_csv(
            absolute_filepath,
            index_col=0, parse_dates=True,
        )

    except (FileNotFoundError, FileExistsError):
        raise FileError(f"File '{absolute_filepath}' not found. Is file name correct and saved in data/ folder?")

    return pd.DataFrame(file)


def simple_moving_average(
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


def regression_price_chart(
        data: pd.DataFrame,
        y_axis: str = 'Close',
        reg_line_count: int = 1,
        plot_vertical_line_separation: bool = True,
) -> None:
    """
    Plot data with Matplotlib and fit 1 overall regression line and n regression lines split into equally sized
    timeframes.
    :param data: Stock history OHLC DataFrame
    :param y_axis: Which data to plot, eg. 'Close', 'Open'
    :param reg_line_count: Number of timeframes lines to split and plot. If data has 9 rows and reg_line_count=3, then
        data will be split into [0, 1, 2], [3, 4, 5], [6, 7, 8]
    :param plot_vertical_line_separation: Plot vertical line to separate regression line splits?
    """
    plt.figure(1, figsize=(16, 10))
    # Reset index column so that we have integers to represent time for later analysis
    history = data.reset_index()
    # Fit linear model using the train data set
    model = LinearRegression()

    time_col = history['Date']
    data_y = history[y_axis]

    split_range = int(len(history) / reg_line_count)
    for split_history in [history[i:i + split_range] for i in range(0, len(history), split_range)]:
        # Reshape index column to 2D array for .fit() method`
        time_as_int = np.array(split_history.index).reshape(-1, 1)
        split_data_y = split_history[y_axis]
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
    plt.legend()
    plt.show()
