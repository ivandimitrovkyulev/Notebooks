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


def regression_price_chart(data: pd.DataFrame, y_axis: str = 'Close') -> None:
    """
    Plot specified data with Matplotlib.
    :param data: Stock history DataFrame
    :param y_axis: Which data to plot, eg. 'Close', 'Open'
    """
    # Reset index column so that we have integers to represent time for later analysis
    history = data.reset_index()

    # Reshape index column to 2D array for .fit() method
    time_as_int = np.array(history.index).reshape(-1, 1)
    data_y = history[y_axis]

    # Create LinearRegression Object
    model = LinearRegression()
    # Fit linear model using the train data set
    model.fit(time_as_int, data_y)
    regression_line = model.predict(time_as_int)

    date_col = history['Date']

    # Graph
    plt.figure(1, figsize=(16, 10))
    plt.title(f"{y_axis} price Linear Regression")
    plt.plot(date_col, data_y, label=f'{y_axis} Price')  # Plot data
    plt.plot(date_col, regression_line, color='r', label='Linear regression')  # Plot Regression line
    plt.xlabel('Date')
    plt.ylabel(f'{y_axis} Price')
    plt.legend()
    plt.show()
