"""Data and utilities for testing."""
import pandas as pd


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
