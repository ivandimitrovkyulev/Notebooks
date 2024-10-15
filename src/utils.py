"""Data and utilities for testing."""
import pandas as pd


class FileError(Exception):
    pass


def has_header(abs_filepath: str) -> bool:
    """
    Check if a file(.csv) has headers provided.
    :param abs_filepath: Absolute file path. If file is saved in /data, pass 'data/file_name.csv' as argument.
    :return: Boolean
    """
    with open(abs_filepath, "r") as file:
        first_line = file.readline().strip().split(",")
        # Check if the first row contains any non-numeric values (common heuristic)
        try:
            # Try to convert the first row to numeric values
            [float(item) for item in first_line]
            # If successful, likely no header
            return False
        except ValueError:
            # If conversion fails, it's likely a header row
            return True


def load_data(abs_filepath: str) -> pd.DataFrame:
    """
    Read a CSV data file and return a DataFrame.
    :param abs_filepath: Absolute file path. If file is saved in /data, pass 'data/file_name.csv' as argument.
    :returns: Pandas DataFrame object
    """
    try:
        file = pd.read_csv(
            abs_filepath,
            index_col=0, parse_dates=True,
        )

    except (FileNotFoundError, FileExistsError):
        raise FileError(f"File '{abs_filepath}' not found. Is file name correct and saved in the right folder?")

    return pd.DataFrame(file)
