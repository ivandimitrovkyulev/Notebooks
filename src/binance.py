"""Data and utilities for testing."""
import os
import datetime
from typing import Literal

import pandas as pd
from binance_historical_data import BinanceDataDumper

from src.utils import has_header


data_freq_type = Literal["1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h", "1d"]


class BinanceDataManager():

    @staticmethod
    def download_data(
            tickers: list,
            path_dir_where_to_dump: str = "./data",
            asset_class: str = "um",
            data_type: str = "klines",
            data_frequency: data_freq_type = "1h",
            date_start: datetime.date = None,
            date_end: datetime.date = None,
            is_to_update_existing: bool = True,
            int_max_tickers_to_get: int | None = None,
            tickers_to_exclude: list | None = None,
    ):
        """
        Download historical data from https://data.binance.vision/ for a given ticker.
        :param tickers: list trading pairs for which to dump databy default all ****USDT pairs will be taken
        :param path_dir_where_to_dump: Folder where to dump data
        :param asset_class: Asset class which data to get [spot, futures]
        :param data_type: data type to dump: [aggTrades, klines, trades]
        :param data_frequency: Frequency of price-volume data
        :param date_start: Date from which to start dump
        :param date_end: The last date for which to dump data
        :param is_to_update_existing: Flag if you want to update data if it's already exists
        :param int_max_tickers_to_get: Max number of trading pairs to get
        :param tickers_to_exclude: list trading pairs which to exclude from dump
        """
        data_dumper = BinanceDataDumper(
            path_dir_where_to_dump=path_dir_where_to_dump,
            asset_class=asset_class,  # spot, um, cm
            data_type=data_type,  # aggTrades, klines, trades
            data_frequency=data_frequency,
        )
        data_dumper.dump_data(
            tickers=tickers,
            date_start=date_start,
            date_end=date_end,
            is_to_update_existing=is_to_update_existing,
            int_max_tickers_to_get=int_max_tickers_to_get,
            tickers_to_exclude=tickers_to_exclude,
        )

    @staticmethod
    def data_to_df(
            abs_dirpath: str,
            data_frequency: data_freq_type = "1h",
    ) -> pd.DataFrame:
        """
        Given an absolute directory path, create a DataFrame for each file in it and then return a combined DataFrame
        of all single DataFrames.
        :param abs_dirpath: Absolute directory path.
        :param data_frequency: Frequency of price-volume data
        :return: Concatenated Dataframe
        """
        column_names = [
            'open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_volume', 'count',
            'taker_buy_volume', 'taker_buy_quote_volume', 'ignore',
        ]

        # Create a DataFrame for each file in provided directory
        frames = []
        for file in [os.path.join(abs_dirpath, file) for file in os.listdir(abs_dirpath)]:
            if has_header(file):
                frame = pd.read_csv(file, index_col=0, parse_dates=True, )
            else:
                frame = pd.read_csv(file, index_col=0, parse_dates=True, names=column_names)
            frames.append(frame)
        # Combine Dataframes into 1 Dataframe
        df = pd.concat(frames)

        # Clean data
        df.index = pd.to_datetime(df.index, unit="ms")
        df = df.drop("close_time", axis=1)
        df.dropna()
        df = df.resample(data_frequency).asfreq()
        df = df.fillna(method="ffill")
        df.sort_index()

        return df
