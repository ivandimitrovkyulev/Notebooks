import os
import datetime
from typing import Literal

from dotenv import load_dotenv
import pandas as pd
from binance_historical_data import BinanceDataDumper

from src.utils import has_header
from src.rest_client import RestBaseClient


# Load env variables
load_dotenv()

data_freq_type = Literal["1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h", "1d"]


def create_crypto_universe(
        binance_data: dict,
        cmc_data: dict,
        limits: tuple = (10, 300),
        quote_asset: str = "USDT",
) -> list:
    """
    Creates and returns a list of crypto assets that are available for Perpetual trading on Binance
    and within specified ranking based on Coin Market Cap.
    :param binance_exchange_info: Binance  Exchange Info data
    :param  cryptocurrency_map_data:  CMC Exchange Assets data
    :param limits: Tuple of upper and lower Market Cap rankings to filter by
    :param quote_asset: Quote Asset of the perpetual
    """
    cmc_rankings = [coin["symbol"] for coin in cmc_data if limits[0] < coin["rank"] <= limits[1]]

    tradeable_assets = []
    for symbol in binance_data.get("symbols"):
        if symbol["isMarginTradingAllowed"] and symbol["quoteAsset"] == quote_asset:

            if symbol["baseAsset"] in cmc_rankings:
                tradeable_assets.append(symbol["symbol"])

    return tradeable_assets

class CMCAPIClient(RestBaseClient):
    """
    Class for interacting with Coin Market Cap's APIs.
    https://coinmarketcap.com/api/documentation/v1/#section/Quick-Start-Guide
    """
    def __init__(
            self,
            base_endpoint: str = "https://pro-api.coinmarketcap.com",
            api_token: str = os.getenv("CMC_API_KEY"),
            api_token_header_name: str = "X-CMC_PRO_API_KEY",
    ):
        super().__init__(
            base_endpoint=base_endpoint,
            api_token=api_token,
            api_token_header_name=api_token_header_name,
        )

    def cryptocurrency_map(self) -> dict:
        """
        Returns a mapping of all cryptocurrencies to unique CoinMarketCap ids. Per our Best Practices we recommend
        utilizing CMC ID instead of cryptocurrency symbols to securely identify cryptocurrencies with our other
        endpoints and in your own application logic. Each cryptocurrency returned includes typical identifiers
        such as name, symbol, and token_address for flexible mapping to id.
        :return:
        """
        response = self.get_request(endpoint="/v1/cryptocurrency/map")
        return response.json().get("data")


class BinanceAPIClient(RestBaseClient):
    """
    Class for interacting with Binance's APIs.
    https://binance-docs.github.io/apidocs
    """
    def __init__(
            self,
            base_endpoint: str = "https://api.binance.com",
            api_token: str = os.getenv("BINANCE_API_KEY"),
            api_token_header_name: str = "X-MBX-APIKEY",
    ):
        super().__init__(
            base_endpoint=base_endpoint,
            api_token=api_token,
            api_token_header_name=api_token_header_name,
        )

    def exchange_info(self) -> dict:
        response = self.get_request(endpoint="/api/v3/exchangeInfo")
        return response.json()

class BinanceDataManager:
    """Class for interacting with Binance data."""
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
            'Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close Time', 'Quote Volume', 'Count',
            'Taker Buy Volume', 'Taker Buy Quote Volume', 'Ignore',
        ]

        # Create a DataFrame for each file in provided directory
        frames = []
        for file in [os.path.join(abs_dirpath, file) for file in os.listdir(abs_dirpath)]:
            if has_header(file):
                frame = pd.read_csv(file, index_col=0, parse_dates=True, skiprows=1, names=column_names)
            else:
                frame = pd.read_csv(file, index_col=0, parse_dates=True, names=column_names)
            frames.append(frame)

        if not frames:
            raise ValueError(f"Not files found in {abs_dirpath}")
        # Combine Dataframes into 1 Dataframe
        df = pd.concat(frames)

        # Clean data
        df.index = pd.to_datetime(df.index, unit="ms")
        df = df.drop(labels=["Close Time", "Ignore", ], axis=1)
        df.dropna()
        df = df.resample(data_frequency).asfreq()
        df = df.fillna(method="ffill")
        df.sort_index()

        return df
