import os
import requests
import pandas as pd


def _query_stlouisfed(
        endpoint: str,
        params: dict | None = None,
        host_url: str = os.getenv("FRED_HOST_URL"),
        timeout: int | float = 10,
) -> dict:
    """
    Make a request to 'https://fred.stlouisfed.org'
    :param endpoint: API endpoint. Must be with leading and without trailing slash
    :param params: Provide parameters as part of the request
    :param host_url: Host URL
    :param timeout: Max number of seconds to wait for response
    :return: Dictionary response
    """
    url = host_url + endpoint

    if not params:
        params = {}
    params.update(
        {
            'file_type': 'json',
            'api_key': os.getenv('FRED_API_KEY'),
        }
    )

    response = requests.get(url, params=params, timeout=timeout)
    response.raise_for_status()

    return response.json()


def search_series(search_text: str) -> pd.DataFrame:
    """
    Get economic data series that match search text.
    :param search_text: Text to search for.
    :return: Pandas DataFrame
    """
    data = _query_stlouisfed("/fred/series/search", {'search_text': str(search_text)})

    return pd.DataFrame(data['seriess'])


def get_series_observation(series_id: str) -> pd.DataFrame:
    """
    Get the observations or data values for an economic data series.
    :param series_id: The id for a series.
    :return: Pandas DataFrame
    """
    data = _query_stlouisfed("/fred/series/observations", {'series_id': str(series_id)})
    y_units = data['units']

    df = pd.DataFrame(data['observations'], columns=['date', 'value'])

    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df['value'] = pd.to_numeric(df['value'], errors='coerce')
    df.dropna()

    df.rename(columns={'value': 'Close', 'date': 'Date'}, inplace=True)
    df.set_index('Date', inplace=True)

    return df
