from urllib3 import Retry

import requests
from requests.adapters import HTTPAdapter


class RestBaseClient:
    """Class for interacting with REST APIs."""

    def __init__(self, base_endpoint: str, api_token: str, api_token_header_name: str):
        self.base_endpoint = base_endpoint
        if not api_token:
            raise ValueError(
                f"You need to provide an API KEY variable either when initialising this class "
                f"or as part of your env variables."
            )

        self.session = requests.Session()
        self.session.headers = {
            "Content-type": "application/json",
            api_token_header_name: api_token,
        }
        self.session.mount(
            prefix="https://",
            adapter=HTTPAdapter(
                max_retries=Retry(
                    total=5,
                    status_forcelist=[429, 500, 503],
                ),
            ),
        )
        print(f"Initialised {type(self).__name__!r} client with base url: {self.base_endpoint}")

    def get_request(self, endpoint: str, params: dict | None = None, data: dict | None = None) -> requests.Response:
        with self.session as sess:
            url = self.base_endpoint + endpoint
            response = sess.get(url=url, params=params, data=data)
            response.raise_for_status()
            return response
