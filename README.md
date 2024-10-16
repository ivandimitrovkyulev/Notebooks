Notebooks
======
### version v0.3.7

------
A collection of Notebooks for analysing stock charts, correlations, balance sheets, backtesting
different strategies, plotting graphs and more.


### Installation

This project uses **[Python 3.11](https://www.python.org/downloads/)** and **[Poetry](https://python-poetry.org/docs/#installation)**

Clone project and navigate in project folder:
```shell
git clone https://github.com/ivandimitrovkyulev/Notebooks.git

cd Notebooks
```

Activate virtual environment and install all third-party project dependencies:
```shell
# Sets Poetry configuration so it creates a virtual environment inside project root folder
poetry config --local virtualenvs.in-project true

# Create a virtual environment
poetry shell

# Install all project dependencies
poetry install
```

Saved the following variables in a `.env` file:
```dotenv
FRED_API_KEY=<your-api-key>
FRED_HOST_URL="https://api.stlouisfed.org/"
BINANCE_API_KEY=<your-api-key>
```


### Run

Run `jupyter notebook` and start by looking at some of the scripts in `/sample` directory:

```shell
jupyter notebook
```

### Example

```python
import yfinance as yf
from src.plotting import plot_n_chart_comparison


ticker_1 = "GC=F"
ticker_2 = "^GSPC"
ticker_3 = "CL=F"
period = "23y"
stock_1 = yf.Ticker(ticker_1.upper()).history(period=period).dropna()
stock_2 = yf.Ticker(ticker_2.upper()).history(period=period).dropna()
stock_3 = yf.Ticker(ticker_3.upper()).history(period=period).dropna()

plot_n_chart_comparison([(ticker_1, stock_1), (ticker_2, stock_2), (ticker_3, stock_3)], log_scale=True)
```

Will display the following:
> ![plot_n_chart_comparison.png](sample%2Fimages%2Fplot_n_chart_comparison.png)
```text
GC=F / ^GSPC correlation: 0.74688249
GC=F / CL=F correlation:  0.50038580
^GSPC / CL=F correlation: 0.15423993
```
