Notebooks
======
### version v0.2.0

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


### Run

Run `jupyter notebook` and start by looking at some of the scripts in `/sample` directory:

```shell
jupyter notebook
```
