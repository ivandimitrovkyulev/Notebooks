import backtrader as bt
import numpy as np


class ZScoreStrategy(bt.Strategy):
    params = (
        ('lookback', 20),  # Lookback period for calculating Z-score
        ('zscore_entry', 2),  # Z-score threshold for entry
        ('zscore_exit', 0.5),  # Z-score threshold for exit
    )

    def __init__(self):
        # Keep references to the two data feeds
        self.data1 = self.datas[0]
        self.data2 = self.datas[1]

        # Spread and Z-score variables
        self.spread = None
        self.zscore = None

    def next(self):
        # Ensure there are enough bars for the lookback period
        if len(self.data1) < self.params.lookback or len(self.data2) < self.params.lookback:
            return

        # Calculate the spread
        prices1 = np.array([self.data1.close[i] for i in range(-self.params.lookback, 0)])
        prices2 = np.array([self.data2.close[i] for i in range(-self.params.lookback, 0)])
        spread = prices1 - prices2

        # Calculate Z-score
        spread_mean = np.mean(spread)
        spread_std = np.std(spread)
        self.zscore = (spread[-1] - spread_mean) / spread_std

        # Logging (optional)
        print(f"Date: {self.data.datetime.date(0)}, Z-score: {self.zscore}")

        # Entry conditions
        if self.zscore >= self.params.zscore_entry and not self.getposition(self.data1) and not self.getposition(self.data2):
            # Long Asset 2, Short Asset 1
            self.sell(data=self.data1, size=100)  # Short Asset 1
            self.buy(data=self.data2, size=100)  # Long Asset 2

        elif self.zscore <= -self.params.zscore_entry and not self.getposition(self.data1) and not self.getposition(self.data2):
            # Long Asset 1, Short Asset 2
            self.buy(data=self.data1, size=100)  # Long Asset 1
            self.sell(data=self.data2, size=100)  # Short Asset 2

        # Exit conditions
        elif abs(self.zscore) <= self.params.zscore_exit:
            # Close positions when Z-score returns to normal range
            self.close(data=self.data1)
            self.close(data=self.data2)


class SmaCross(bt.Strategy):
    # list of parameters which are configurable for the strategy
    params = dict(
        pfast=10,  # period for the fast moving average
        pslow=30   # period for the slow moving average
    )

    def __init__(self):
        sma1 = bt.ind.SMA(period=self.p.pfast)  # fast moving average
        sma2 = bt.ind.SMA(period=self.p.pslow)  # slow moving average
        self.crossover = bt.ind.CrossOver(sma1, sma2)  # crossover signal

    def next(self):
        if not self.position:  # not in the market
            if self.crossover > 0:  # if fast crosses slow to the upside
                self.buy()  # enter long

        elif self.crossover < 0:  # in the market & cross to the downside
            self.close()  # close long position


class TestStrategy(bt.Strategy):
    params = (
        ("maperiod", 15),
        ("printlog", False),
    )

    def log(self, txt, dt=None, doprint=False):
        """Logging function fot this strategy"""
        if self.params.printlog or doprint:
            dt = dt or self.datas[0].datetime.date(0)
            print("%s, %s" % (dt.isoformat(), txt))

    def __init__(self):
        # Keep a reference to the "close" line in the data[0] dataseries
        self.dataclose = self.datas[0].close

        # To keep track of pending orders and buy price/commission
        self.order = None
        self.buyprice = None
        self.buycomm = None

        # Add a MovingAverageSimple indicator
        self.sma = bt.indicators.SimpleMovingAverage(self.datas[0], period=self.params.maperiod)

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            # Buy/Sell order submitted/accepted to/by broker - Nothing to do
            return

        # Check if an order has been completed
        # Attention: broker could reject order if not enough cash
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(
                    "BUY EXECUTED, Price: %.2f, Cost: %.2f, Comm %.2f"
                    % (order.executed.price, order.executed.value, order.executed.comm)
                )

                self.buyprice = order.executed.price
                self.buycomm = order.executed.comm
            else:  # Sell
                self.log(
                    "SELL EXECUTED, Price: %.2f, Cost: %.2f, Comm %.2f"
                    % (order.executed.price, order.executed.value, order.executed.comm)
                )

            self.bar_executed = len(self)

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log("Order Canceled/Margin/Rejected")

        # Write down: no pending order
        self.order = None

    def notify_trade(self, trade):
        if not trade.isclosed:
            return

        self.log("OPERATION PROFIT, GROSS %.2f, NET %.2f" % (trade.pnl, trade.pnlcomm))

    def next(self):
        # Simply log the closing price of the series from the reference
        self.log("Close, %.2f" % self.dataclose[0])

        # Check if an order is pending ... if yes, we cannot send a 2nd one
        if self.order:
            return

        # Check if we are in the market
        if not self.position:

            # Not yet ... we MIGHT BUY if ...
            if self.dataclose[0] > self.sma[0]:

                # BUY, BUY, BUY!!! (with all possible default parameters)
                self.log("BUY CREATE, %.2f" % self.dataclose[0])

                # Keep track of the created order to avoid a 2nd order
                self.order = self.buy()

        else:

            if self.dataclose[0] < self.sma[0]:
                # SELL, SELL, SELL!!! (with all possible default parameters)
                self.log("SELL CREATE, %.2f" % self.dataclose[0])

                # Keep track of the created order to avoid a 2nd order
                self.order = self.sell()

    def stop(self):
        self.log(
            "(MA Period %2d) Ending Value %.2f" % (self.params.maperiod, self.broker.getvalue()),
            doprint=True,
        )
