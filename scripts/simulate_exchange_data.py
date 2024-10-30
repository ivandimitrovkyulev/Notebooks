import zmq
import math
import time
import random


class InstrumentPrice(object):
    """Simulate price values of an instrument over time."""

    def __init__(self):
        self.symbol = "TICKER"
        self.t = time.time()
        self.value = 100.0
        self.sigma = 0.4
        self.r = 0.01

    def simulate_value(self):
        """Generates a new, random stock price."""
        t = time.time()
        dt = (t - self.t) / (252 * 8 * 60 * 60)
        dt *= 500
        self.t = t
        self.value *= math.exp((self.r - 0.5 * self.sigma**2) * dt + self.sigma * math.sqrt(dt) * random.gauss(0, 1))
        return self.value


if __name__ == "__main__":
    context = zmq.Context()
    socket = context.socket(zmq.PUB)
    socket.bind("tcp://0.0.0.0:5555")

    ip = InstrumentPrice()
    n = 1
    while True:
        msg = f"{ip.symbol} | {n} | Price: {ip.simulate_value():.2f}"
        print(msg)
        socket.send_string(msg)
        time.sleep(random.random() * 2)
        n += 1
