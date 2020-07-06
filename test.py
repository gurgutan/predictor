import MetaTrader5 as mt5
import numpy as np


if not mt5.initialize():
    mt5.shutdown()

print(mt5.TIMEFRAME_M5)
print(mt5.TIMEFRAME_M1)
print(mt5.TIMEFRAME_H1)

a = [0, 1, 2, 3]
print(a[-4:])
