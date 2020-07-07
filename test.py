import MetaTrader5 as mt5
import numpy as np
import logging
import datetime as dt
import pytz


# logging.info("adfdsfsd")
# if not mt5.initialize():
#     mt5.shutdown()
# logging.info("add" + str(mt5.TIMEFRAME_M5))
# print(mt5.TIMEFRAME_M5)
# print(mt5.TIMEFRAME_M1)
# print(mt5.TIMEFRAME_H1)

timezone = pytz.timezone("Etc/UTC")
print(dt.datetime.fromtimestamp(1594104300, tz=timezone))
print(dt.datetime.fromtimestamp(1594104300))

# 2020.07.06 16:11:15.574	MMTE (EURUSD_i,M5)	1594048216


a = [0, 1, 2, 3]
# print(a[-4:])
