import MetaTrader5 as mt5


def initmt5():
    # подключимся к MetaTrader 5
    if not mt5.initialize():
        mt5.shutdown()
        return False
    return mt5


def get_rates(mt5, from_date, to_date):
    rates = mt5.copy_rates_range(
        "EURUSD", mt5.TIMEFRAME_M5, from_date, to_date)
    return rates
