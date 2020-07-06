import MetaTrader5 as mt5
from timer import DelayTimer
import numpy as np
import pandas as pd
import datetime as dt

# ORDER_TYPE_SELL=-1
# ORDER_TYPE_BUY=1


class Adviser:
    def __init__(self, predictor, delay=300, symbol="EURUSD"):
        if not mt5.initialize():
            print("Ошибка подключения")
            self.ready = False
            mt5.shutdown()
            return
        self.predictor = predictor
        self.sl = 1000
        self.tp = 1000
        self.max_vol = 1
        self.vol = 0.1
        self.confidence = 0.4
        self.std = 0.003
        self.symbol = symbol
        self.timeunit = 300  # секунд
        self.ready = True

    def _get_pos_vol(self):
        positions = mt5.positions_get(symbol=self.symbol)
        if positions == None:
            print("Не найдено открытых позиций, ошибка={}".format(mt5.last_error()))
            return None
        # df=pd.DataFrame(list(positions),columns=usd_positions[0]._asdict().keys())
        # ticket time type magic identifier reason volume price_open sl tp price_current swap profit symbol comment
        vol = 0
        for pos in positions:
            type_mult = -pos.type * 2 - 1  # -1=sell, +1=buy
            vol += pos.volume * type_mult
        return pos

    def deal(self):
        trend = max(-1, min(1, self.get_trend()))
        targ_vol = self.max_vol * trend
        pos_vol = self._get_pos_vol()
        d = targ_vol - pos_vol
        if d == 0:
            return
        if d >= self.vol and pos_vol < self.max_vol:
            self.order(order_type=-1, volume=self.vol)
        if -d >= self.vol and -pos_vol < self.max_vol:
            self.order(order_type=1, volume=self.vol)

    def get_trend(self):
        cur_date, cur_price, future_date, low, high, confidence = self.compute()
        d = (low + high) / 2 - cur_price
        trend = d / self.std
        return trend

    def order(self, order_type, volume):
        if order_type == 1:
            result = mt5.Buy(symbol=self.symbol, volume=vol)
        elif order_type == -1:
            result = mt5.Sell(symbol=self.symbol, volume=vol)
        return result

    def compute(self):
        """
        Вычисление прогноза на текущую дату
        на выходе: (low, high, confidence)
        """
        input_size = self.predictor.datainfo._in_size() + 1
        delta = dt.timedelta(seconds=input_size * self.timeunit)
        cur_date = dt.datetime.now()
        from_date = date - delta
        # TODO заменить mt5.TIMEFRAME_M5 на сохраненное в config.json значение
        rates = mt5.copy_rates_range(
            self.symbol, mt5.TIMEFRAME_M5, date_from=from_date, date_to=cur_date,
        )
        if len(rates) <= input_size:
            print("Нет котировок")
            return None
        closes = rates["close"][-input_size:]
        times = rates["time"][-input_size:]
        future_date = int(times[-1] + self.timeunit * self.predictor.datainfo.future)
        low, high, confidence = self.predictor.eval(closes)
        return (times[-1], closes[-1], future_date, low, high, confidence)

    def run(self):
        if not self.ready:
            print("Робот не готов к торговле")
            return False
        dtimer = DelayTimer(600)
        while True:
            if not dtimer.elapsed():
                continue
            self.deal()


def main():
    predictor = Predictor(modelname="models/19")
    if not predictor.trained:
        print("Ошибка инициализации модели")
        return False
    return True
    adviser = Adviser(predictor=predictor)
    if not adviser.ready:
        print("Ошибка инициализации робота")
        return
    adviser.start()


# -----------------------------------------------------------
if __name__ == "__main__":
    main()
