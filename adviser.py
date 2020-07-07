import MetaTrader5 as mt5
from timer import DelayTimer
import numpy as np
import pandas as pd
import datetime as dt
import logging
from predictor import Predictor

# ORDER_TYPE_SELL=-1
# ORDER_TYPE_BUY=1


class Adviser:
    def __init__(self, predictor, delay=300, symbol="EURUSD"):
        self.__init_logger__()
        logging.debug("Подключение к терминалу МТ5")
        if not mt5.initialize():
            logging.error("Ошибка подключения")
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
        self.delay = 300  # секунд
        self.mt5path = "D:/Dev/MT5/terminal64.exe"
        self.ready = True

    def __init_mt5__(self):
        logging.info("Подключение к терминалу MT5 ")
        # if not mt5.initialize(path="D:/Dev/Alpari MT5/terminal64.exe"):   # реальный
        if not mt5.initialize(path=self.mt5path):  # тестовый
            logging.error("Ошибка подключпения к терминалу MT5")
            mt5.shutdown()
            return False
        logging.info("... версия MT5:" + str(mt5.version()))
        return True

    def __init_logger__(self):
        logging.basicConfig(
            # filename="srv.log",
            level=logging.DEBUG,
            format="%(asctime)s [%(levelname)s] %(message)s",
            # filemode="w",
        )
        return True

    def _get_pos_vol(self):
        positions = mt5.positions_get(symbol=self.symbol)
        if positions == None:
            logging.error(
                "Не найдено открытых позиций, ошибка={}".format(mt5.last_error())
            )
            return None
        # df=pd.DataFrame(list(positions),columns=usd_positions[0]._asdict().keys())
        # ticket time type magic identifier reason volume price_open sl tp price_current swap profit symbol comment
        vol = 0
        for pos in positions:
            type_mult = -pos.type * 2 - 1  # -1=sell, +1=buy
            vol += pos.volume * type_mult
        return pos

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
        на выходе: (rdate, rclose, future_date, low, high, confidence)
        """
        rates_count = self.predictor.datainfo._in_size() + 1
        # TODO заменить mt5.TIMEFRAME_M5 на сохраненное в config.json значение
        rates = mt5.copy_rates_from_pos(
            self.symbol, mt5.TIMEFRAME_M5, start_pos=0, count=rates_count,
        )
        if len(rates) <= rates_count:
            logging.error("Нет котировок")
            return None
        closes = rates["close"][-rates_count:]
        times = rates["time"][-rates_count:]
        future_date = int(times[-1] + self.timeunit * self.predictor.datainfo.future)
        low, high, confidence = self.predictor.eval(closes)
        result = (
            times[-1],
            closes[-1],
            future_date,
            closes[-1] + low,
            closes[-1] + high,
            confidence,
        )
        logging.debug(str(result))
        return result

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

    def run(self):
        if not self.ready:
            logging.error("Робот не готов к торговле")
            return False
        dtimer = DelayTimer(self.delay)
        while True:
            if not dtimer.elapsed():
                continue
            self.deal()


def main():
    modelname = "models/19"
    logging.debug("Загрузка модели " + modelname)
    predictor = Predictor(modelname=modelname)
    if not predictor.trained:
        logging.error("Ошибка инициализации модели")
        return False
    return True
    adviser = Adviser(predictor=predictor)
    if not adviser.ready:
        logging.error("Ошибка инициализации робота")
        return
    adviser.run()


# -----------------------------------------------------------
if __name__ == "__main__":
    main()
