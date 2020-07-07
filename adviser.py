import MetaTrader5 as mt5
from timer import DelayTimer
import numpy as np
import pandas as pd
import datetime as dt
import logging
from predictor import Predictor

# ORDER_TYPE_SELL=-1
# ORDER_TYPE_BUY=1

version = "0.1"
author = "СИИ"


def __init_logger__():
    logging.basicConfig(
        # filename="srv.log",
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(message)s",
        # filemode="w",
    )
    return True


class Adviser:
    def __init__(self, predictor, delay=300, symbol="EURUSD"):
        self.mt5path = "D:/Dev/MT5/terminal64.exe"
        self.__init_mt5__()
        self.predictor = predictor
        self.sl = 1000
        self.tp = 1000
        self.max_vol = 1.0
        self.vol = 0.1
        self.confidence = 0.4
        self.std = 0.003
        self.symbol = symbol
        self.timeunit = 300  # секунд
        self.delay = delay  # секунд
        self.ready = True

    def __init_mt5__(self):
        if self.IsMT5Connected():
            return True
        logging.info("Подключение к терминалу MT5")
        # if not mt5.initialize(path="D:/Dev/Alpari MT5/terminal64.exe"):   # реальный
        if not mt5.initialize(path=self.mt5path):  # тестовый
            logging.error("Ошибка подключпения к терминалу MT5")
            mt5.shutdown()
            return False
        logging.info("... версия MT5:" + str(mt5.version()))
        return True

    def IsMT5Connected(self):
        info = mt5.account_info()
        if mt5.last_error()[0] < 0:
            return False
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
        return vol

    def get_trend(self):
        result = self.compute()
        if result is None:
            return (0, 0)
        cur_date, cur_price, future_date, low, high, confidence = result
        d = (low + high) / 2 - cur_price
        trend = (max(-1, min(1, d / self.std)), confidence)
        return trend

    def order(self, order_type, volume):
        if order_type == 1:
            result = mt5.Buy(symbol=self.symbol, volume=volume)
        elif order_type == -1:
            result = mt5.Sell(symbol=self.symbol, volume=volume)
        return result

    def compute(self):
        """
        Вычисление прогноза на текущую дату
        на выходе: (rdate, rclose, future_date, low, high, confidence)
        """
        rates_count = self.predictor.datainfo._in_size() + 1
        # TODO заменить mt5.TIMEFRAME_M5 на сохраненное в config.json значение
        rates = mt5.copy_rates_from_pos(self.symbol, mt5.TIMEFRAME_M5, 0, rates_count)
        if rates is None:
            logging.error("Ошибка запроса котировок: " + str(mt5.last_error()))
            return None
        if len(rates) < rates_count:
            logging.error("Нет котировок")
            return None
        closes = rates["close"][-rates_count:]
        times = rates["time"][-rates_count:]
        future_date = int(times[-1] + self.timeunit * self.predictor.datainfo.future)
        low, high, confidence = self.predictor.eval(closes)[0]
        result = (
            times[-1],
            closes[-1],
            future_date,
            closes[-1] + low,
            closes[-1] + high,
            confidence,
        )
        # logging.debug(str(result))
        return result

    def deal(self):
        trend = self.get_trend()
        targ_vol = self.max_vol * trend[0]
        pos_vol = self._get_pos_vol()
        d = targ_vol - pos_vol
        logging.debug("trend=" + str(trend) + " diff=" + str(d))
        if trend[1] < self.confidence:
            return
        if d == 0:
            return
        if d > self.vol and pos_vol < self.max_vol:
            if self.order(order_type=1, volume=self.vol):
                logging.info("Покупка " + str(self.vol))
            else:
                logging.error("Ошибка покупки: " + str(mt5.last_error()))
        if -d > self.vol and -pos_vol < self.max_vol:
            if self.order(order_type=-1, volume=self.vol):
                logging.info("Продажа " + self.vol)
            else:
                logging.error("Ошибка продажи: " + str(mt5.last_error()))

    def run(self):
        if not self.ready:
            logging.error("Робот не готов к торговле")
            return False
        logging.info("Запуск планировщика с периодом " + str(self.delay) + " сек.")
        dtimer = DelayTimer(self.delay)
        while True:
            if not dtimer.elapsed():
                continue
            if not self.IsMT5Connected():
                if not self.__init_mt5__():
                    continue
            self.deal()


def main():
    print("Adviser v=" + version + " author=" + author)
    __init_logger__()
    modelname = "models/19"
    logging.debug("Загрузка модели " + modelname)
    predictor = Predictor(modelname=modelname)
    if not predictor.trained:
        logging.error("Ошибка инициализации модели")
        return False
    adviser = Adviser(predictor=predictor, delay=60)
    if not adviser.ready:
        logging.error("Ошибка инициализации робота")
        return
    adviser.run()


# -----------------------------------------------------------
if __name__ == "__main__":
    main()
