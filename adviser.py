# ---------------------------------------------------------
# Слеповичев И.И. 12.07.2020
# Исполняемый скрипт торгового робота
# Для работы необходимо:
#   - наличие установленного Python 3.7 и выше
#   - наличие установленного MetaTrader5
#   - модель в соответствующей папке
#   - tensorflow >=2.0.0
#   - numpy
#   - pandas
#   - MetaTrader5
#   - pandas
#   - predictor.py
#   - datainfo.py
#   - patterns.py
#   - модель .pb
# ---------------------------------------------------------

# Общие константы
ROBOT_NAME = "Аля"
VERSION = "0.1"
AUTHOR = "Слеповичев Иван Иванович"
MT5_PATH = "D:/Dev/Alpari MT5/terminal64.exe"
MODEL_PATH = "D:/Dev/python/predictor/models/19"
# MODEL_PATH = "D:/Dev/python/predictor/TFLite/20.tflite"
SYMBOL = "EURUSD"
SL = 512
TP = 512
MAX_VOL = 1.0
VOL = 0.2
CONFIDENCE = 0.1
DELAY = 300
USE_TFLITE = False
REINVEST = 0.1
# ---------------------------------------------------------


import MetaTrader5 as mt5
import numpy as np
import pandas as pd
from datetime import timedelta, datetime
import logging
from predictor import Predictor
from time import sleep
from timer import DelayTimer
from mt5common import send_order, is_trade_allowed, get_account_info, get_equity


logging.basicConfig(
    handlers=(
        logging.FileHandler("srv.log", encoding="utf-8", mode="a"),
        logging.StreamHandler(),
    ),
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)-6s:: %(message)s",
    datefmt="%d.%m.%Y %H:%M:%S",
)
logger = logging.getLogger(__name__)


def sign(x):
    return 1 if x >= 0 else -1


class Adviser:
    def __init__(
        self,
        predictor,
        delay=300,
        symbol="EURUSD",
        mt5path="D:/Dev/Alpari MT5/terminal64.exe",
        confidence=0.2,
        sl=512,
        tp=512,
        max_vol=1.0,
        vol=0.1,
    ):
        self.mt5path = mt5path
        self.__init_mt5__()
        self.predictor = predictor
        self.sl = sl
        self.tp = tp
        self.max_vol = max_vol
        self.min_vol = vol
        self.confidence = confidence
        self.std = round(self.predictor.datainfo.y_std, 8)
        self.symbol = symbol
        self.timeunit = self.predictor.datainfo.timeunit  # секунд
        self.delay = delay  # секунд
        self.ready = True
        self.try_order_count = 3  # количество повторов order_send в случае неудачи

    def __init_mt5__(self):
        if not mt5.initialize(path=self.mt5path):  # тестовый
            logger.error("Ошибка подключения к терминалу MT5")
            mt5.shutdown()
            return False
        logger.info(
            "Подключено к терминалу '%s' версия %s" % (self.mt5path, str(mt5.version()))
        )
        logger.info(get_account_info())
        return True

    def IsMT5Connected(self):
        info = mt5.account_info()
        if mt5.last_error()[0] < 0:
            return False
        return True

    def _get_pos_vol(self):
        positions = mt5.positions_get(symbol=self.symbol)
        if positions == None:
            logger.error(f"Ошибка запроса позиций: {mt5.last_error()}")
            return None
        # df=pd.DataFrame(list(positions),columns=usd_positions[0]._asdict().keys())
        # ticket time type magic identifier reason volume price_open sl tp price_current swap profit symbol comment
        vol = 0.0
        for pos in positions:
            type_mult = -(pos.type * 2.0 - 1.0)  # -1=sell, +1=buy
            vol += pos.volume * type_mult
        return vol

    def order(self, order_type, volume):
        # TODO заменить Buy, Sell на order_send
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
        rates = mt5.copy_rates_from_pos(self.symbol, mt5.TIMEFRAME_M5, 0, rates_count)
        if rates is None:
            logger.error("Ошибка запроса котировок: " + str(mt5.last_error()))
            return None
        if len(rates) < rates_count:
            logger.error("Нет котировок")
            return None
        closes = rates["close"][-rates_count:]
        times = rates["time"][-rates_count:]
        future_date = int(times[-1] + self.timeunit * self.predictor.datainfo.future)
        low, high, confidence = self.predictor.eval(closes)
        result = (
            times[-1],
            round(closes[-1], 6),
            future_date,
            round(closes[-1] + low, 6),
            round(closes[-1] + high, 6),
            round(confidence, 6),
        )
        return result

    def get_trend(self):
        result = self.compute()
        if result is None:
            return (0.0, 0.0)
        cur_date, cur_price, future_date, low, high, confidence = result
        d = (low + high) / 2.0 - cur_price
        # logging.debug(f"d={d}")
        # коэфициент учета ширины интервала прогноза
        interval_length_coef = float(self.predictor.datainfo._out_size() / 2.0)
        trend = (
            round(max(-1.0, min(1.0, d / self.std / interval_length_coef)), 6),
            round(confidence, 6),
            round(cur_price, 6),
        )
        return trend

    def deal(self):
        equity = get_equity()
        if equity is None:
            equity = 0
        reinvest_k = 1.0 + REINVEST * equity / 10000.0  # для RUB
        trend = self.get_trend()
        targ_vol = self.max_vol * round(trend[0], 2) * reinvest_k
        lot = self.min_vol * reinvest_k
        pos_vol = self._get_pos_vol()
        # защита от открытия ордера при неизвестном объеме позиции
        if pos_vol == None:
            return
        d = round(targ_vol - pos_vol, 4)
        logger.debug(
            f"прогноз={trend[0]} цена={trend[2]} уверен={trend[1]} актив={pos_vol} цель={targ_vol} разность={d}"
        )
        if trend[1] < self.confidence:
            return
        if (d >= self.min_vol and pos_vol < self.max_vol * reinvest_k) or (
            -d >= self.min_vol and -pos_vol < self.max_vol * reinvest_k
        ):
            i = 0
            while (
                not send_order(
                    self.symbol,
                    round(self.min_vol * sign(d), 2),
                    tp=self.tp,
                    sl=self.sl,
                    comment=f"{ROBOT_NAME} {round(self.confidence, 2)}",
                )
                and i < self.try_order_count
            ):
                sleep(1)
                i += 1
        # if (d >= self.min_vol and pos_vol < self.max_vol * reinvest_k)
        # if self.order(order_type=1, volume=self.min_vol):
        #     logging.info("Покупка " + str(self.min_vol))
        # else:
        #     logging.error("Ошибка покупки: " + str(mt5.last_error()))
        # elif(-d >= self.min_vol and -pos_vol < self.max_vol * reinvest_k)
        # if self.order(order_type=-1, volume=self.min_vol):
        #     logging.info("Продажа " + str(self.min_vol))
        # else:
        #     logging.error("Ошибка продажи: " + str(mt5.last_error()))

    def run(self):
        if not self.ready:
            logger.error("Робот не готов к торговле")
            return False
        robot_info = f"Робот(SL={self.sl},TP={self.tp},max_vol={self.max_vol},vol={self.min_vol},confidence={self.confidence},std={self.std},symbol={self.symbol},timeunit={self.timeunit},delay={self.delay}"
        logging.info(robot_info)
        dtimer = DelayTimer(self.delay)
        while True:
            if not dtimer.elapsed():
                remained = dtimer.remained()
                if remained > 1:
                    sleep(1)
                continue
            if not self.IsMT5Connected():
                logger.error("Потеряно подключение к MetaTrader5")
                if not self.__init_mt5__():
                    continue
            if not is_trade_allowed():
                logger.info("Торговля не разрешена или отключен алготрейдинг")
            self.deal()


def main():
    logger.info(
        "==============================================================================="
    )
    logger.info(f"Робот {ROBOT_NAME} v{VERSION}, автор: {AUTHOR}")
    predictor = Predictor(modelname=MODEL_PATH, use_tflite=USE_TFLITE)
    if not predictor.trained:
        logger.error(f"Ошибка загрузки модели {MODEL_PATH}")
        return False
    else:
        logger.debug(f"Модель {MODEL_PATH} загружена")
    adviser = Adviser(
        predictor=predictor,
        delay=DELAY,
        mt5path=MT5_PATH,
        symbol=SYMBOL,
        confidence=CONFIDENCE,
        tp=TP,
        sl=SL,
        max_vol=MAX_VOL,
        vol=VOL,
    )
    if not adviser.ready:
        logger.error("Ошибка инициализации робота")
        return
    adviser.run()


# TODO 1. Сравнить сети 19, 20 и 20.tflite
# TODO 2. Определять tflite модели по расширению
# -----------------------------------------------------------
if __name__ == "__main__":
    main()
