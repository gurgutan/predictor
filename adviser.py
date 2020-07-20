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
MODEL_PATH = "D:/Dev/python/predictor/models/22"
# MODEL_PATH = "D:/Dev/python/predictor/TFLite/20.tflite"
SYMBOL = "EURUSD"
SL = 512
TP = 512
MAX_VOL = 1.0
VOL = 0.1  # 0.4 с 16.07.20 18:55
CONFIDENCE = 0.2
DELAY = 300
USE_TFLITE = False
REINVEST = 0.01
# ---------------------------------------------------------


import MetaTrader5 as mt5
import numpy as np
import pandas as pd
from datetime import timedelta, datetime
import logging
from predictor import Predictor
from time import sleep
from timer import DelayTimer
from mt5common import (
    send_order,
    is_trade_allowed,
    get_account_info,
    get_equity,
    is_mt5_connected,
    init_mt5,
)


logging.basicConfig(
    handlers=(
        logging.FileHandler("srv.log", encoding="utf-8", mode="a"),
        logging.StreamHandler(),
    ),
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)-6s | %(message)s",
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
        self.predictor = predictor
        self.sl = sl
        self.tp = tp
        self.max_vol = max_vol
        self.min_vol = vol
        self.confidence = confidence
        self.std = round(self.predictor.datainfo.y_std, 8)
        self.symbol = symbol
        self.delay = delay  # секунд
        self.ready = True
        self.__try_order_count = 3  # количество повторов order_send в случае неудачи

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

    def get_last_rates(self, count):
        mt5rates = mt5.copy_rates_from_pos(self.symbol, mt5.TIMEFRAME_M5, 0, count)
        if mt5rates is None:
            logging.error("Ошибка:" + str(mt5.last_error()))
            return None
        rates = pd.DataFrame(mt5rates)
        # logging.debug("Получено " + str(len(rates)) + " котировок")
        return rates

    def compute(self):
        """
        Вычисление прогноза на текущую дату
        на выходе: (rdate, rclose, future_date, low, high, confidence)
        """
        rates_count = self.predictor.datainfo._in_size() + 1
        rates = self.get_last_rates(rates_count)
        # rates = mt5.copy_rates_from_pos(self.symbol, mt5.TIMEFRAME_M5, 0, rates_count)
        if rates is None:
            logger.error("Ошибка запроса котировок: " + str(mt5.last_error()))
            return None
        if len(rates) < rates_count:
            logger.error("Нет котировок")
            return None
        closes = rates["close"].tolist()[-rates_count:]
        times = rates["time"].tolist()[-rates_count:]
        future_date = int(
            times[-1]
            + self.predictor.datainfo.timeunit * self.predictor.datainfo.future
        )
        low, high, confidence = self.predictor.eval(closes)
        result = (
            times[-1],
            round(closes[-1], 8),
            future_date,
            round(closes[-1] + low, 8),
            round(closes[-1] + high, 8),
            round(confidence, 8),
        )
        return result

    def get_trend(self):
        result = self.compute()
        if result is None:
            return (0, 0)
        cur_date, cur_price, future_date, low, high, confidence = result
        d = (low + high) / 2.0 - cur_price
        # logging.debug(f"d={d}")
        # коэфициент учета ширины интервала прогноза
        interval_length_coef = float(self.predictor.datainfo._out_size() / 2.0)
        trend = (
            round(max(-1.0, min(1.0, d / self.std / interval_length_coef)), 4),
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
        targ_vol = round(self.max_vol * round(trend[0], 2) * reinvest_k, 2)
        pos_vol = self._get_pos_vol()
        # защита от открытия ордера при неизвестном объеме позиции
        if pos_vol == None:
            return
        d = round(targ_vol - pos_vol, 2)
        lot = self.min_vol * sign(d)
        logger.debug(
            f"прогноз={trend[0]} цена={trend[2]} уверен={trend[1]} актив={pos_vol} цель={targ_vol} разность={d}"
        )
        if trend[1] < self.confidence:
            return
        if abs(d) >= self.min_vol and abs(pos_vol) < self.max_vol * reinvest_k:
            i = 0
            while (
                not send_order(
                    self.symbol,
                    round(lot, 2),
                    tp=self.tp,
                    sl=self.sl,
                    comment=f"{ROBOT_NAME} {round(self.confidence, 2)}",
                )
                and i < self.__try_order_count
            ):
                sleep(1)
                i += 1

    def run(self):
        if not self.ready:
            logger.error("Робот не готов к торговле")
            return False
        robot_info = (
            f"Робот(\n  SL={self.sl},\n  TP={self.tp},\n  max_vol={self.max_vol},\n"
            + f"  vol={self.min_vol},\n  confidence={self.confidence},\n  std={self.std},\n"
            + f"  symbol={self.symbol},\n  timeunit={self.predictor.datainfo.timeunit},\n"
            + f"  delay={self.delay})"
        )
        logging.info(robot_info)
        dtimer = DelayTimer(self.delay)
        while True:
            if not dtimer.elapsed():
                remained = dtimer.remained()
                if remained > 1:
                    sleep(remained - 0.01)  # ожидаем на 10 мс меньше, чем осталось
                continue
            if not is_mt5_connected():
                logger.error("Потеряно подключение к MetaTrader5")
                if not init_mt5(self.mt5path)():
                    continue
            if not is_trade_allowed():
                logger.info("Торговля не разрешена или отключен алготрейдинг")
            self.deal()


def main():
    logger.info("===============================================================")
    logger.info(f"Робот {ROBOT_NAME} v{VERSION}, автор: {AUTHOR}")
    predictor = Predictor(modelname=MODEL_PATH)
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
