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
# MODEL_PATH = "D:/Dev/python/predictor/models/19"
MODEL_PATH = "D:/Dev/python/predictor/TFLite/20.tflite"
SYMBOL = "EURUSD"
SL = 512
TP = 512
MAX_VOL = 1.0
VOL = 0.1
CONFIDENCE = 0.5
DELAY = 10
USE_TFLITE = True
# ---------------------------------------------------------


import MetaTrader5 as mt5
import numpy as np
import pandas as pd
from datetime import timedelta, datetime
import logging
from predictor import Predictor
from time import sleep
from timer import DelayTimer
from mt5common import send_order, is_trade_allowed, get_account_info


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
        self.vol = vol
        self.confidence = confidence
        self.std = round(self.predictor.datainfo.y_std, 8)
        self.symbol = symbol
        self.timeunit = self.predictor.datainfo.timeunit  # секунд
        self.delay = delay  # секунд
        self.ready = True

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
        vol = 0
        for pos in positions:
            type_mult = -(pos.type * 2 - 1)  # -1=sell, +1=buy
            vol += pos.volume * type_mult
        return vol

    # def order(self, order_type, volume):
    #     # TODO заменить Buy, Sell на order_send
    #     if order_type == 1:
    #         result = mt5.Buy(symbol=self.symbol, volume=volume)
    #     elif order_type == -1:
    #         result = mt5.Sell(symbol=self.symbol, volume=volume)
    #     return result

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
            return (0, 0)
        cur_date, cur_price, future_date, low, high, confidence = result
        d = (low + high) / 2 - cur_price
        # logging.debug(f"d={d}")
        # коэфициент учета ширины интервала прогноза
        interval_length_coef = self.predictor.datainfo._out_size() / 2
        trend = (
            max(-1, min(1, d / self.std / interval_length_coef)),
            round(confidence, 2),
            round(cur_price, 5),
        )
        return trend

    def deal(self):
        trend = self.get_trend()
        targ_vol = round(self.max_vol * trend[0], 2)
        pos_vol = self._get_pos_vol()
        if pos_vol == None:
            return
        d = round(targ_vol - pos_vol, 2)
        logger.debug(
            f"прогноз={round(trend[0],2)} цена={trend[2]} уверен={trend[1]} актив={pos_vol} цель={targ_vol} разность={str(d)}"
        )
        if trend[1] < self.confidence:
            return
        if d >= self.vol and pos_vol < self.max_vol:
            send_order(
                self.symbol,
                self.vol * sign(d),
                tp=self.tp,
                sl=self.sl,
                comment=f"{ROBOT_NAME} {round(self.confidence, 2)}",
            )
            # if self.order(order_type=1, volume=self.vol):
            #     logging.info("Покупка " + str(self.vol))
            # else:
            #     logging.error("Ошибка покупки: " + str(mt5.last_error()))
        elif -d >= self.vol and -pos_vol < self.max_vol:
            send_order(
                self.symbol,
                self.vol * sign(d),
                tp=self.tp,
                sl=self.sl,
                comment=f"{ROBOT_NAME} {round(self.confidence, 2)}",
            )
            # if self.order(order_type=-1, volume=self.vol):
            #     logging.info("Продажа " + str(self.vol))
            # else:
            #     logging.error("Ошибка продажи: " + str(mt5.last_error()))

    def run(self):
        if not self.ready:
            logger.error("Робот не готов к торговле")
            return False
        robot_info = f"Робот(SL={self.sl},TP={self.tp},max_vol={self.max_vol},vol={self.vol},confidence={self.confidence},std={self.std},symbol={self.symbol},timeunit={self.timeunit},delay={self.delay}"
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
        "=================================================================================="
    )
    logger.info(f"Робот {ROBOT_NAME} v{VERSION}, автор:{AUTHOR}")
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


# -----------------------------------------------------------
if __name__ == "__main__":
    main()
