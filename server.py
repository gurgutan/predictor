"""
Сервер для обновления БД прогнозов
"""
import dbcommon
from predictor import Predictor
import json
import sqlite3
import datetime as dt
import pandas as pd
from time import sleep
from timer import DelayTimer
import logging
import MetaTrader5 as mt5
import os
import pytz

logger = logging.getLogger(__name__)


class Server(object):
    def __init__(self):
        configname = "config.json"
        self.version = 1.02
        self.p = None
        self.ready = False
        logger.info(f"Робот Аля v{self.version}, автор: Слеповичев Иван Иванович")
        with open(configname) as config_file:
            data = json.load(config_file)
            self.dbname = data["dbname"]  # полное имя БД
            self.initialdate = dt.datetime.fromisoformat(data["initialdate"])
            self.modelname = data["modelname"]  # полное имя модели (с путем)
            self.symbol = data["symbol"]  # символ инструмента
            self.mt5path = data["mt5path"]
            # self.timeframe = int(data["timeframe"])  # тайм-фрэйм в секундах
            self.delay = data["delay"]  # задержка в секундах цикла сервера
        if self.__init_db__() and self.__init_mt5__() and self.__init_predictor__():
            self.ready = True
        else:
            logger.error("Ошибка инициализации сервера")
            self.ready = False

    def is_tflite(self):
        """
        Возвращает True, если используется модель tflite, иначе False
        Определяется по расширению имени модели
        """
        return self.p.is_tflite()

    def __init_db__(self):
        logger.info("Открытие БД " + self.dbname)
        self.db = dbcommon.db_open(self.dbname)
        if self.db is None:
            logger.error("Ошибка открытия БД '%s':" % self.dbname)
            return False
        return True

    def __init_mt5__(self):
        # подключимся к MetaTrader 5
        if not mt5.initialize(path=self.mt5path):
            logger.error("Ошибка подключпения к терминалу MT5")
            mt5.shutdown()
            return False
        logger.info("Подключение к терминалу MT5, версия:" + str(mt5.version()))
        return True

    def __init_predictor__(self):
        logger.info("Загрузка и инициализация модели '%s'" % self.modelname)
        self.p = Predictor(modelname=self.modelname)
        if not self.p.trained:
            logger.error("Ошибка инициализации модели '%s'" % self.modelname)
            return False
        return True

    def __get_rates_from_date__(self, from_date):
        # установим таймзону в UTC
        timezone = pytz.timezone("Etc/UTC")
        rates_count = 0
        while rates_count < self.p.datainfo._in_size() + 1:
            mt5rates = mt5.copy_rates_range(
                self.symbol, mt5.TIMEFRAME_H1, from_date, dt.datetime.now(tz=timezone)
            )
            if mt5rates is None:
                logger.error("Ошибка:" + str(mt5.last_error()))
                return None
            rates_count = len(mt5rates)
            if rates_count < self.p.datainfo._in_size() + 1:
                from_date = from_date - dt.timedelta(days=1)
        rates = pd.DataFrame(mt5rates)
        logger.debug("Получено " + str(len(rates)) + " котировок")
        return rates

    def __get_last_rates__(self, count):
        mt5rates = mt5.copy_rates_from_pos(self.symbol, mt5.TIMEFRAME_H1, 0, count)
        if mt5rates is None:
            logger.error("Ошибка:" + str(mt5.last_error()))
            return None
        rates = pd.DataFrame(mt5rates)
        # logging.debug("Получено " + str(len(rates)) + " котировок")
        return rates

    def compute(self, rates, verbose=0):
        if len(rates) == 0:
            logger.error(f"Ошибка: пустой список котировок")
            return None
        opens, times = rates["open"], rates["time"]
        input_size = self.p.datainfo.input_shape[0]
        count = len(opens) - input_size
        results = []

        # вычисляем прогноз
        if self.is_tflite():
            output_data = self.p.tflite_predict(opens, verbose=verbose)
        else:
            output_data = self.p.predict(opens, verbose=verbose)
        if output_data is None:
            logger.error(f"Ошибка: не удалось получить прогноз для {times[-1]}")
            return None
        # сформируем результирующий список кортежей для записи в БД
        for i in range(count):
            plow, phigh, confidence, center = output_data[i]
            rdate = int(times[i + input_size])
            rprice = opens[i + input_size]
            pdate = int(
                rdate + self.p.datainfo.timeunit * self.p.datainfo.future
            )  # секунды*M5*future
            db_row = (
                rdate,
                round(rprice, 8),
                self.symbol,
                self.p.name,
                pdate,
                round(rprice + plow, 8),
                round(rprice + phigh, 8),
                round(confidence, 8),
                round(center, 8),
            )
            results.append(db_row)
        return results

    def __compute_old__(self):
        from_date = self.initialdate
        # определяем "крайнюю" дату для последующих вычислений
        date = dbcommon.db_get_lowdate(self.db)
        delta = dt.timedelta(days=2)  # за 4 дня до
        # delta = dt.timedelta(minutes=(self.p.datainfo._in_size() + 1) * 5)
        if not date is None:
            from_date = date - delta
        logger.info(f"Вычисление прошлых значений с даты {from_date}")
        rates = self.__get_rates_from_date__(from_date)
        if rates is None:
            logger.error("Отсутствуют новые котировки")
            return
        results = self.compute(rates, verbose=1)
        if results is None:
            return
        # logging.info("Вычислено %d " % len(results))
        dbcommon.db_replace(self.db, results)

    def __is_mt5_connected__(self):
        info = mt5.account_info()
        if mt5.last_error()[0] < 0:
            return False
        return True

    def is_waiting(self, dtimer):
        if not dtimer.elapsed():
            remained = dtimer.remained()
            if remained > 1:
                sleep(1)
            return True
        else:
            return False

    def is_mt5_ready(self):
        if not self.__is_mt5_connected__():
            logger.error("Ошибка подклбчения к МТ5:" + str(mt5.last_error()))
            if not self.__init_mt5__():
                return False
        return True

    def start(self):
        dtimer = DelayTimer(self.delay)
        self.__compute_old__()  # обновление данных начиная с даты
        logger.info(f"Запуск таймера с периодом {self.delay}")
        while True:
            if self.is_waiting(dtimer):
                continue

            if not self.is_mt5_ready():
                continue
            sleep(2)  # задержка для получения последнего бара
            rates = self.__get_last_rates__(
                self.p.datainfo.input_shape[0] + self.p.datainfo.input_shape[1]
            )
            if rates is None:
                logger.debug("Отсутствуют новые котировки")
                continue

            results = self.compute(rates, verbose=0)
            if results is None or len(results) == 0:
                logger.error("Ошибка вычислений")
                continue

            # Запись в БД
            dbcommon.db_replace(self.db, results)

            # Вывод на экран
            rdate, rprice, _, _, future, low, high, conf, center = results[-1]
            d = round(((low + high) / 2.0 - rprice) / self.p.datainfo.y_std / 4, 5)
            logger.debug(
                f"delta={round(d,2)} center={round(center, 2)} confidence={round(conf,2)}"
            )


DEBUG = False
