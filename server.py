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


class Server(object):
    def __init__(self):
        configname = "config.json"
        self.p = None
        self.ready = False
        self.__init_logger__()
        with open(configname) as config_file:
            data = json.load(config_file)
            self.dbname = data["dbname"]  # полное имя БД
            self.initialdate = dt.datetime.fromisoformat(data["initialdate"])
            self.modelname = data["modelname"]  # полное имя модели (с путем)
            self.symbol = data["symbol"]  # символ инструмента
            self.timeframe = int(data["timeframe"])  # тайм-фрэйм в секундах
            self.delay = data["delay"]  # задержка в секундах цикла сервера
        if self.__init_db__() and self.__init_mt5__() and self.__init_predictor__():
            self.ready = True
        else:
            logging.error("Ошибка инициализации сервера")
            self.ready = False

    def is_tflite(self):
        """
        Возвращает True, если используется модель tflite, иначе False
        Определяется по расширению имени модели
        """
        return self.p.is_tflite()

    def __init_db__(self):
        logging.info("Открытие БД " + self.dbname)
        self.db = dbcommon.db_open(self.dbname)
        if self.db is None:
            logging.error("Ошибка открытия БД '%s':" % self.dbname)
            return False
        return True

    def __init_mt5__(self):
        # подключимся к MetaTrader 5
        if not mt5.initialize(path="D:/Dev/Alpari MT5/terminal64.exe"):
            logging.error("Ошибка подключпения к терминалу MT5")
            mt5.shutdown()
            return False
        logging.info("Подключение к терминалу MT5, версия:" + str(mt5.version()))
        return True

    def __init_logger__(self):
        logging.basicConfig(
            level=logging.DEBUG, format="%(asctime)s %(levelname)-6s :: %(message)s",
        )
        return True

    def __init_predictor__(self):
        logging.info("Загрузка и инициализация модели '%s'" % self.modelname)
        self.p = Predictor(modelname=self.modelname)
        if not self.p.trained:
            logging.error("Ошибка инициализации модели '%s'" % self.modelname)
            return False
        return True

    def get_rates_from_date(self, from_date):
        # установим таймзону в UTC
        timezone = pytz.timezone("Etc/UTC")
        rates_count = 0
        while rates_count < self.p.datainfo._in_size() + 1:
            mt5rates = mt5.copy_rates_range(
                self.symbol, mt5.TIMEFRAME_M5, from_date, dt.datetime.now(tz=timezone)
            )
            if mt5rates is None:
                logging.error("Ошибка:" + str(mt5.last_error()))
                return None
            rates_count = len(mt5rates)
            if rates_count < self.p.datainfo._in_size() + 1:
                from_date = from_date - dt.timedelta(days=1)
        rates = pd.DataFrame(mt5rates)
        logging.debug("Получено " + str(len(rates)) + " котировок")
        return rates

    def get_last_rates(self, count):
        mt5rates = mt5.copy_rates_from_pos(self.symbol, mt5.TIMEFRAME_M5, 0, count)
        if mt5rates is None:
            logging.error("Ошибка:" + str(mt5.last_error()))
            return None
        rates = pd.DataFrame(mt5rates)
        # logging.debug("Получено " + str(len(rates)) + " котировок")
        return rates

    def compute(self, rates, verbose=0):
        if len(rates) == 0:
            loggin.error(f"Ошибка: пустой список котировок")
            return None
        closes, times = rates["close"], rates["time"]
        count = len(closes)
        shift = self.p.datainfo._in_size() + 1
        results = []
        # сделаем "нарезку" входных векторов
        input_data = []
        for i in range(shift, count + 1):
            x = closes[i - shift : i].to_numpy()
            input_data.append(x)
        # вычисляем прогноз
        if self.is_tflite():
            output_data = self.p.tflite_predict(input_data, verbose=verbose)
        else:
            output_data = self.p.predict(input_data, verbose=verbose)
        if output_data is None:
            logging.error(f"Ошибка: не удалось получить прогноз для {times[-1]}")
            return None
        # сформируем результирующий список кортежей для записи в БД
        for i in range(shift, count + 1):
            plow, phigh, confidence = output_data[i - shift]
            rdate = int(times[i - 1])
            rprice = closes[i - 1]
            pdate = int(rdate + 60 * 5 * self.p.datainfo.future)  # секунды*M5*future
            db_row = (
                rdate,
                round(rprice, 6),
                self.symbol,
                self.p.name,
                pdate,
                round(rprice + plow, 6),
                round(rprice + phigh, 6),
                round(confidence, 6),
            )
            results.append(db_row)
        d = round((results[-1][5] + results[-1][6]) / 2.0 - results[-1][1], 4)
        logging.debug(f"{results[-1]} d={d}")
        return results

    def calc_old(self):
        from_date = self.initialdate
        # определяем "крайнюю" дату для последующих вычислений
        date = dbcommon.db_get_lowdate(self.db)
        delta = dt.timedelta(days=4)  # за 4 дня до
        # delta = dt.timedelta(minutes=(self.p.datainfo._in_size() + 1) * 5)
        if not date is None:
            from_date = date - delta
        logging.info(f"Вычисление прошлых значений с даты {from_date}")
        rates = self.get_rates_from_date(from_date)
        if rates is None:
            logging.error("Отсутствуют новые котировки")
            return
        results = self.compute(rates, verbose=1)
        if results is None:
            return
        # logging.info("Вычислено %d " % len(results))
        dbcommon.db_replace(self.db, results)

    def IsMT5Connected(self):
        info = mt5.account_info()
        if mt5.last_error()[0] < 0:
            return False
        return True

    def start(self):
        dtimer = DelayTimer(self.delay)
        self.calc_old()  # обновление данных начиная с даты
        logging.info(f"Запуск таймера с периодом {self.delay}")
        while True:
            if not dtimer.elapsed():
                remained = dtimer.remained()
                if remained > 1:
                    sleep(1)
                continue
            if not self.IsMT5Connected():
                logging.error("Ошибка подклбчения к МТ5:" + str(mt5.last_error()))
                if not self.__init_mt5__():
                    continue
            rates = self.get_last_rates(self.p.datainfo._in_size() + 1)
            if rates is None:
                logging.debug("Отсутствуют новые котировки")
                continue
            results = self.compute(rates, verbose=0)
            if results is None:
                logging.error("Ошибка вычислений")
                continue
            dbcommon.db_replace(self.db, results)
            # logging.info("В БД записано " + str(len(results)) + " строк")


DEBUG = False
