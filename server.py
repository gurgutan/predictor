"""
Сервер для обновления БД прогнозов
"""
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
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
import tensorflow as tf

tf.get_logger().setLevel("INFO")

logger = logging.getLogger(__name__)


class Server(object):
    def __init__(self):
        configname = "config.json"
        self.version = 1.1
        self.p = None
        self.ready = False
        logger.info(f"Робот Аля v{self.version}, автор: Слеповичев Иван Иванович")
        with open(configname) as config_file:
            data = json.load(config_file)
            self.dbname = data["dbname"]  # полное имя БД
            self.initialdate = dt.datetime.fromisoformat(data["initialdate"])
            self.modelname = data["modelname"]  # полное имя модели (с путем)
            self.symbol = data["symbol"]  # символ инструмента
            self.timeunit = data["timeunit"]
            self.mt5path = data["mt5path"]
            self.compute_delay = data["compute_delay"]
            self.train_delay = data["train_delay"]  # задержка в секундах цикла сервера
            self.train_rates_count = data["train_rates_count"]
            self.input_width = data["input_width"]
            self.label_width = data["label_width"]
            self.sample_width = data["sample_width"]
            self.shift = data["shift"]
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
        self.p = Predictor(
            datafile=None,
            model=self.modelname,
            input_width=self.input_width,
            label_width=self.label_width,
            shift=self.shift,
        )
        return True

    def __get_rates_from_date__(self, from_date):
        # установим таймзону в UTC
        timezone = pytz.timezone("Etc/UTC")
        rates_count = 0
        while rates_count < self.input_width + 1:
            mt5rates = mt5.copy_rates_range(
                self.symbol, mt5.TIMEFRAME_H1, from_date, dt.datetime.now(tz=timezone)
            )
            if mt5rates is None:
                logger.error("Ошибка:" + str(mt5.last_error()))
                return None
            rates_count = len(mt5rates)
            if rates_count < self.input_width + 1:
                from_date = from_date - dt.timedelta(days=1)
        rates = pd.DataFrame(mt5rates)
        logger.debug("Получено " + str(len(rates)) + " котировок")
        return rates

    def __get_last_rates__(self, count):
        mt5rates = mt5.copy_rates_from_pos(self.symbol, mt5.TIMEFRAME_H1, 0, count)
        if mt5rates is None:
            logger.error("Ошибка:" + str(mt5.last_error()))
            return None
        rates = pd.DataFrame(
            mt5rates,
            columns=[
                "time",
                "open",
                "high",
                "low",
                "close",
                "tickvol",
                "spread",
                "real_volume",
            ],
        )
        # logging.debug("Получено " + str(len(rates)) + " котировок")
        return rates

    def compute(self, times, prices, verbose=0):
        assert len(times) != 0, f"Ошибка: пустой список котировок"
        # count = len(opens) - self.input_width
        results = []
        # вычисляем прогноз
        output_data = self.p.predict(prices, verbose=verbose)
        count = output_data.shape[0]
        # сформируем результирующий список кортежей для записи в БД
        for i in range(count):
            forecast = output_data[i].flatten()
            for j in range(forecast.shape[0]):
                rdate = int(times[i + self.input_width])
                rprice = prices[i + self.input_width]
                pdate = int(rdate + self.timeunit * (j + 1))  # секунды*shift
                price = float(forecast[j])
                confidence = 0
                db_row = (
                    rdate,
                    round(rprice, 8),
                    self.symbol,
                    self.modelname,
                    pdate,
                    round(rprice + price, 8),
                    round(rprice + price, 8),
                    round(rprice + price, 8),
                    round(confidence, 8),
                )
                results.append(db_row)
        return results

    def __compute_old__(self):
        from_date = self.initialdate
        # определяем "крайнюю" дату для последующих вычислений
        date = dbcommon.db_get_lowdate(self.db)
        delta = dt.timedelta(days=2)
        # delta = dt.timedelta(minutes=(self.p.datainfo._in_size() + 1) * 5)
        if not date is None:
            from_date = date - delta
        logger.info(f"Вычисление прошлых значений с даты {from_date}")
        rates = self.__get_rates_from_date__(from_date)
        if rates is None:
            logger.error("Отсутствуют новые котировки")
            return
        times, prices = rates["time"], rates["open"]
        results = self.compute(times, prices, verbose=1)
        if results is None:
            return
        # logging.info("Вычислено %d " % len(results))
        dbcommon.db_replace(self.db, results)

    def __is_mt5_connected__(self):
        info = mt5.account_info()
        if mt5.last_error()[0] < 0:
            return False
        return True

    def is_waiting_compute(self, dtimer):
        if not dtimer.elapsed():
            if dtimer.remained() > 1:
                sleep(1)
            return True
        else:
            return False

    def is_waiting_train(self, dtimer):
        return not dtimer.elapsed()

    def train(self) -> bool:
        df = self.__get_last_rates__(self.train_rates_count)
        if df is None:
            return False
        logger.info(f"Получено {len(df.index)} котировок")
        self.p.dataloader.load_df(
            df,
            input_column="open",
            train_ratio=1 - 1.0 / 8,
            val_ratio=1.0 / 8,
            test_ratio=0,
            verbose=1,
        )
        self.p.fit(
            batch_size=2 ** 14,
            epochs=8,
            use_tensorboard=False,
            use_early_stop=False,
            verbose=1,
            use_multiprocessing=True,
        )
        logger.info(f"Модель дообучена")
        self.p.save_model()
        logger.info("Модель сохранена")
        return True

    def is_mt5_ready(self):
        if not self.__is_mt5_connected__():
            logger.error("Ошибка подключения к МТ5:" + str(mt5.last_error()))
            if not self.__init_mt5__():
                return False
        return True

    def start(self):
        compute_timer = DelayTimer(self.compute_delay)
        train_timer = DelayTimer(self.train_delay, shift=600)
        self.__compute_old__()  # обновление данных начиная с даты
        logger.info(f"Запуск таймера с периодом {self.compute_delay}")
        while True:
            if self.is_waiting_compute(compute_timer):
                continue
            if not self.is_mt5_ready():
                continue

            sleep(2)  # задержка для получения последнего бара
            rates = self.__get_last_rates__(self.input_width + 1)
            if rates is None:
                logger.debug("Отсутствуют новые котировки")
                continue
            times, prices = rates["time"], rates["open"]
            results = self.compute(times, prices, verbose=0)
            if results is None or len(results) == 0:
                logger.error("Ошибка вычислений")
                continue
            # Запись в БД
            dbcommon.db_replace(self.db, results)
            # Вывод на экран
            (
                rdate,
                rprice,
                symbol,
                modelname,
                pdate,
                price,
                low,
                high,
                confidence,
            ) = results[-1]
            d = round((price - rprice), 5)
            logger.debug(f"delta={d}")

            if train_timer.elapsed():
                logger.debug(f"Дообучение...")
                self.train()


DEBUG = False
