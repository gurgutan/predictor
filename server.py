"""
Сервер для обновления БД прогнозов
"""

from tensorflow.keras import backend as K
import tensorflow as tf
import pytz
import MetaTrader5 as mt5
import logging
from timer import DelayTimer
from time import sleep
import pandas as pd
import datetime as dt
import json
from predictor import Predictor
import dbcommon
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


tf.get_logger().setLevel("INFO")

logger = logging.getLogger(__name__)


class Server(object):
    def __init__(self):
        configname = "config.json"
        self.version = 1.2
        # self.p = None
        self.ready = False
        logger.info(f"Робот Аля v{self.version}, автор: Слеповичев И.И.")
        with open(configname) as config_file:
            data = json.load(config_file)
            self.dbname = data["dbname"]  # полное имс БД
            self.initialdate = dt.datetime.fromisoformat(data["initialdate"])
            self.modelname = data["modelname"]  # полное имс модели (с путем)
            self.symbol = data["symbol"]  # символ инструмента
            self.timeunit = data["timeunit"]
            self.mt5path = data["mt5path"]
            self.compute_delay = data["compute_delay"]
            # задержка в секундах цикла сервера
            self.train_delay = data["train_delay"]
            self.train_rates_count = data["train_rates_count"]
            self.input_width = data["input_width"]
            self.label_width = data["label_width"]
            self.sample_width = data["sample_width"]
            self.shift = data["shift"]
        if self.init_db() and self.init_mt5() and self.init_predictor():
            self.ready = True
        else:
            logger.error("Ошибка инициализации сервера")
            self.ready = False

    def init_db(self):
        logger.info("Открытие БД " + self.dbname)
        self.db = dbcommon.db_open(self.dbname)
        if self.db is None:
            logger.error("Ошибка открытис БД '%s':" % self.dbname)
            return False
        return True

    def init_mt5(self):
        # подключимсс к MetaTrader 5
        if not mt5.initialize(path=self.mt5path):  # type: ignore
            logger.error("Ошибка подключения к терминалу MT5")
            mt5.shutdown()  # type: ignore
            return False
        logger.info("Подключение к MT5, версия:" + str(mt5.version()))
        return True

    def init_predictor(self):
        logger.info("Загрузка и инициализация модели '%s'" % self.modelname)
        self.p = Predictor(
            datafile=None,
            model=self.modelname,
            input_width=self.input_width,
            label_width=self.label_width,
            shift=self.shift,
            batch_size=4096,
        )
        return True

    def get_rates_from_date(self, from_date):
        # установим таймзону в UTC
        timezone = pytz.timezone("Etc/UTC")
        rates_count = 0
        while rates_count < self.input_width + 1:
            mt5rates = mt5.copy_rates_range(
                self.symbol, mt5.TIMEFRAME_H1, from_date, dt.datetime.now(
                    tz=timezone)
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

    def get_rates(self, count, start_pos=0):
        if not self.is_mt5_ready():
            return None
        mt5rates = mt5.copy_rates_from_pos(
            self.symbol, mt5.TIMEFRAME_H1, start_pos, count
        )
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

    def compute(self, times: list, prices: list, verbose=0):
        assert len(times) != 0, f"Ошибка: пустой список котировок"
        assert len(prices) != 0, f"Ошибка: пустой список котировок"

        # count = len(opens) - self.input_width
        results = []
        # вычислсем прогноз
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

    def mean(self, a: list) -> float:
        return sum(a) / len(a)

    def compute_old(self):
        from_date = self.initialdate
        # определяем "крайнюю" дату для последующих вычислений
        date = dbcommon.db_get_lowdate(self.db)
        delta = dt.timedelta(days=2)
        # delta = dt.timedelta(minutes=(self.p.datainfo._in_size() + 1) * 5)
        if date is not None:
            from_date = date - delta
        logger.info(f"Вычисление прошлых значений с даты {from_date}")
        rates = self.get_rates_from_date(from_date)
        if rates is None:
            logger.error("Отсутствуют новые котировки")
            return
        times = rates["time"].to_list()
        prices = rates["open"].to_list()
        results = self.compute(times, prices, verbose=1)
        if results is None:
            return
        # logging.info("Вычислено %d " % len(results))
        dbcommon.db_replace(self.db, results)

    def is_mt5_connected(self):
        info = mt5.account_info()
        if info is None or mt5.last_error()[0] < 0:
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
        if not dtimer.elapsed():
            return True
        return False

    def is_mt5_ready(self):
        if not self.is_mt5_connected():
            logger.error("Ошибка подключения к МТ5:" + str(mt5.last_error()))
            if not self.init_mt5():
                return False
        return True

    def train(self, epochs=8, lr=1e-4, batch_size=2**10) -> bool:
        self.p.dataloader.batch_size = batch_size
        df = self.get_rates(self.train_rates_count, start_pos=0)
        if df is None:
            return False
        logger.info(f"Получено {len(df.index)} котировок")
        self.p.dataloader.load_df(
            df,
            input_column="open",
            train_ratio=1 - 1.0 / 4,
            val_ratio=1.0 / 4,
            test_ratio=0,
            verbose=1,
        )
        K.set_value(self.p.model.optimizer.learning_rate, lr)
        self.p.fit(
            batch_size=2**16,
            epochs=epochs,
            use_tensorboard=False,
            use_early_stop=False,
            use_checkpoints=True,
            verbose=1,
            use_multiprocessing=True,
        )
        logger.info(f"Модель дообучена")
        self.p.save_model()
        logger.info("Модель сохранена")
        return True

    def predict(self):
        rates = None
        results = None
        if self.is_mt5_ready():
            rates = self.get_rates(self.input_width + 1)
        else:
            logger.debug("Отсутствуют подключение к MT5")
            return
        if rates is None:
            logger.debug("Отсутствуют новые котировки")
            return
        times = rates["time"].to_list()
        prices = rates["open"].to_list()
        # усредняем последнюю цену
        # prices[-1] = (prices[-2] + prices[-1]) / 2.0
        results = self.compute(times, prices, verbose=0)
        if results is None or len(results) == 0:
            logger.error("Ошибка вычислений")
            return
        # Запись в БД
        dbcommon.db_replace(self.db, results)
        # Вывод на экран
        _, rprice, _, _, _, price, _, _, _ = results[-1]
        d = round((price - rprice), 5)
        logger.debug(f"delta={d}")

    def start(self):
        # количество эпох обучение в каждом цикле планировщика
        epochs = 384
        lr = 1e-3
        batch_size = 2**15
        # Pretrain
        self.train(epochs=epochs*8, lr=lr, batch_size=batch_size)
        predict_timer = DelayTimer(self.compute_delay, shift=30)
        train_timer = DelayTimer(self.train_delay, shift=5*60)
        # вычисление прогнозов за "прошедший период"
        self.compute_old()
        logger.info(f"Запуск таймера с периодом {self.compute_delay}")
        while True:
            if predict_timer.elapsed():
                self.predict()
            if train_timer.elapsed():
                logger.debug(f"Дообучение...")
                self.train(epochs=epochs, lr=lr, batch_size=batch_size)
            sleep(1)


# ====================================================
DEBUG = False
