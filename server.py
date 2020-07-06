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


class Server(object):
    def __init__(self):
        configname = "config.json"
        self.p = None
        self.ready = False
        with open(configname) as config_file:
            data = json.load(config_file)
            self.dbname = data["dbname"]  # полное имя БД
            self.initialdate = dt.datetime.fromisoformat(data["initialdate"])
            self.modelname = data["modelname"]  # полное имя модели (с путем)
            self.symbol = data["symbol"]  # символ инструмента
            self.timeframe = data["timeframe"]  # тайм-фрэйм сервера
            self.delay = data["delay"]  # задержка в секундах цикла сервера
        if (
            self.__init_db__()
            and self.__init_mt5__()
            and self.__init_predictor__()
            and self.__init_logger__()
        ):
            self.ready = True
        else:
            logging.error("Ошибка инициализации сервера")
            self.ready = False

    def __init_db__(self):
        logging.info("Открытие БД " + self.dbname)
        self.db = dbcommon.db_open(self.dbname)
        if self.db is None:
            logging.error("Ошибка открытия БД '%s':" % self.dbname)
            return False
        return True

    def __init_mt5__(self):
        # подключимся к MetaTrader 5
        logging.info("Подключение к терминалу MT5 ")
        if not mt5.initialize():
            logging.error("Ошибка подключпения к терминалу MT5")
            mt5.shutdown()
            return False
        logging.info("... версия MT5:" + str(mt5.version()))
        return True

    def __init_logger__(self):
        logging.basicConfig(
            # filename="srv.log",
            level=logging.DEBUG,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            # filemode="w",
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
        mt5rates = mt5.copy_rates_range(
            self.symbol, mt5.TIMEFRAME_M5, from_date, dt.datetime.now()
        )
        if mt5rates is None:
            logging.error("Ошибка:" + str(mt5.last_error()))
            return None
        rates = pd.DataFrame(mt5rates)
        logging.info("Получено " + str(len(rates)) + " котировок")
        return rates

    def get_rates(self, count):
        mt5rates = mt5.copy_rates_from_pos(self.symbol, mt5.TIMEFRAME_M5, 0, count)
        if mt5rates is None:
            logging.error("Ошибка:" + str(mt5.last_error()))
            return None
        rates = pd.DataFrame(mt5rates)
        logging.info("Получено " + str(len(rates)) + " котировок")
        return rates

    def compute(self, rates):
        if len(rates) == 0:
            return None
        closes, times = rates["close"], rates["time"]
        last_idx = len(closes)
        shift = self.p.datainfo._in_size() + 1
        results = []
        input_data = []
        for i in range(shift, last_idx):
            x = closes[i - shift : i].to_numpy()
            input_data.append(x)
        output_data = self.p.predict(input_data)
        if output_data is None:
            return None
        for i in range(shift, last_idx):
            plow, phigh, prob = output_data[i - shift]
            rdate = int(times[i - 1])
            rprice = closes[i - 1]
            pmodel = self.p.name
            pdate = int(rdate + 60 * 5 * self.p.datainfo.future)  # секунды*M5*future
            db_row = (
                rdate,
                round(rprice, 6),
                pmodel,
                pdate,
                round(rprice + plow, 6),
                round(rprice + phigh, 6),
                round(prob, 6),
            )
            results.append(db_row)
        logging.info("Вычисления завершены. Получено " + str(len(results)) + " строк")
        return results

    def calc_old(self):
        from_date = self.initialdate
        date = dbcommon.db_get_lowdate(self.db)
        delta = dt.timedelta(minutes=(self.p.datainfo._in_size() + 1) * 5)
        if not date is None:
            from_date = date - delta
        rates = self.get_rates_from_date(from_date)
        if rates is None:
            logging.error("Отсутствуют новые котировки")
            return
        results = self.compute(rates)
        if results is None:
            return
        print("Вычислено %d прогнозов" % len(results))
        logging.info("Вычислено %d прогнозов" % len(results))
        if not results is None:
            dbcommon.db_replace(self.db, results)

    def start(self):
        dtimer = DelayTimer(self.delay)
        self.calc_old()  # обновление данных начиная с даты
        while True:
            if not dtimer.elapsed():
                continue
            rates = self.get_rates(self.p.datainfo._in_size() + 1)
            if rates is None:
                logging.info("Отсутствуют новые котировки")
                continue
            results = self.compute(rates)
            if not results is None:
                dbcommon.db_replace(self.db, results)


DEBUG = False
